import os
import time
from datetime import timedelta
from typing import Optional, Literal

import torch

from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan import utils
from torchtitan.checkpoint import CheckpointManager, TrainState

# from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_hf_data_loader, build_tokenizer
from torchtitan.float8 import Float8Handler
from torchtitan.logging import init_logger, logger
from torchtitan.metrics import build_device_memory_monitor, build_metric_logger
from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config
from torchtitan.optimizer import build_lr_schedulers, build_optimizers
from torchtitan.parallelisms import (
    models_parallelize_fns,
    models_pipelining_fns,
    ParallelDims,
)
from torchtitan.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
from torchtitan.utils import device_module, device_type

from aixsim_models.llm import build_tokenizer
from aixsim_models.llm.arg_manager import (
    ArgJob,
    ArgProfiling,
    ArgMetrics,
    ArgModel,
    ArgOptimizer,
    ArgTraining,
    ArgExperimental,
    ArgCheckpoint,
    ArgActivationCheckpoint,
    ArgFloat8,
    ArgComm,
    ArgMemoryEstimation,
    PreTrainArgs,
)


@record
def pretrain(
    job_args: Optional[ArgJob] = ArgJob(),
    profiling_args: Optional[ArgProfiling] = ArgProfiling(),
    metrics_args: Optional[ArgMetrics] = ArgMetrics(),
    model_args: Optional[ArgModel] = ArgModel(),
    optimizer_args: Optional[ArgOptimizer] = ArgOptimizer(),
    training_args: Optional[ArgTraining] = ArgTraining(),
    experimental_args: Optional[ArgExperimental] = ArgExperimental(),
    checkpoint_args: Optional[ArgCheckpoint] = ArgCheckpoint(),
    activation_checkpoint_args: Optional[
        ArgActivationCheckpoint
    ] = ArgActivationCheckpoint(),
    float8_args: Optional[ArgFloat8] = ArgFloat8(),
    comm_args: Optional[ArgComm] = ArgComm(),
    memory_estimation_args: Optional[ArgMemoryEstimation] = ArgMemoryEstimation(),
):
    args = PreTrainArgs(
        job=job_args,
        profiling=profiling_args,
        metrics=metrics_args,
        model=model_args,
        optimizer=optimizer_args,
        training=training_args,
        experimental=experimental_args,
        checkpoint=checkpoint_args,
        activation_checkpoint=activation_checkpoint_args,
        float8=float8_args,
        comm=comm_args,
        memory_estimation=memory_estimation_args,
    )

    init_logger()
    logger.info(f"Starting job: {args.job.description}")

    if args.job.print_args:
        logger.info(f"Running with args: {args.to_dict()}")

    # used for colorful printing
    color = utils.NoColor if args.metrics.disable_color_printing else utils.Color

    # take control of garbage collection to avoid stragglers
    gc_handler = utils.GarbageCollection(gc_freq=args.training.gc_freq)

    # init distributed
    world_size = int(os.environ["WORLD_SIZE"])
    parallel_dims = ParallelDims(
        dp_shard=args.training.data_parallel_shard_degree,
        dp_replicate=args.training.data_parallel_replicate_degree,
        cp=args.experimental.context_parallel_degree,
        tp=args.training.tensor_parallel_degree,
        pp=args.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=not args.training.disable_loss_parallel,
    )
    device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
    device_module.set_device(device)
    utils.init_distributed(args)
    # initialize device memory monitor and get peak flops for MFU calculation
    device_memory_monitor = build_device_memory_monitor()
    gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type=device_type)
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    if parallel_dims.pp_enabled:
        pp_mesh = world_mesh["pp"]

    # Set random seed, and maybe enable deterministic mode (mainly for debugging, expect perf loss)
    utils.set_determinism(
        world_mesh,
        device,
        args.training.seed,
        args.training.deterministic,
    )

    # build tokenizer
    tokenizer = build_tokenizer(
        model_name_to_tokenizer[args.model.name], args.model.tokenizer_path
    )
    # build dataloader
    data_loader = build_hf_data_loader(
        args.training.dataset,
        args.training.dataset_path,
        tokenizer,
        args.training.batch_size,
        args.training.seq_len,
        dp_degree,
        dp_rank,
    )
    # build model
    model_name = args.model.name
    model_config = models_config[model_name][args.model.flavor]
    model = build_meta_model(
        model_name=args.model.name,
        model_flavor=args.model.flavor,
        model_config=model_config,
        norm_type=args.model.norm_type,
        n_words=tokenizer.n_words,
        seq_len=args.training.seq_len,
    )

    # a no-op hander if float8 is not enabled
    float8_handler = Float8Handler(args, parallel_dims)
    # swap to Float8Linear based on float8 configs
    float8_handler.convert_to_float8_training(model)

    # log model size
    num_flop_per_token = count_params(
        model_name=model_name,
        model_flavor=args.model.flavor,
        model_config=model_config,
        model=model,
        seq_len=args.training.seq_len,
        color=color,
    )

    # loss function to be shared by Pipeline Parallel and SPMD training
    def loss_fn(pred, labels):
        return torch.nn.functional.cross_entropy(
            pred.flatten(0, 1).float(), labels.flatten(0, 1)
        )

    # TODO: compiling loss function causes CUDA errors, turning off for now
    # if job_config.training.compile:
    #     loss_fn = torch.compile(loss_fn)

    # shard model
    # move sharded model to CPU/GPU and initialize weights via DTensor
    if args.checkpoint.create_seed_checkpoint:
        init_device = "cpu"
        buffer_device = None
    elif args.training.enable_cpu_offload:
        init_device = "cpu"
        buffer_device = device_type
    else:
        init_device = device_type
        buffer_device = None

    # apply parallelisms and initialization
    if parallel_dims.pp_enabled:
        # apply PT-D Pipeline Parallel
        (
            pp_schedule,
            model_parts,
            has_first_stage,
            has_last_stage,
        ) = models_pipelining_fns[model_name](
            model, pp_mesh, parallel_dims, args, device, model_config, loss_fn
        )
        # when PP is enabled, `model` obj is no longer used after this point, model_parts is used instead
        del model

        # For PP with looped schedules, each item in model_parts is one stage-model-chunk.
        # We need to iterate through model_parts to apply SPMD parallelisms, compilation,
        # optimizer, and checkpointing
        for m in model_parts:
            # apply SPMD-style PT-D techniques
            models_parallelize_fns[model_name](m, world_mesh, parallel_dims, args)
            m.to_empty(device=init_device)
            with torch.no_grad():
                m.init_weights(buffer_device=buffer_device)
            m.train()
    else:
        # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
        models_parallelize_fns[model_name](model, world_mesh, parallel_dims, args)
        model.to_empty(device=init_device)
        with torch.no_grad():
            model.init_weights(buffer_device=buffer_device)
        model.train()

        model_parts = [model]

    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )

    # build optimizer after applying parallelisms to the model
    optimizers = build_optimizers(model_parts, args)
    # build lr schedulers
    lr_schedulers = build_lr_schedulers(optimizers.optimizers, args)

    # initialize train state, which tracks training progress
    train_state = TrainState()

    # load initial checkpoint
    checkpoint = CheckpointManager(
        dataloader=data_loader,
        model_parts=model_parts,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        states={"train_state": train_state},
        job_config=args,
    )

    if args.checkpoint.create_seed_checkpoint:
        assert (
            world_size == 1
        ), "Must create seed checkpoint using a single device, to disable sharding"
        assert (
            args.checkpoint.enable_checkpoint
        ), "Must enable checkpointing when creating a seed checkpoint"
        checkpoint.save(curr_step=0, force=True)
        logger.info("Created seed checkpoint")
        return

    checkpoint.load(step=args.checkpoint.load_step)
    metric_logger = build_metric_logger(args, parallel_dims)

    # plot losses loaded from checkpoint (if any) to TensorBoard
    # NOTE: Loss info after the last log step before checkpoint saving will not be ploted.
    #       This can be avoided by setting checkpoint.interval to be a multiple of metrics.log_freq
    if train_state.step > 0:
        for idx, step in enumerate(train_state.log_steps):
            metrics = {
                "loss_metrics/global_avg_loss": train_state.global_avg_losses[idx],
                "loss_metrics/global_max_loss": train_state.global_max_losses[idx],
            }
            metric_logger.log(metrics, step=step)

    data_iterator = iter(data_loader)

    train_context = utils.get_train_context(
        parallel_dims.loss_parallel_enabled,
        args.experimental.enable_compiled_autograd,
    )

    try:
        train_loop(
            args=args,
            train_state=train_state,
            dp_degree=dp_degree,
            gc_handler=gc_handler,
            device_memory_monitor=device_memory_monitor,
            checkpoint=checkpoint,
            data_iterator=data_iterator,
            optimizers=optimizers,
            world_mesh=world_mesh,
            model_parts=model_parts,
            parallel_dims=parallel_dims,
            train_context=train_context,
            has_last_stage=has_last_stage if parallel_dims.pp_enabled else None,
            has_first_stage=has_first_stage if parallel_dims.pp_enabled else None,
            pp_schedule=pp_schedule if parallel_dims.pp_enabled else None,
            model=model if not parallel_dims.pp_enabled else None,
            loss_fn=loss_fn,
            pp_mesh=pp_mesh if parallel_dims.pp_enabled else None,
            lr_schedulers=lr_schedulers,
            float8_handler=float8_handler,
            num_flop_per_token=num_flop_per_token,
            gpu_peak_flops=gpu_peak_flops,
            metric_logger=metric_logger,
            color=color,
        )
        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        metric_logger.close()
        torch.distributed.destroy_process_group()
        logger.info("Training completed")
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"OOM error: {e}")
        metric_logger.close()
        torch.distributed.destroy_process_group()
        raise e


def build_meta_model(
    model_name: str,
    model_flavor: str,
    model_config,
    norm_type: Literal["layernorm", "np_layernorm", "rmsnorm"],
    n_words: int,
    seq_len: int,
):
    # build model (using meta init)
    model_cls = model_name_to_cls[model_name]
    # set the model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. vocab size from tokenizer
    # 3. max_seq_len base on inputs
    model_config.norm_type = norm_type
    model_config.vocab_size = n_words
    model_config.max_seq_len = seq_len

    logger.info(f"Building {model_name} {model_flavor} with {model_config}")
    with torch.device("meta"):
        model = model_cls.from_model_args(model_config)

    return model


def count_params(
    model_name, model_flavor, model_config, model, seq_len, color: utils.Color
) -> int:
    # log model size
    model_param_count = utils.get_num_params(model)
    num_flop_per_token = utils.get_num_flop_per_token(
        utils.get_num_params(model, exclude_embedding=True),
        model_config,
        seq_len,
    )
    logger.info(
        f"{color.blue}Model {model_name} {model_flavor} "
        f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
    )
    return num_flop_per_token


def train_loop(
    args,
    train_state,
    dp_degree,
    gc_handler,
    device_memory_monitor,
    checkpoint,
    data_iterator,
    optimizers,
    world_mesh,
    model_parts,
    parallel_dims,
    train_context,
    has_last_stage,
    has_first_stage,
    pp_schedule,
    model,
    loss_fn,
    pp_mesh,
    lr_schedulers,
    float8_handler,
    num_flop_per_token,
    gpu_peak_flops,
    metric_logger,
    color,
):
    # variables used to keep info for metrics logging
    ntokens_since_last_log = 0
    data_loading_times = []
    time_last_log = time.perf_counter()
    device_memory_monitor.reset_peak_stats()

    checkpoint.reset()
    # train loop
    logger.info(
        f"Training starts at step {train_state.step + 1}, "
        f"with local batch size {args.training.batch_size}, "
        f"global batch size {args.training.batch_size * dp_degree}, "
        f"sequence length {args.training.seq_len}, "
        f"total steps {args.training.steps} "
        f"(warmup {args.training.warmup_steps})"
    )
    with maybe_enable_profiling(
        args, global_step=train_state.step
    ) as torch_profiler, maybe_enable_memory_snapshot(
        args, global_step=train_state.step
    ) as memory_profiler:
        while train_state.step < args.training.steps:
            train_state.step += 1
            gc_handler.run(train_state.step)

            # get batch
            data_load_start = time.perf_counter()
            batch = next(data_iterator)
            input_ids, labels = batch
            ntokens_since_last_log += labels.numel()
            data_loading_times.append(time.perf_counter() - data_load_start)

            input_ids = input_ids.to(device_type)
            labels = labels.to(device_type)
            optimizers.zero_grad()

            # apply context parallelism if cp is enabled
            # ensure CP handles the separate freqs_cis buffer for each pp stage
            optional_context_parallel_ctx = (
                utils.create_context_parallel_ctx(
                    cp_mesh=world_mesh["cp"],
                    cp_buffers=[input_ids, labels] + [m.freqs_cis for m in model_parts],
                    cp_seq_dims=[1, 1] + [0 for _ in model_parts],
                    cp_no_restore_buffers={input_ids, labels},
                    cp_rotate_method=args.experimental.context_parallel_rotate_method,
                )
                if parallel_dims.cp_enabled
                else None
            )

            if parallel_dims.pp_enabled:
                # Pipeline Parallel forward / backward inside step() call
                with train_context(optional_context_parallel_ctx):
                    targets, losses = (labels, []) if has_last_stage else (None, None)
                    if has_first_stage:
                        pp_schedule.step(input_ids, target=targets, losses=losses)
                    else:
                        pp_schedule.step(target=targets, losses=losses)

                # accumulate losses across pipeline microbatches
                # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
                loss = (
                    torch.mean(torch.stack(losses)).to(device)
                    if has_last_stage
                    else torch.tensor([-1.0], device=device)
                )
            else:
                # Non-PP forward / backward
                with train_context(optional_context_parallel_ctx):
                    pred = model(input_ids)
                    loss = loss_fn(pred, labels)
                    # pred.shape=(bs, seq_len, vocab_size)
                    # need to free to before bwd to avoid peaking memory
                    del pred
                    loss.backward()

            # clip gradients
            utils.clip_grad_norm_(
                [p for m in model_parts for p in m.parameters()],
                args.training.max_norm,
                foreach=True,
                pp_mesh=pp_mesh if parallel_dims.pp_enabled else None,
            )

            # optimizer step
            checkpoint.maybe_wait_for_staging()
            optimizers.step()
            lr_schedulers.step()

            # calculate float8 dynamic amax/scale for all-parameter for FSDP2
            # it issues a single all-reduce for all parameters at once for better performance
            float8_handler.precompute_float8_dynamic_scale_for_fsdp(model_parts)

            # log metrics
            if train_state.step == 1 or train_state.step % args.metrics.log_freq == 0:
                if (
                    parallel_dims.dp_replicate_enabled
                    or parallel_dims.dp_shard_enabled
                    or parallel_dims.cp_enabled
                ):
                    loss = loss.detach()
                    global_avg_loss, global_max_loss = (
                        utils.dist_mean(loss, world_mesh["dp_cp"]),
                        utils.dist_max(loss, world_mesh["dp_cp"]),
                    )
                else:
                    global_avg_loss = global_max_loss = loss.item()

                # update train state
                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)

                time_delta = time.perf_counter() - time_last_log

                # tokens per second per device, abbreviated as tps
                tps = ntokens_since_last_log / (
                    time_delta * parallel_dims.non_data_parallel_size
                )
                # model FLOPS utilization
                # For its definition and calculation, please refer to the PaLM paper:
                # https://arxiv.org/abs/2204.02311
                mfu = 100 * num_flop_per_token * tps / gpu_peak_flops

                time_end_to_end = time_delta / args.metrics.log_freq
                time_data_loading = sum(data_loading_times) / len(data_loading_times)
                time_data_loading_pct = 100 * sum(data_loading_times) / time_delta

                device_mem_stats = device_memory_monitor.get_peak_stats()

                metrics = {
                    "loss_metrics/global_avg_loss": global_avg_loss,
                    "loss_metrics/global_max_loss": global_max_loss,
                    "throughput(tps)": tps,
                    "mfu(%)": mfu,
                    "time_metrics/end_to_end(s)": time_end_to_end,
                    "time_metrics/data_loading(s)": time_data_loading,
                    "time_metrics/data_loading(%)": time_data_loading_pct,
                    "memory/max_active(GiB)": device_mem_stats.max_active_gib,
                    "memory/max_active(%)": device_mem_stats.max_active_pct,
                    "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
                    "memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
                    "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
                    "memory/num_ooms": device_mem_stats.num_ooms,
                }
                metric_logger.log(metrics, step=train_state.step)

                logger.info(
                    f"{color.cyan}step: {train_state.step:2}  "
                    f"{color.green}loss: {global_avg_loss:7.4f}  "
                    f"{color.yellow}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
                    f"({device_mem_stats.max_reserved_pct:.2f}%)  "
                    f"{color.blue}tps: {round(tps):,}  "
                    f"{color.magenta}mfu: {mfu:.2f}%{color.reset}"
                )

                ntokens_since_last_log = 0
                data_loading_times.clear()
                time_last_log = time.perf_counter()
                device_memory_monitor.reset_peak_stats()

            checkpoint.save(
                train_state.step,
                force=(train_state.step == args.training.steps),
            )

            # signal the profiler that the next profiling step has started
            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step()

            # reduce timeout after first train step for faster signal
            # (assuming lazy init and compilation are finished)
            if train_state.step == 1:
                utils.set_pg_timeouts(
                    timeout=timedelta(seconds=args.comm.train_timeout_seconds),
                    world_mesh=world_mesh,
                )
