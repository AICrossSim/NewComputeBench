import logging
from typing import Optional
import contextlib
import gc
import os

import torch
from torch._guards import active_fake_mode
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.fsdp2_mem_tracker import FSDPMemTracker
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config
from torchtitan import utils
from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_tokenizer
from torchtitan.float8 import Float8Handler
from torchtitan.optimizer import build_lr_schedulers, build_optimizers
from torchtitan.parallelisms import models_parallelize_fns, ParallelDims

from .tokenizer import build_tokenizer
from .arg_manager import (
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

logger = logging.getLogger(__name__)


def beatify_num_params(num_params: int) -> str:
    if num_params < 1e6:
        return f"{num_params / 1e3:.0f}K"
    elif num_params < 1e9:
        return f"{num_params / 1e6:.1f}M"
    elif num_params < 1e12:
        return f"{num_params / 1e9:.1f}B"
    else:
        return f"{num_params / 1e12:.1f}T"


def profile_num_params(
    model_arch: str,
    model_flavor: str,
    tokenizer_path: str,
    exclude_embedding: Optional[bool] = True,
    silent: Optional[bool] = False,
):
    """
    Profiles the number of parameters in a specified model architecture and flavor.

    Args:
        model_arch (str): The architecture of the model (e.g., 'aixsim').
        model_flavor (str): The specific flavor or variant of the model (e.g., '60M', '200M').
        tokenizer_path (str): The name/path to the tokenizer files (e.g., 'HuggingFaceTB/cosmo2-tokenizer').
        exclude_embedding (Optional[bool], optional): Whether to exclude the embedding parameters from the count. Defaults to True.
        silent (Optional[bool], optional): If True, suppresses logging output. Defaults to False.

    Returns:
        int: The number of parameters in the model.
    """
    tokenizer_type = model_name_to_tokenizer[model_arch]
    tokenizer = build_tokenizer(tokenizer_type, tokenizer_path)
    model_cls = model_name_to_cls[model_arch]
    model_cfg = models_config[model_arch][model_flavor]
    model_cfg.vocab_size = tokenizer.n_words

    with torch.device("meta"):
        model = model_cls.from_model_args(model_cfg)

    num_params = utils.get_num_params(model=model, exclude_embedding=exclude_embedding)
    if not silent:
        logger.info(
            f"Model {model_arch}-{model_flavor} has {beatify_num_params(num_params)} parameters"
        )
    return num_params


def estimate_memory(
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
    """
    Estimate the memory usage of a model during training.
    """
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
    logger.info("Estimating memory usage...")
    gc.disable()
    gc.collect(1)

    # Get the world size
    world_size = int(os.environ["WORLD_SIZE"])

    if args.model.norm_type == "compiled_rmsnorm":
        logger.info("Compiled RMSNorm is not supported yet. Switching to RMSNorm.")
        args.model.norm_type = "rmsnorm"

    if args.training.compile or args.experimental.enable_compiled_autograd:
        logger.info("Compile mode is not supported yet. Switching to eager mode.")
        args.training.compile = False
        args.experimental.enable_compiled_autograd = False

    parallel_dims = ParallelDims(
        dp_shard=args.training.data_parallel_shard_degree,
        dp_replicate=args.training.data_parallel_replicate_degree,
        cp=args.experimental.context_parallel_degree,
        tp=args.training.tensor_parallel_degree,
        pp=args.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=not args.training.disable_loss_parallel,
    )

    # only FSDP and HSDP are supported
    if (
        (parallel_dims.dp_replicate_enabled and not parallel_dims.dp_shard_enabled)
        or parallel_dims.tp_enabled
        or parallel_dims.pp_enabled
        or parallel_dims.cp_enabled
    ):
        logger.warning("DDP, TP, PP, CP are not supported yet.")
        return
    if not parallel_dims.dp_shard_enabled:
        logger.warning("FSDP or HSDP is not enabled. Skipping memory estimation.")
        return

    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)

    # init fake pg
    store = FakeStore()
    torch.distributed.init_process_group(
        "fake", rank=int(os.environ["LOCAL_RANK"]), world_size=world_size, store=store
    )

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type="cuda")

    model_name = args.model.name

    # build tokenizer
    tokenizer_type = model_name_to_tokenizer[model_name]
    tokenizer = build_tokenizer(tokenizer_type, args.model.tokenizer_path)

    train_context = utils.get_train_context(
        parallel_dims.loss_parallel_enabled,
        args.experimental.enable_compiled_autograd,
    )

    # loss fn can be shared by pipeline-parallel or non-pp execution
    def loss_fn(pred, labels):
        return torch.nn.functional.cross_entropy(
            pred.flatten(0, 1).float(), labels.flatten(0, 1)
        )

    # build model (using meta init)
    model_cls = model_name_to_cls[model_name]
    model_config = models_config[model_name][args.model.flavor]
    # set the model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. vocab size from tokenizer
    # 3. max_seq_len base on inputs
    model_config.norm_type = args.model.norm_type
    model_config.vocab_size = tokenizer.n_words
    model_config.max_seq_len = args.training.seq_len

    with (
        FakeTensorMode()
        if not args.memory_estimation.disable_fake_mode
        else contextlib.nullcontext()
    ):

        logger.info(f"Building {model_name} {args.model.flavor} with {model_config}")
        with torch.device("meta"):
            model = model_cls.from_model_args(model_config)

        # a no-op hander if float8 is not enabled
        float8_handler = Float8Handler(args, parallel_dims)
        # swap to Float8Linear based on float8 configs
        float8_handler.convert_to_float8_training(model)

        # apply PT-D DP/TP parallelisms and activation checkpointing
        models_parallelize_fns[model_name](model, world_mesh, parallel_dims, args)

        model.to_empty(device="cuda")
        if not active_fake_mode():
            model.init_weights()
        model.train()

        # build optimizer after applying parallelisms to the model
        optimizers = build_optimizers([model], args)
        lr_schedulers = build_lr_schedulers(optimizers.optimizers, args)

        logger.info(f"Vocab size: {model_config.vocab_size}")
        # Create a dummy batch instead of loading from a dataset
        batch = (
            torch.randint(
                0,
                model_config.vocab_size,
                (args.training.batch_size, model_config.max_seq_len),
                device="cuda",
            ),
            torch.randint(
                0,
                model_config.vocab_size,
                (args.training.batch_size, model_config.max_seq_len),
                device="cuda",
            ),
        )
        fsdp_memtracker = FSDPMemTracker(mod=model, optm=optimizers.optimizers[0])
        fsdp_memtracker.track_inputs(batch)

        with fsdp_memtracker:
            for iter_idx in range(2):
                input_ids, labels = batch
                # train step
                with train_context():
                    pred = model(input_ids)
                    loss = loss_fn(pred, labels)
                    del pred
                    loss.backward()

                # clip gradients
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.training.max_norm, foreach=True
                )
                # optimizer step
                optimizers.step()
                lr_schedulers.step()
                # calculate float8 dynamic amax/scale for all-parameter for FSDP2
                # it issues a single all-reduce for all parameters at once for better performance
                float8_handler.precompute_float8_dynamic_scale_for_fsdp(model)
                optimizers.zero_grad()
                print(f"Peak Memory at iter: {iter_idx}")
                fsdp_memtracker.display_snapshot("peak", units="MiB", tabulate=True)
                if iter_idx == 0:
                    fsdp_memtracker.reset_mod_stats()  # iter 0 does not have optimizer state
                gc.collect(1)

        fsdp_memtracker.display_modulewise_snapshots(
            depth=3, units="MiB", tabulate=True
        )
        mem_stats = torch.cuda.memory_stats()
        peak_active = mem_stats["active_bytes.all.peak"]
        peak_reserved = mem_stats["reserved_bytes.all.peak"]
        num_retries = mem_stats["num_alloc_retries"]
        dev = torch.device(torch.cuda.current_device())
        tracker_peak = fsdp_memtracker.get_tracker_snapshot("peak")[dev]["Total"]
        gib = 1024**3
        print(
            f"peak active: {peak_active / gib} GiB | peak reserved:"
            f" {peak_reserved / gib} GiB | num_retries: {num_retries}"
        )
        print(f"Tracker Max: {tracker_peak / gib} GiB")
        if args.memory_estimation.disable_fake_mode and peak_active > 0:
            print(f"Tracker Accuracy: {tracker_peak/peak_active}")
        gc.enable()


if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()
    try:
        estimate_memory(config)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
