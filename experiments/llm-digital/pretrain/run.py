from typing import Literal, Optional
import datetime
import yaml
import math

from aixsim_models.llm.profile import profile_num_params, estimate_memory
from aixsim_models.llm import register_model_configs, register_pretrain_dataset
from aixsim_models.utils.download import (
    download_dataset,
    download_tiktoken_tokenizer,
    download_hf_tokenizer,
)
from aixsim_models.utils.convert_ckpt import dcp_to_torch, torch_to_dcp
from aixsim_models.utils.logging import set_logging_verbosity
from aixsim_models.llm.pretrain import pretrain
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

register_model_configs()
register_pretrain_dataset()


def generate_pretrain_cfg(
    model_arch: Literal["aixsim", "llama"] = "aixsim",
    model_flavor: str = "60M",
    tokenizer_path: str = "HuggingFaceTB/cosmo2-tokenizer",
    batch_size: int = 8,
    data_parallel_replicate_degree: int = 1,
    data_parallel_shard_degree: int = -1,
    tensor_parallel_degree: int = 1,
    mixed_precision_param: Literal["bfloat16", "float32"] = "bfloat16",
    token_num_scale: float = 22.0,
    compile: bool = False,
    learning_rate: float = 1e-4,
    seed: int = 42,
    save_path: Optional[str] = None,
    keep_last_k_ckpts: int = 3,
):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    num_params = profile_num_params(
        model_arch=model_arch,
        model_flavor=model_flavor,
        tokenizer_path=tokenizer_path,
        exclude_embedding=True,
        silent=True,
    )
    num_tokens = token_num_scale * num_params
    effective_batch_size = (
        batch_size * data_parallel_replicate_degree * data_parallel_shard_degree
    )
    num_steps = math.ceil(num_tokens / effective_batch_size)

    print(f"Estimated number of tokens: {num_tokens}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Estimated number of steps: {num_steps}")

    pretrain_args = PreTrainArgs(
        job=ArgJob(
            dump_folder=f"outputs/checkpoints/{model_arch}-{model_flavor}",
            description=f"Pretrain {model_arch} {model_flavor}",
        ),
        profiling=ArgProfiling(),
        metrics=ArgMetrics(enable_tensorboard=False, enable_wandb=True),
        model=ArgModel(
            name=model_arch, flavor=model_flavor, tokenizer_path=tokenizer_path
        ),
        optimizer=ArgOptimizer(lr=learning_rate),
        training=ArgTraining(
            dataset="fineweb",
            dataset_path="HuggingFaceFW/fineweb",
            batch_size=batch_size,
            warmup_steps=int(0.2 * num_steps),
            steps=num_steps,
            data_parallel_replicate_degree=data_parallel_replicate_degree,
            data_parallel_shard_degree=data_parallel_shard_degree,
            tensor_parallel_degree=tensor_parallel_degree,
            mixed_precision_param=mixed_precision_param,
            compile=compile,
            seed=seed,
        ),
        experimental=ArgExperimental(),
        checkpoint=ArgCheckpoint(
            enable_checkpoint=True,
            folder=f"{timestamp}",
            keep_latest_k=keep_last_k_ckpts,
        ),
        activation_checkpoint=ArgActivationCheckpoint(),
        float8=ArgFloat8(),
        comm=ArgComm(),
        memory_estimation=ArgMemoryEstimation(),
    )

    if save_path is None:
        save_path = f"configs/{model_arch}-{model_flavor}.yaml"

    cfg_dict = pretrain_args.to_dict()
    cfg_dict_ = {f"{k}_args": v for k, v in cfg_dict.items()}
    with open(save_path, "w") as f:
        yaml.safe_dump(cfg_dict_, f)
    print(f"Config saved to {save_path}")


if __name__ == "__main__":
    from jsonargparse import CLI

    set_logging_verbosity("INFO")

    cli_map = {
        "download-tiktoken-tokenizer": download_tiktoken_tokenizer,
        "download-hf-tokenizer": download_hf_tokenizer,
        "download-dataset": download_dataset,
        "count-params": profile_num_params,
        "estimate-mem": estimate_memory,
        "pretrain": pretrain,
        "convert-ckpt": {
            "dcp2torch": dcp_to_torch,
        },
        "generate-cfg": generate_pretrain_cfg,
    }

    CLI(cli_map)
