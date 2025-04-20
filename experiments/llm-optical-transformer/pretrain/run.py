import sys
from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[3].joinpath("src").as_posix())
from typing import Literal, Optional
import datetime
import yaml
import math
from pathlib import Path

import torch
from aixsim_models.llm.profiler import profile_num_params
from aixsim_models.llm import register_model_configs, register_pretrain_dataset
from aixsim_models.utils.logging import set_logging_verbosity

from aixsim_models.llm.evaluator import pt_evaluate_ppl, hf_check_ppl, hf_lm_eval
from aixsim_models.llm.utils import convert_torch_to_hf, convert_hf_to_torch
from aixsim_models.optical_compute.optical_transformer.pretrainer import pretrain
from aixsim_models.optical_compute.optical_transformer.arg_manager import (
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
    ArgOpticalTransformerTransform,
)

register_model_configs()
register_pretrain_dataset()


def generate_pretrain_cfg(
    transform_config: Path,
    model_arch: Literal["aixsim", "llama"] = "aixsim",
    model_flavor: str = "debug",
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
    seq_len: int = 2048,
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
    effective_batch_size = batch_size * data_parallel_replicate_degree * abs(data_parallel_shard_degree)
    num_steps = math.ceil(num_tokens / (effective_batch_size * seq_len))

    print(
        f"Estimated number of tokens = token_num_scale * num_params = {token_num_scale} * {num_params} = {num_tokens}"
    )
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Estimated number of steps: {num_steps}")

    assert transform_config.exists(), f"Transform config file {transform_config} does not exist"
    with open(transform_config, "r") as f:
        transform_config = yaml.safe_load(f)

    pretrain_args = PreTrainArgs(
        job=ArgJob(
            dump_folder=f"outputs/checkpoints/{model_arch}-{model_flavor}",
            description=f"Pretrain {model_arch} {model_flavor}",
        ),
        profiling=ArgProfiling(),
        metrics=ArgMetrics(enable_tensorboard=False, enable_wandb=True),
        model=ArgModel(name=model_arch, flavor=model_flavor, tokenizer_path=tokenizer_path),
        optimizer=ArgOptimizer(lr=learning_rate),
        training=ArgTraining(
            dataset="fineweb-edu",
            dataset_path="HuggingFaceFW/fineweb-edu",
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
        transform=ArgOpticalTransformerTransform(**transform_config),
    )

    if save_path is None:
        save_path = f"configs/{model_arch}-{model_flavor}.yaml"

    cfg_dict = pretrain_args.to_dict()
    cfg_dict_ = {f"{k}_args": v for k, v in cfg_dict.items()}
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        yaml.safe_dump(cfg_dict_, f)
    print(f"Config saved to {save_path}")


if __name__ == "__main__":
    from jsonargparse import CLI

    set_logging_verbosity("INFO")

    cli_map = {
        "generate-cfg": generate_pretrain_cfg,
        "pretrain": pretrain,
        "eval": {
            "hf-ppl": hf_check_ppl,
            "pt-ppl": pt_evaluate_ppl,
        },
        "convert-ckpt": {
            "hf-to-pt": convert_hf_to_torch,
            "pt-to-hf": convert_torch_to_hf,
        },
    }

    CLI(cli_map)
