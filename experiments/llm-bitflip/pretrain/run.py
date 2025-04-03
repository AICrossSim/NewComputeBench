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
from aixsim_models.llm.utils import convert_torch_to_hf
from aixsim_models.bitflip.pretrainer import pretrain
from aixsim_models.bitflip.arg_manager import ArgRandomBitFlipTransform
from aixsim_models.bitflip.arg_manager import (
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
from aixsim_models.bitflip.profiler import profile_stats_hf

register_model_configs()
register_pretrain_dataset()


def generate_pretrain_cfg(
    transform_config: Path,
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
    seq_len: int = 2048,
):
    """
    Generate a configuration for pre-training a language model.

    This function creates a complete configuration for pre-training either an AIXSim or LLaMA model,
    including settings for distributed training, optimization, checkpointing, and more.

    Args:
        model_arch (Literal["aixsim", "llama"]): Architecture of the model to train. Defaults to "aixsim".
        model_flavor (str): Model size/variant (e.g., "60M"). Defaults to "60M".
        tokenizer_path (str): Path to the tokenizer. Defaults to "HuggingFaceTB/cosmo2-tokenizer".
        batch_size (int): Training batch size per device. Defaults to 8.
        data_parallel_replicate_degree (int): Number of data parallel replications. Defaults to 1.
        data_parallel_shard_degree (int): Degree of data parallel sharding (-1 for auto). Defaults to -1.
        tensor_parallel_degree (int): Degree of tensor parallelism. Defaults to 1.
        mixed_precision_param (Literal["bfloat16", "float32"]): Parameter precision type. Defaults to "bfloat16".
        token_num_scale (float): Scale factor for total training tokens. Defaults to 22.0.
        compile (bool): Whether to compile the model. Defaults to False.
        learning_rate (float): Training learning rate. Defaults to 1e-4.
        seed (int): Random seed for reproducibility. Defaults to 42.
        save_path (Optional[str]): Path to save the config file. Defaults to None.
        keep_last_k_ckpts (int): Number of latest checkpoints to keep. Defaults to 3.
        seq_len (int): Sequence length for training. Defaults to 2048.

    Returns:
        None. Saves the configuration to a YAML file at the specified path.

    Notes:
        - The total number of training tokens is calculated as token_num_scale * number of model parameters
        - The effective batch size is batch_size * data_parallel_replicate_degree * data_parallel_shard_degree
        - The number of training steps is computed based on total tokens and effective batch size
        - Configuration is saved in YAML format with timestamp-based checkpoint folders
    """
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
        transform=ArgRandomBitFlipTransform(**transform_config),
    )

    if save_path is None:
        save_path = f"configs/{model_arch}-{model_flavor}.yaml"

    cfg_dict = pretrain_args.to_dict()
    cfg_dict_ = {f"{k}_args": v for k, v in cfg_dict.items()}
    with open(save_path, "w") as f:
        yaml.safe_dump(cfg_dict_, f)
    print(f"Config saved to {save_path}")


def pt_eval_ppl(
    model_arch: Literal["aixsim", "llama"],
    model_flavor: str,
    checkpoint_path: Path,
    transform_config: ArgRandomBitFlipTransform,
    tokenizer_path: str = "HuggingFaceTB/cosmo2-tokenizer",
    dataset_name: str = "fineweb",
    dataset_subset: str = "HuggingFaceFW/fineweb",
    batch_size: int = 32,
    num_batches: int = 32,
    seq_len: int = 2048,
):
    from pprint import pformat
    from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config
    from aixsim_models.llm.tokenizer import build_tokenizer
    from aixsim_models.bitflip.transform import transform_model, TransformConfigManager, make_transform_histogram

    transform_config_manager = TransformConfigManager(
        layer_name_to_config=transform_config.layer_name_to_config,
        use_regex=transform_config.use_regex,
    )
    tokenizer = build_tokenizer(model_name_to_tokenizer[model_arch], tokenizer_path)

    model_config = models_config[model_arch][model_flavor]
    model_config.vocab_size = tokenizer.n_words
    model_config.max_seq_len = seq_len
    model_cls = model_name_to_cls[model_arch]
    model = model_cls.from_model_args(model_config)
    transform_model(model, config_manager=transform_config_manager)
    # transform_histogram = make_transform_histogram(replaced_layers)
    # print(f"Transformed model with the following layers:\n{pformat(transform_histogram)}")

    ppl = pt_evaluate_ppl(
        model_arch=model_arch,
        model_flavor=model_flavor,
        tokenizer_path=tokenizer_path,
        checkpoint_path=checkpoint_path,
        model=model,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        dataset_subset=dataset_subset,
        batch_size=batch_size,
        num_batches=num_batches,
        seq_len=seq_len,
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    set_logging_verbosity("INFO")

    cli_map = {
        "generate-cfg": generate_pretrain_cfg,
        "pretrain": pretrain,
        "eval": {
            "pt-ppl": pt_eval_ppl,
            "hf-ppl": hf_check_ppl,
            "hf-lm-eval": hf_lm_eval,
        },
        "profile-hf": profile_stats_hf,
        "convert-ckpt": {
            "pt2hf": convert_torch_to_hf,
        },
    }

    CLI(cli_map)
