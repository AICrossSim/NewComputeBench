import sys
from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[3].joinpath("src").as_posix())
from typing import Optional, Union, Literal
import re
import yaml
from dataclasses import dataclass, asdict, field

from tabulate import tabulate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jsonargparse import CLI

from aixsim_models.llm.evaluator import hf_lm_eval, hf_generate
from aixsim_models.bitflip.transform import transform_model, TransformConfigManager

DEFAULT_DTYPE = "float16"
DEFAULT_TASKS = ["wikitext"]


def eval_ori(
    model_name: str = "meta-llama/Llama-3.1-8B",
    dtype: Literal["float32", "float16", "bfloat16"] = DEFAULT_DTYPE,
    tasks: Optional[list[str]] = DEFAULT_TASKS,
    num_fewshot: Optional[int] = None,
    batch_size: Optional[Union[int, str]] = 32,
    limit: Optional[Union[int, float]] = None,
    max_seq_len: Optional[int] = 2048,
    save_dir: Optional[Path] = None,
):
    """Evaluate a pretrained model as baseline."""
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=getattr(torch, dtype)).eval()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    results = hf_lm_eval(
        model=model,
        tokenizer=tokenizer,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
        max_seq_len=max_seq_len,
        save_dir=save_dir,
    )


@dataclass
class DefaultBitFlipConfig:
    x_p_exp: float = field(default=None)
    x_p_frac: float = field(default=None)
    x_zero_out_t: float = field(default=None)
    w_p_exp: float = field(default=None)
    w_p_frac: float = field(default=None)
    w_zero_out_t: float = field(default=None)


def eval_random_bitflip(
    model_name: str = "meta-llama/Llama-3.1-8B",
    bitflip_config: Union[Literal["default"], Path, dict] = "default",
    default_bitflip_config: DefaultBitFlipConfig = DefaultBitFlipConfig(
        x_p_exp=None,
        x_p_frac=None,
        x_zero_out_t=None,
        w_p_exp=None,
        w_p_frac=None,
        w_zero_out_t=None,
    ),
    dtype: Literal["float32", "float16", "bfloat16"] = DEFAULT_DTYPE,
    tasks: Optional[list[str]] = DEFAULT_TASKS,
    num_fewshot: Optional[int] = None,
    batch_size: Optional[Union[int, str]] = 32,
    limit: Optional[Union[int, float]] = None,
    max_seq_len: Optional[int] = 2048,
    save_dir: Optional[Path] = None,
):
    """
    Randomly flip bit in linear layers of a model and evaluate it.

    Args:
        model_name (str): The name of the model to use. Default is "meta-llama/Llama-3.1-8B".
        bitflip_config (Union[Literal["default"], Path, dict]): The config for the bitflip. Default is "default".
        default_bitflip_config (DefaultBitFlipConfig): The default config for the bitflip. Default is DefaultBitFlipConfig.
        dtype (Literal["float32", "float16", "bfloat16"]): The dtype to use. Default is "float16".
        tasks (Optional[list[str]]): The tasks to evaluate. Default is ["wikitext"].
        num_fewshot (Optional[int]): The number of fewshot examples to use. Default is None.
        batch_size (Optional[Union[int, str]]): The batch size to use. Default is 32.
        limit (Optional[Union[int, float]]): The limit to use. Default is None.
        max_seq_len (Optional[int]): The maximum sequence length to use. Default is 2048.
        save_dir (Optional[Path]): The directory to save the results. Default is None.
    """
    # fmt: off

    if isinstance(bitflip_config, str) and bitflip_config == "default":
        print("Using default bitflip config:")
        print(asdict(default_bitflip_config))
        bitflip_config_default = asdict(default_bitflip_config)
        bitflip_config = {"default": bitflip_config_default, "lm_head": None}
    elif isinstance(bitflip_config, Path) or isinstance(bitflip_config, str):
        with bitflip_config.open("r") as f:
            bitflip_config = yaml.safe_load(f)
    else:
        assert isinstance(bitflip_config, dict), f"Invalid bitflip_config: {bitflip_config}"

    bitflip_config = TransformConfigManager(layer_name_to_config=bitflip_config, use_regex=True)
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=getattr(torch, dtype)).eval()
    transform_model(model, config_manager=bitflip_config)
    # replaced_layers = flip_bits_in_linear(model, bitflip_config)
    # print(f"Replaced layers:\n{tabulate(replaced_layers, headers=['Layer', 'Config'])}")

    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    results = hf_lm_eval(
        model=model,
        tokenizer=tokenizer,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
        max_seq_len=max_seq_len,
        save_dir=save_dir,
    )


def bitflip_hf_generate(
    model_name: str = "meta-llama/Llama-3.1-8B",
    bitflip_config: Union[Literal["default"], Path, dict] = "default",
    default_bitflip_config: DefaultBitFlipConfig = DefaultBitFlipConfig(
        x_p_exp=None,
        x_p_frac=None,
        x_zero_out_t=None,
        w_p_exp=None,
        w_p_frac=None,
        w_zero_out_t=None,
    ),
    dtype: Literal["float32", "float16", "bfloat16"] = DEFAULT_DTYPE,
    prompt: str = "London is ",
    max_new_tokens: int = 100,
    seed: int = 42,
    do_sample: bool = True,
    temperature: float = 0.6,
    top_k: int = 50,
    top_p: float = 0.9,
    save_dir: Optional[Path] = None,
):
    """Randomly flip bit in linear layers of a model and evaluate it.

    Args:
        model_name (str): The name of the model to use. Default is "meta-llama/Llama-3.1-8B".
        bitflip_config (Union[Literal["default"], Path, dict]): The config for the bitflip. Default is "default".
        default_bitflip_config (DefaultBitFlipConfig): The default config for the bitflip. Default is DefaultBitFlipConfig.
        dtype (Literal["float32", "float16", "bfloat16"]): The dtype to use. Default is "float16".
        prompt (str): The prompt to use. Default is "London is ".
        max_new_tokens (int): The maximum number of new tokens to generate. Default is 100.
        seed (int): The seed to use. Default is 42.
        do_sample (bool): Whether to sample or not. Default is True.
        temperature (float): The temperature to use. Default is 0.6.
        top_k (int): The top_k to use. Default is 50.
        top_p (float): The top_p to use. Default is 0.9.
        save_dir (Optional[Path]): The directory to save the results. Default is None.
    """
    # fmt: off

    if isinstance(bitflip_config, str) and bitflip_config == "default":
        print("Using default bitflip config:")
        print(asdict(default_bitflip_config))
        bitflip_config_default = asdict(default_bitflip_config)
        bitflip_config = {"default": bitflip_config_default, "lm_head": None}
    elif isinstance(bitflip_config, Path) or isinstance(bitflip_config, str):
        with bitflip_config.open("r") as f:
            bitflip_config = yaml.safe_load(f)
    else:
        assert isinstance(bitflip_config, dict), f"Invalid bitflip_config: {bitflip_config}"

    bitflip_config = TransformConfigManager(layer_name_to_config=bitflip_config, use_regex=True)

    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=getattr(torch, dtype)).eval()
    transform_model(model, config_manager=bitflip_config)
    # print(f"Replaced layers:\n{tabulate(replaced_layers, headers=['Layer', 'Config'])}")

    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    response = hf_generate(prompt=prompt, model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens, dtype=dtype, seed=seed,
                do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "prompt-response.txt", "w") as f:
            f.write(f"Prompt: {prompt}\n\nResponse:\n{response}")


if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    cli_map = {
        "eval-ori": eval_ori,
        "eval-bitflip": eval_random_bitflip,
        "hf-gen": bitflip_hf_generate,
    }
    CLI(cli_map)

    """Sanity check experiments:

    $ python minimal.py eval-ori --model_name meta-llama/Llama-3.1-8B
    | Tasks  |Version|Filter|n-shot|    Metric     |   |Value |   |Stderr|
    |--------|------:|------|-----:|---------------|---|-----:|---|------|
    |wikitext|      2|none  |     0|bits_per_byte  |↓  |0.5588|±  |   N/A|
    |        |       |none  |     0|byte_perplexity|↓  |1.4730|±  |   N/A|
    |        |       |none  |     0|word_perplexity|↓  |7.9336|±  |   N/A|

    $ python minimal.py eval-bitflip --model_name meta-llama/Llama-3.1-8B
    # a-w-both, p_frac=2^-16, p_exp=2^-16, a_t=30, w_t=1.25
    | Tasks  |Version|Filter|n-shot|    Metric     |   |  Value   |   |Stderr|
    |--------|------:|------|-----:|---------------|---|---------:|---|------|
    |wikitext|      2|none  |     0|bits_per_byte  |↓  |    2.5481|±  |   N/A|
    |        |       |none  |     0|byte_perplexity|↓  |    5.8484|±  |   N/A|
    |        |       |none  |     0|word_perplexity|↓  |12638.9580|±  |   N/A|
    """
