from pathlib import Path
from typing import Optional, Union, Literal
import re
import yaml
from dataclasses import dataclass, asdict, field

from tabulate import tabulate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jsonargparse import CLI

from mase_triton.random_bitflip.layers import RandomBitFlipLinear
from mase_triton.utils.torch_module import set_layer_by_name
from aixsim_models.llm.evaluator import hf_lm_eval, hf_generate

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


class RandomBitFlipConfigManager:
    """
    Random bit flip configuration manager.
    The configuration is a dict mapping layer names to configs.
    Note that the keys in the dict are regex patterns.

    Example of layer_name_to_config:

    ```python
    {
        "default": {
            "x_p_exp": 0.5**10,
            "x_p_frac": 0.5**6,
            "x_zero_out_t": 10,
            "w_p_exp": None,
            "w_p_frac": 0.5**8,
            "w_zero_out_t": None,
        },
        "lm_head": None,
    }
    ```
    """

    def __init__(self, layer_name_to_config: dict[str, dict]):
        self.layer_name_to_config = layer_name_to_config

    @staticmethod
    def find_matched_pattern(layer_name: str, patterns: list[str]) -> str | None:
        for pattern in patterns:
            if re.fullmatch(pattern, layer_name):
                return pattern
        if "default" in patterns:
            return "default"
        return None

    def get_layer_config_entry(self, layer_name: str) -> str:
        matched_pattern = self.find_matched_pattern(layer_name, self.layer_name_to_config.keys())
        if matched_pattern is None:
            raise ValueError(f"No matched pattern for layer {layer_name} and no default")
        return matched_pattern

    def get_layer_config(self, layer_name: str) -> dict | None:
        """Return the config for the layer with the given name. The config is a dict, or None if skip."""
        matched_entry = self.get_layer_config_entry(layer_name)
        return self.layer_name_to_config[matched_entry]


def flip_bits_in_linear(model: torch.nn.Module, config_manager: RandomBitFlipConfigManager) -> list[tuple[str, str]]:
    replaced_layers = []
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            layer_cfg = config_manager.get_layer_config(name)
            if layer_cfg is None:
                continue
            new_layer = RandomBitFlipLinear.from_linear(layer, **layer_cfg)
            set_layer_by_name(model, name, new_layer)
            replaced_layers.append((name, config_manager.get_layer_config_entry(name)))
    return replaced_layers


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
    """Randomly flip bit in linear layers of a model and evaluate it."""
    # fmt: off
    # 2^-6 = 0.015625, 2^-13 = 0.0001220703125

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

    bitflip_config = RandomBitFlipConfigManager(bitflip_config)
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=getattr(torch, dtype)).eval()
    replaced_layers = flip_bits_in_linear(model, bitflip_config)
    print(f"Replaced layers:\n{tabulate(replaced_layers, headers=['Layer', 'Config'])}")

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
    """Randomly flip bit in linear layers of a model and evaluate it."""
    # fmt: off
    # 2^-6 = 0.015625, 2^-13 = 0.0001220703125

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

    bitflip_config = RandomBitFlipConfigManager(bitflip_config)
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=getattr(torch, dtype)).eval()
    replaced_layers = flip_bits_in_linear(model, bitflip_config)
    print(f"Replaced layers:\n{tabulate(replaced_layers, headers=['Layer', 'Config'])}")

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
