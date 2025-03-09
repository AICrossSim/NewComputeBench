from pathlib import Path
from typing import Optional, Union, Literal
import re
import yaml

from tabulate import tabulate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from jsonargparse import CLI

from mase_triton.random_bitflip.layers import RandomBitFlipDropout, RandomBitFlipLinear
from mase_triton.utils.torch_module import set_layer_by_name

DEFAULT_DTYPE = "float16"
DEFAULT_TASKS = ["wikitext"]


def eval_ori(
    model_name: str,
    dtype: Literal["float32", "float16", "bfloat16"] = DEFAULT_DTYPE,
    tasks: Optional[list[str]] = DEFAULT_TASKS,
    num_fewshot: Optional[int] = None,
    batch_size: Optional[Union[int, str]] = "auto",
    limit: Optional[Union[int, float]] = None,
):
    """Evaluate a pretrained model as baseline."""
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=getattr(torch, dtype)).eval()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

    results = simple_evaluate(
        model=model,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
    )

    if results is not None:
        results.pop("samples")
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))


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


def eval_random_bitflip(
    model_name: str,
    bitflip_config: Union[Literal["a-only-frac", "a-only-both", "w-only", "w-only-frac", "w-only-both"], str, dict],
    dtype: Literal["float32", "float16", "bfloat16"] = DEFAULT_DTYPE,
    tasks: Optional[list[str]] = DEFAULT_TASKS,
    num_fewshot: Optional[int] = None,
    batch_size: Optional[Union[int, str]] = "auto",
    limit: Optional[Union[int, float]] = None,
):
    """Randomly flip bit in linear layers of a model and evaluate it."""
    # fmt: off
    # 2^-6 = 0.015625, 2^-13 = 0.0001220703125
    # flipping in fraction-only does not lead to new NaN values
    P_FRAC = 0.5**6
    P_EXP = 0.5**13
    A_T = 10
    W_T = 1.5
    ACT_ONLY_FRAC_ONLY = dict(default=dict(x_p_exp=None, x_p_frac=P_FRAC, x_zero_out_t=None, w_p_exp=None, w_p_frac=None, w_zero_out_t=None))
    ACT_ONLY_BOTH = dict(default=dict(x_p_exp=0.5**10, x_p_frac=P_FRAC, x_zero_out_t=A_T, w_p_exp=None, w_p_frac=None, w_zero_out_t=None))
    WEIGHT_ONLY_FRAC_ONLY = dict(default=dict(x_p_exp=None, x_p_frac=None, x_zero_out_t=None, w_p_exp=None, w_p_frac=P_FRAC, w_zero_out_t=None))
    WEIGHT_ONLY_BOTH = dict(default=dict(x_p_exp=None, x_p_frac=None, x_zero_out_t=None, w_p_exp=P_EXP, w_p_frac=P_FRAC, w_zero_out_t=W_T))
    DEFAULT_CONFIG_MAP = {"a-only-frac": ACT_ONLY_FRAC_ONLY, "a-only-both": ACT_ONLY_BOTH, "w-only-frac": WEIGHT_ONLY_FRAC_ONLY, "w-only-both": WEIGHT_ONLY_BOTH}
    # fmt: on

    if isinstance(bitflip_config, str):
        if bitflip_config in DEFAULT_CONFIG_MAP:
            bitflip_config = DEFAULT_CONFIG_MAP[bitflip_config]
        else:
            bitflip_config = Path(bitflip_config)
            assert bitflip_config.exists(), f"Bitflip config file {bitflip_config} does not exist"
            with bitflip_config.open("r") as f:
                bitflip_config = yaml.safe_load(f)
    bitflip_config = RandomBitFlipConfigManager(bitflip_config)

    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=getattr(torch, dtype)).eval()
    replaced_layers = flip_bits_in_linear(model, bitflip_config)
    print(f"Replaced layers:\n{tabulate(replaced_layers, headers=['Layer', 'Config'])}")

    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

    results = simple_evaluate(
        model=model,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
    )

    if results is not None:
        results.pop("samples")
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))


if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    cli_map = {
        "eval-ori": eval_ori,
        "eval-bitflip": eval_random_bitflip,
    }
    CLI(cli_map)

    """Sanity check experiments:

    $ python minimal.py eval-ori meta-llama/Llama-3.1-8B
    llama-3.1-8B, fp16, original checkpoint
    | Tasks  |Version|Filter|n-shot|    Metric     |   |Value |   |Stderr|
    |--------|------:|------|-----:|---------------|---|-----:|---|------|
    |wikitext|      2|none  |     0|bits_per_byte  |↓  |0.5375|±  |   N/A|
    |        |       |none  |     0|byte_perplexity|↓  |1.4515|±  |   N/A|
    |        |       |none  |     0|word_perplexity|↓  |7.3330|±  |   N/A|

    $ python minimal.py eval-bitflip meta-llama/Llama-3.1-8B a-only-frac
    llama-3.1-8B, BF16, random bitflip [act-only::frac-only, prob=2^-6=0.015625]
    | Tasks  |Version|Filter|n-shot|    Metric     |   |Value|   |Stderr|
    |--------|------:|------|-----:|---------------|---|----:|---|------|
    |wikitext|      2|none  |     0|bits_per_byte  |↓  |0.544|±  |   N/A|
    |        |       |none  |     0|byte_perplexity|↓  |1.458|±  |   N/A|
    |        |       |none  |     0|word_perplexity|↓  |7.510|±  |   N/A|


    $ python minimal.py eval-bitflip meta-llama/Llama-3.1-8B a-only-both
    llama-3.1-8B, BF16, random bitflip [act-only::both, prob=2^-10=0.0009765625, 2^-6=0.015625, t=10]
    | Tasks  |Version|Filter|n-shot|    Metric     |   | Value  |   |Stderr|
    |--------|------:|------|-----:|---------------|---|--------|---|------|
    |wikitext|      2|none  |     0|bits_per_byte  |↓  |Infinity|±  |   N/A|
    |        |       |none  |     0|byte_perplexity|↓  |Infinity|±  |   N/A|
    |        |       |none  |     0|word_perplexity|↓  |Infinity|±  |   N/A|
    """
