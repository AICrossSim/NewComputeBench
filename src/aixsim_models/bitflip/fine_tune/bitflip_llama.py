import torch
from mase_triton.random_bitflip.layers import RandomBitFlipLinear
from torch import nn
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from ...utils.torch_module import set_layer_by_name
from .bitflip_lora import BitFlipLinearLora


def transform_llama(
    model: LlamaForCausalLM,
    fc_config: dict,
    use_lora: bool,
) -> list[str]:
    """Replace all Linear layers (except lm_head) with bitflip-aware layers.

    Args:
        model: A LlamaForCausalLM model.
        fc_config: Config dict passed to BitFlipLinearLora.from_linear or
            RandomBitFlipLinear.from_linear. When use_lora is True, this should
            include both bitflip params and lora params (r, lora_alpha).
        use_lora: If True, use BitFlipLinearLora; otherwise use RandomBitFlipLinear.

    Returns:
        List of replaced layer names.
    """
    assert isinstance(model, LlamaForCausalLM)
    replaced_layers = []

    for name, layer in model.named_modules():
        if not isinstance(layer, nn.Linear):
            continue

        if "lm_head" in name:
            continue

        if use_lora:
            new_layer = BitFlipLinearLora.from_linear(layer, **fc_config)
        else:
            new_layer = RandomBitFlipLinear.from_linear(layer, **fc_config)

        set_layer_by_name(model, name, new_layer)
        replaced_layers.append(name)

    return replaced_layers
