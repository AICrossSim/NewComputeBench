import logging

import torch
from torchtitan.models.llama.model import Attention as TTLlamaAttention
from torchtitan.models.llama.model import Transformer as TTLlamaTransformer
from transformers import LlamaForCausalLM as HFLlamaForCausalLM
from ..utils.torch_module import set_layer_by_name, get_layer_name
from ..utils.deps import all_packages_are_available
from ..utils.torch_module import TransformConfigManager
from .cim_layer import CIMLinear, CIMConv2d

def transform_hf_model(model: HFLlamaForCausalLM, config_manager: TransformConfigManager):
    assert isinstance(model, HFLlamaForCausalLM), "Model is not a HuggingFace transformer model."

    transformed_layers = []
    for decoder_layer in model.model.layers:
        attn = decoder_layer.self_attn
        attn_name = get_layer_name(model, attn)
        ot_config = config_manager.get_layer_config(get_layer_name(model, attn))
        if ot_config is None:
            continue
        new_attn = HFOpticalTransformerLlamaAttention.from_pretrained(attn=attn, **ot_config)
        set_layer_by_name(model, attn_name, new_attn)
        transformed_layers.append([attn_name, config_manager.get_layer_config_entry(attn_name)])

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        module: torch.nn.Linear
        ot_config = config_manager.get_layer_config(name)
        if ot_config is None:
            continue
        new_module = OpticalTransformerLinear.from_linear(linear=module, **ot_config)
        set_layer_by_name(model, name, new_module)
        transformed_layers.append([name, config_manager.get_layer_config_entry(name)])
    return transformed_layers

def make_transform_histogram(transformed_layers: list[list[str, str]]) -> dict[str, dict[str, int | list[str]]]:

    patterns = set(layer[1] for layer in transformed_layers)
    histogram = {pattern: {"count": 0, "layers": []} for pattern in patterns}
    for layer, pattern in transformed_layers:
        histogram[pattern]["count"] += 1
        histogram[pattern]["layers"].append(layer)
    histogram["total"] = {"layer count": len(transformed_layers), "pattern count": len(patterns)}
    return histogram
