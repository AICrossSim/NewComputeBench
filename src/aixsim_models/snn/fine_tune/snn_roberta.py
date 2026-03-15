import logging
import math
from collections import namedtuple
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from copy import deepcopy
import re

from aixsim_models.snn.quantized_layers import (
    RobertaClassificationHeadLSQInteger,
    RobertaIntermediateLSQInteger,
    RobertaOutputLSQInteger,
    RobertaSelfAttentionLSQInteger,
    RobertaSelfOutputLSQInteger,
)
from aixsim_models.snn.snn_layers import (
    RobertaSelfAttentionZIPTF,
    SoftmaxZIPTF,
    LayerNormZIPTF,
    EmbeddingZIPTF,
    LinearUnfoldBias
)
from aixsim_models.snn.quantized_layers.quantizer.LSQ import LSQInteger
from aixsim_models.snn.neuron.st_bifnode import ST_BIFNode
from aixsim_models.utils.torch_module import get_layer_name, set_layer_by_name
from .state_dict_map import (
    match_a_pattern,
    attn_convert,
    lsqinteger_to_st_bif,
)
from transformers.models.roberta.modeling_roberta import (
    RobertaAttention,
    RobertaForSequenceClassification,
    RobertaLayer,
    RobertaSelfAttention,
)

quantized_roberta_module_map = {
    "roberta_self_attention_lsqinteger": RobertaSelfAttentionLSQInteger,
    "roberta_intermediate_lsqinteger": RobertaIntermediateLSQInteger,
    "roberta_output_lsqinteger": RobertaOutputLSQInteger,
    "roberta_classification_head_lsqinteger": RobertaClassificationHeadLSQInteger,
    "roberta_self_output_lsqinteger": RobertaSelfOutputLSQInteger,
}

spiking_varied_module_map = {
    "roberta_self_attention_zip_tf": RobertaSelfAttentionZIPTF,
    "softmax_zip_tf": SoftmaxZIPTF,
    "layernorm_zip_tf": LayerNormZIPTF,
    "embedding_zip_tf": EmbeddingZIPTF,
    "linear_unfold_bias": LinearUnfoldBias,
    "identity": nn.Identity,
}
spiking_neuron_module_map = {
    "st_bif": ST_BIFNode,
}
spiking_module_map = {
    **spiking_varied_module_map,
    **spiking_neuron_module_map,
}

def set_module_by_name(
    model, name, target_module, parent_name=None, current_name=None, parent_model=None
):
    if name == parent_name:
        setattr(parent_model, current_name, target_module)
        return model

    for n, module in model.named_children():
        ## compound module, go inside it
        new_parent_name = n if parent_name is None else f"{parent_name}.{n}"
        set_module_by_name(module, name, target_module, new_parent_name, n, model)
    return model

def weight_replacement(x, y):
    target_state_dict = deepcopy(x.state_dict())
    missing_keys, unexpected_keys = y.load_state_dict(target_state_dict, strict=False)
    if missing_keys:
        logging.warning(
            f"Missing keys when loading state_dict: {missing_keys} from {x} to {y}"
        )
    if unexpected_keys:
        logging.warning(
            f"Unexpected keys when loading state_dict: {unexpected_keys} from {x} to {y}"
        )
    return y

SPECIAL_CONVERT_PATTERNS = {
    (RobertaSelfAttentionLSQInteger, RobertaSelfAttentionZIPTF): attn_convert,
    (LSQInteger, ST_BIFNode): lsqinteger_to_st_bif,
}


def get_roberta_layer_type(module_name: str) -> str:
    """Extract the layer type from a full module name.
    
    Args:
        module_name: Full module name like 'roberta.encoder.layer.0.attention.self'
    
    Returns:
        Layer type like 'roberta_self_attention'
    """
    if "attention.self" in module_name and "output" not in module_name:
        return "roberta_self_attention"
    elif "attention.output" in module_name:
        return "roberta_self_output"
    elif module_name.endswith(".intermediate"):
        return "roberta_intermediate"
    elif module_name.endswith(".output") and "attention" not in module_name:
        return "roberta_output"
    elif module_name == "classifier":
        return "roberta_classification_head"
    else:
        raise ValueError(f"Unknown module type for: {module_name}")


def transform_roberta_by_regex_name(
    network: RobertaForSequenceClassification, pass_args: dict
):
    replaced_layers = []
    n_m = {}

    for n, m in network.named_modules():
        n_m[n] = m

    patterns = list(pass_args.keys())
    for n, m in n_m.items():
        matched_pattern = match_a_pattern(n, patterns)
        if not matched_pattern:
            continue

        quan_config = pass_args[matched_pattern]["config"]
        postfix = quan_config["name"]

        roberta_layer_type = get_roberta_layer_type(n)

        if f"{roberta_layer_type}_{postfix}" in quantized_roberta_module_map:
            roberta_cls = quantized_roberta_module_map[f"{roberta_layer_type}_{postfix}"]
        elif f"{roberta_layer_type}_{postfix}" in spiking_module_map:
            roberta_cls = spiking_module_map[f"{roberta_layer_type}_{postfix}"]
        else:
            raise ValueError(f"Unknown roberta layer type: {roberta_layer_type}_{postfix}")

        new_m = roberta_cls(
            config=network.config,
            q_config=quan_config,
        )

        special_replacement = (type(m), type(new_m)) in SPECIAL_CONVERT_PATTERNS
        if special_replacement:
            new_m = SPECIAL_CONVERT_PATTERNS[(type(m), type(new_m))](m, new_m)
        else:
            new_m = weight_replacement(m, new_m)

        network = set_module_by_name(network, n, new_m)
        replaced_layers.append(n)

    return network, replaced_layers


def transform_roberta_by_type(
    network: RobertaForSequenceClassification, pass_args: dict
):
    replaced_layers = []
    for type_name, conversion_config in pass_args.items():
        n_m = {}
        for n, m in network.named_modules():
            n_m[n] = m

        # SNN only concerns the following types of layers, so we can directly match by type and convert all modules of that type. If there are some specific modules that we don't want to convert, we can add more specific matching rules in the future.
        if type_name == "linear":
            module_type = torch.nn.Linear
        elif type_name == "embedding":
            module_type = torch.nn.Embedding
        elif type_name == "layernorm":
            module_type = torch.nn.LayerNorm
        elif type_name == "relu":
            module_type = torch.nn.ReLU
        elif type_name == "lsqinteger":
            module_type = LSQInteger
        else:
            raise ValueError(f"{type_name} is not supported!")

        is_manual_instantiate = conversion_config.get("manual_instantiate", False)
        conversion_config = conversion_config["config"]
        postfix = conversion_config.pop("name")

        for n, m in n_m.items():
            if isinstance(m, module_type):
                # same across all convert methods
                additional_module_args = (
                    {"config": conversion_config, "network_config": network.config}
                )

                if is_manual_instantiate:
                    new_m = spiking_module_map[postfix](**additional_module_args["config"])
                else:
                    # normal module
                    if isinstance(m, torch.nn.Linear):
                        linear_cls = spiking_module_map[f"linear_{postfix}"]
                        has_bias = not (m.bias is None)
                        new_m = linear_cls(
                            in_features=m.in_features,
                            out_features=m.out_features,
                            bias=has_bias,
                            **additional_module_args["config"],
                        )
                    elif isinstance(m, torch.nn.Embedding):
                        embedding_cls = spiking_module_map[f"embedding_{postfix}"]
                        new_m = embedding_cls(
                            num_embeddings=m.num_embeddings,
                            embedding_dim=m.embedding_dim,
                            padding_idx=m.padding_idx,
                            max_norm=m.max_norm,
                            norm_type=m.norm_type,
                            scale_grad_by_freq=m.scale_grad_by_freq,
                            sparse=m.sparse
                        )
                    elif isinstance(m, torch.nn.LayerNorm):
                        layernorm_cls = spiking_module_map[f"layernorm_{postfix}"]
                        has_bias = not (m.bias is None)
                        new_m = layernorm_cls(
                            normalized_shape=m.normalized_shape,
                            eps=m.eps,
                            elementwise_affine=m.elementwise_affine,
                            bias=has_bias
                        )
                
                special_replacement = (type(m), type(new_m)) in SPECIAL_CONVERT_PATTERNS
                if special_replacement:
                    new_m = SPECIAL_CONVERT_PATTERNS[(type(m), type(new_m))](m, new_m)
                else:
                    new_m = weight_replacement(m, new_m)

                network = set_module_by_name(network, n, new_m)
                replaced_layers.append(n)

    return network, replaced_layers


def transform_roberta(
    network: RobertaForSequenceClassification, pass_args: dict
):
    transform_by = pass_args.pop("by", None)
    if transform_by == "regex":
        return transform_roberta_by_regex_name(network, pass_args)
    elif transform_by == "type":
        return transform_roberta_by_type(network, pass_args)
    else:
        raise ValueError(f"transform_by method {transform_by} not supported!")

