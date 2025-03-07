from typing import Literal, Union
from dataclasses import dataclass

import torch

from aixsim_models.utils.torch_module import set_layer_by_name, TransformConfigManager
from aixsim_models.utils.deps import all_packages_are_available


if not all_packages_are_available(("mase_triton",)):

    def RandomBitFlipDropout(*args, **kwargs):
        raise ImportError("mase-triton not installed. Please install mase-triton to use this feature.")

    def RandomBitFlipLinear(*args, **kwargs):
        raise ImportError("mase-triton not installed. Please install mase-triton to use this feature.")

else:
    from mase_triton.random_bitflip.layers import RandomBitFlipDropout, RandomBitFlipLinear


def flip_bits_in_linear(model: torch.nn.Module, config_manager: TransformConfigManager) -> list[tuple[str, str]]:
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


def transform_model(
    model: torch.nn.Module, config_manager: TransformConfigManager, transform_flavor: Literal["fc"]
) -> list[tuple[str, str]]:
    """
    Transform a model into the random bitflip form using the given configuration manager and transform flavor.

    Args:
        model (torch.nn.Module): The model to transform.
        config_manager (TransformConfigManager): The configuration manager for the transformation.
        transform_flavor (Literal["fc"]): The flavor of the transformation to apply.

    Returns:
        list[tuple[str, str]]: A list of tuples containing the names of the layers that were replaced and the configuration
            entry that was used for the replacement
    """
    if transform_flavor == "fc":
        return flip_bits_in_linear(model, config_manager)
    else:
        raise ValueError(f"Unknown transform flavor {transform_flavor}")


def make_transform_histogram(replaced_layers: list[tuple[str, str]]) -> dict[str, dict[str, int | list[str]]]:
    patterns = set(layer[1] for layer in replaced_layers)
    histogram = {pattern: {"count": 0, "layers": []} for pattern in patterns}
    for layer, pattern in replaced_layers:
        histogram[pattern]["count"] += 1
        histogram[pattern]["layers"].append(layer)
    histogram["total"] = {"layer count": len(replaced_layers), "pattern count": len(patterns)}
    return histogram
