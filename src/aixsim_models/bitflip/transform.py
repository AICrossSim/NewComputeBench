from typing import Literal, Optional
import logging

import torch
from chop.passes.module.transforms.bitflip import bitflip_module_transform_pass
from ..utils.torch_module import TransformConfigManager
from ..utils.deps import all_packages_are_available

logger = logging.getLogger(__name__)


if not all_packages_are_available(("mase_triton", "chop")):

    def transform_model(*args, **kwargs):
        raise ImportError("mase-triton or chop not installed. Please install mase-triton to use this feature.")

    def make_transform_histogram(*args, **kwargs):
        raise ImportError("mase-triton or chop not installed. Please install mase-triton to use this feature.")

else:

    def transform_model(
        model: torch.nn.Module, config_manager: TransformConfigManager, transform_flavor: Optional[Literal["fc"]] = None
    ) -> torch.nn.Module:
        """
        Transform a model into the random bitflip form using the given configuration manager and transform flavor.

        Args:
            model (torch.nn.Module): The model to transform.
            config_manager (TransformConfigManager): The configuration manager for the transformation.
            transform_flavor (Optional[Literal["fc"]]): The flavor of the transformation. Defaults to None.

        Returns:
            torch.nn.Module: The transformed model.
        """

        if transform_flavor is None or transform_flavor == "fc":
            # *: use the bitflip transform pass in mase-tools
            pass_args = config_manager.layer_name_to_config
            pass_args = pass_args | {"by": "regex_name" if config_manager.use_regex else "name"}
            bitflip_module_transform_pass(model, pass_args=pass_args)
            return model
        else:
            raise ValueError(f"Unknown transform flavor {transform_flavor}")

    def make_transform_histogram(replaced_layers: list[tuple[str, str]]) -> dict[str, dict[str, int | list[str]]]:
        raise NotImplementedError("make_transform_histogram is not implemented.")
        # patterns = set(layer[1] for layer in replaced_layers)
        # histogram = {pattern: {"count": 0, "layers": []} for pattern in patterns}
        # for layer, pattern in replaced_layers:
        #     histogram[pattern]["count"] += 1
        #     histogram[pattern]["layers"].append(layer)
        # histogram["total"] = {"layer count": len(replaced_layers), "pattern count": len(patterns)}
        # return histogram
