import re
import torch


def get_layer_name(module: torch.nn.Module, layer: str) -> str:
    # get the name of the op relative to the module
    for name, m in module.named_modules():
        if m is layer:
            return name
    raise ValueError(f"Cannot find op {layer} in module {module}")


def get_layer_by_name(module: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    # get the op by its name relative to the module
    for name, m in module.named_modules():
        if name == layer_name:
            return m
    raise ValueError(f"Cannot find op {layer_name} in module {module}")


def set_layer_by_name(module: torch.nn.Module, name: str, new_layer: torch.nn.Module) -> None:
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = module
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit() and isinstance(mod_, (torch.nn.ModuleList, torch.nn.Sequential)):
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_layer)
    else:
        setattr(module, name, new_layer)


class TransformConfigManager:
    """
    Module transform configuration manager.
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

    def __init__(self, layer_name_to_config: dict[str, dict], use_regex: bool = True):
        self.layer_name_to_config = layer_name_to_config
        self.use_regex = use_regex

    @staticmethod
    def find_matched_pattern(layer_name: str, patterns: list[str], use_regex: bool) -> str | None:
        if use_regex:
            for pattern in patterns:
                if re.fullmatch(pattern, layer_name):
                    return pattern
            if "default" in patterns:
                return "default"
            return None
        else:
            for pattern in patterns:
                if pattern == layer_name:
                    return pattern
            if "default" in patterns:
                return "default"
            return None

    def get_layer_config_entry(self, layer_name: str) -> str:
        matched_pattern = self.find_matched_pattern(
            layer_name, self.layer_name_to_config.keys(), use_regex=self.use_regex
        )
        if matched_pattern is None:
            raise ValueError(f"No matched pattern for layer {layer_name} and no default")
        return matched_pattern

    def get_layer_config(self, layer_name: str) -> dict | None:
        """Return the config for the layer with the given name. The config is a dict, or None if skip."""
        matched_entry = self.get_layer_config_entry(layer_name)
        return self.layer_name_to_config[matched_entry]
