"""ModelConverter that swaps nn.Linear -> BitFlipLinearLora in a torchtitan model.

Follows the torchtitan ModelConverter protocol:
  - convert(model): in-place replacement of Linear layers
  - post_optimizer_hook(model): increment the step counter on all BitFlipLinearLora layers
"""

from dataclasses import dataclass

import torch.nn as nn

from .bitflip_lora_linear import BitFlipLinearLora


@dataclass
class BitFlipLoRAConfig:
    """Configuration for the bitflip + LoRA converter."""

    # Bitflip probabilities (None = disabled for that component)
    x_p_exp: float | None = None
    x_p_frac: float | None = None
    x_zero_out_t: float | None = None
    w_p_exp: float | None = None
    w_p_frac: float | None = None
    w_zero_out_t: float | None = None

    # LoRA hyperparameters
    r: int = 32
    lora_alpha: int = 32

    # Random seed base
    base_seed: int = 42

    # Module name patterns to skip (e.g. "output" for the lm_head)
    skip_patterns: tuple[str, ...] = ("output",)


class BitFlipLoRAConverter:
    """Converts nn.Linear layers to BitFlipLinearLora layers.

    Implements the torchtitan ModelConverter protocol.
    """

    def __init__(self, config: BitFlipLoRAConfig):
        self.config = config

    def convert(self, model: nn.Module) -> list[str]:
        """Replace all nn.Linear layers (except skipped ones) with BitFlipLinearLora.

        Returns list of replaced layer names.
        """
        cfg = self.config
        replaced = []
        layer_idx = 0

        # Collect (name, module, parent, attr) for all Linear layers
        replacements = []
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if any(pat in name for pat in cfg.skip_patterns):
                continue
            replacements.append((name, module, layer_idx))
            layer_idx += 1

        for name, module, idx in replacements:
            new_layer = BitFlipLinearLora.from_linear(
                module,
                x_p_exp=cfg.x_p_exp,
                x_p_frac=cfg.x_p_frac,
                x_zero_out_t=cfg.x_zero_out_t,
                w_p_exp=cfg.w_p_exp,
                w_p_frac=cfg.w_p_frac,
                w_zero_out_t=cfg.w_zero_out_t,
                base_seed=cfg.base_seed,
                layer_idx=idx,
                r=cfg.r,
                lora_alpha=cfg.lora_alpha,
            )
            _set_module_by_name(model, name, new_layer)
            replaced.append(name)

        return replaced

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
        """Increment the step counter on all BitFlipLinearLora layers.

        Called after each optimizer step so that the next forward pass
        uses different random seeds.
        """
        models = model if isinstance(model, list) else [model]
        for m in models:
            for module in m.modules():
                if isinstance(module, BitFlipLinearLora):
                    module._step += 1


def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    """Replace a submodule identified by dotted name."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def freeze_non_lora_params(model: nn.Module):
    """Freeze all parameters except LoRA A/B matrices."""
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def get_lora_params(model: nn.Module) -> list[nn.Parameter]:
    """Return only the trainable LoRA parameters."""
    return [p for n, p in model.named_parameters() if "lora_A" in n or "lora_B" in n]


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
