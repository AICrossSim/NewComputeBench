"""BitFlipLinearLora module compatible with FSDP2 and torch.compile.

Key differences from the mase-triton BitFlipLinearLora:
  1. No mutable seed state - seeds are derived from a global step counter
     plus a per-layer offset, making the module torch.compile-safe.
  2. Works on meta-device init (torchtitan builds models on meta then
     materializes later).
  3. LoRA A/B are registered as Parameters so FSDP2 shards them properly.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from .triton_kernels import (
    find_nearest_prob_n_halves,
    random_bitflip_fn,
)

# Large prime for seed mixing to avoid correlated randomness across layers
_SEED_PRIME = 1000003


class BitFlipLinearLora(nn.Module):
    """Linear layer with random bitflip injection and LoRA adaptation.

    Forward:  Y = bitflip(X) @ bitflip(W + B @ A * scaling)^T + bias

    Only ``lora_A`` and ``lora_B`` are trainable; ``weight`` and ``bias``
    are frozen during fine-tuning.

    Seed management:
        Instead of mutating ``self.seed_*`` on each forward call (which
        breaks torch.compile), we derive seeds from an external step counter:
            seed = base_seed + step * _SEED_PRIME + layer_offset
        The step counter is stored as a 0-d tensor buffer so it can be
        updated in-place without graph breaks.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
        # bitflip config
        x_exp_halves: int | None = None,
        x_frac_halves: int | None = None,
        x_zero_out_t: float | None = None,
        w_exp_halves: int | None = None,
        w_frac_halves: int | None = None,
        w_zero_out_t: float | None = None,
        # seed config
        base_seed: int = 0,
        layer_idx: int = 0,
        # lora config
        r: int = 32,
        lora_alpha: int = 32,
    ):
        super().__init__()

        # Base linear parameters (frozen during fine-tuning)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)
        self.in_features = in_features
        self.out_features = out_features

        # Bitflip configuration (stored as plain ints, no Parameters)
        self.x_exp_halves = x_exp_halves if x_exp_halves is not None else 0
        self.x_frac_halves = x_frac_halves if x_frac_halves is not None else 0
        self.x_zero_out_t = x_zero_out_t if x_zero_out_t is not None else 0.0
        self.w_exp_halves = w_exp_halves if w_exp_halves is not None else 0
        self.w_frac_halves = w_frac_halves if w_frac_halves is not None else 0
        self.w_zero_out_t = w_zero_out_t if w_zero_out_t is not None else 0.0
        self.skip_x_exp = x_exp_halves is None
        self.skip_x_frac = x_frac_halves is None
        self.enable_x_zero_out = x_zero_out_t is not None
        self.skip_w_exp = w_exp_halves is None
        self.skip_w_frac = w_frac_halves is None
        self.enable_w_zero_out = w_zero_out_t is not None
        self.skip_x_bitflip = self.skip_x_exp and self.skip_x_frac and not self.enable_x_zero_out
        self.skip_w_bitflip = self.skip_w_exp and self.skip_w_frac and not self.enable_w_zero_out

        # Seed derivation: seed = base_seed + step * _SEED_PRIME + layer_offset
        self.base_seed = base_seed
        self.layer_idx = layer_idx
        # Step counter as a buffer (updated externally, not a Parameter)
        self.register_buffer(
            "_step", torch.tensor(0, dtype=torch.int64), persistent=False
        )

        # LoRA parameters
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0

        if r > 0:
            self.lora_A = nn.Parameter(
                torch.empty(r, in_features, device=device, dtype=dtype)
            )
            self.lora_B = nn.Parameter(
                torch.empty(out_features, r, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def reset_lora_parameters(self):
        """Initialize LoRA weights (call after materializing from meta device)."""
        if self.r > 0 and self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def _get_seeds(self) -> tuple[int, int, int, int]:
        """Derive 4 seeds (x_exp, x_frac, w_exp, w_frac) from step + layer."""
        step = self._step.item()
        base = self.base_seed + step * _SEED_PRIME + self.layer_idx * 4
        return base, base + 1, base + 2, base + 3

    def forward(self, x: Tensor) -> Tensor:
        x_seed_exp, x_seed_frac, w_seed_exp, w_seed_frac = self._get_seeds()

        # 1. Input bitflip
        if not self.skip_x_bitflip:
            x = random_bitflip_fn(
                x,
                exp_halves=self.x_exp_halves,
                frac_halves=self.x_frac_halves,
                seed_exp=x_seed_exp,
                seed_frac=x_seed_frac,
                zero_out_threshold=self.x_zero_out_t,
                skip_exp=self.skip_x_exp,
                skip_frac=self.skip_x_frac,
                enable_zero_out=self.enable_x_zero_out,
            )

        # 2. Compute adapted weight: W + B @ A * scaling
        w = self.weight
        if self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            w = w + (self.lora_B @ self.lora_A) * self.scaling

        # 3. Weight bitflip
        if not self.skip_w_bitflip:
            w = random_bitflip_fn(
                w,
                exp_halves=self.w_exp_halves,
                frac_halves=self.w_frac_halves,
                seed_exp=w_seed_exp,
                seed_frac=w_seed_frac,
                zero_out_threshold=self.w_zero_out_t,
                skip_exp=self.skip_w_exp,
                skip_frac=self.skip_w_frac,
                enable_zero_out=self.enable_w_zero_out,
            )

        # 4. Linear transform
        return torch.nn.functional.linear(x, w, self.bias)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        x_p_exp: float | None = None,
        x_p_frac: float | None = None,
        x_zero_out_t: float | None = None,
        w_p_exp: float | None = None,
        w_p_frac: float | None = None,
        w_zero_out_t: float | None = None,
        base_seed: int = 0,
        layer_idx: int = 0,
        r: int = 32,
        lora_alpha: int = 32,
    ) -> "BitFlipLinearLora":
        """Create from an existing nn.Linear, copying weights."""
        new_layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            x_exp_halves=find_nearest_prob_n_halves(x_p_exp),
            x_frac_halves=find_nearest_prob_n_halves(x_p_frac),
            x_zero_out_t=x_zero_out_t,
            w_exp_halves=find_nearest_prob_n_halves(w_p_exp),
            w_frac_halves=find_nearest_prob_n_halves(w_p_frac),
            w_zero_out_t=w_zero_out_t,
            base_seed=base_seed,
            layer_idx=layer_idx,
            r=r,
            lora_alpha=lora_alpha,
        )
        # Copy weights (skip if on meta device - torchtitan inits later)
        if linear.weight.device != torch.device("meta"):
            with torch.no_grad():
                new_layer.weight.copy_(linear.weight)
                if linear.bias is not None and new_layer.bias is not None:
                    new_layer.bias.copy_(linear.bias)
        return new_layer

    def merge_lora(self) -> None:
        """Merge LoRA weights into the base weight (for inference)."""
        if self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            with torch.no_grad():
                self.weight += (self.lora_B @ self.lora_A) * self.scaling
                self.lora_A = None
                self.lora_B = None
                self.r = 0

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"bias={self.bias is not None}, r={self.r}, "
            f"x_exp_halves={self.x_exp_halves}, x_frac_halves={self.x_frac_halves}, "
            f"w_exp_halves={self.w_exp_halves}, w_frac_halves={self.w_frac_halves}"
        )
