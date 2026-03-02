import math

import torch
from mase_triton.random_bitflip.core import random_bitflip_fn
from mase_triton.random_bitflip.layers import RandomBitFlipLinear
from torch import Tensor, nn


class BitFlipLinearLora(RandomBitFlipLinear):
    """RandomBitFlipLinear with LoRA adaptation.

    Forward: Y = bitflip(X) @ bitflip(W + B @ A * scaling)^T + bias
    Only lora_A and lora_B are trainable during fine-tuning.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        device,
        dtype,
        x_p_exp: float | None,
        x_p_frac: float | None,
        x_zero_out_t: float | None,
        w_p_exp: float | None,
        w_p_frac: float | None,
        w_zero_out_t: float | None,
        x_seed_exp: int = 0,
        x_seed_frac: int = 0,
        w_seed_exp: int = 0,
        w_seed_frac: int = 0,
        r: int = 32,
        lora_alpha: int = 32,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias,
            device,
            dtype,
            x_p_exp=x_p_exp,
            x_p_frac=x_p_frac,
            x_zero_out_t=x_zero_out_t,
            w_p_exp=w_p_exp,
            w_p_frac=w_p_frac,
            w_zero_out_t=w_zero_out_t,
            x_seed_exp=x_seed_exp,
            x_seed_frac=x_seed_frac,
            w_seed_exp=w_seed_exp,
            w_seed_frac=w_seed_frac,
        )
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r if r > 0 else 1

        if r > 0:
            self.lora_A = nn.Parameter(
                torch.zeros((r, in_features), device=device, dtype=dtype)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((out_features, r), device=device, dtype=dtype)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
        self.r: int
        self.lora_A: nn.Parameter | None
        self.lora_B: nn.Parameter | None
        self.scaling: float

    def forward(self, x: Tensor) -> Tensor:
        # 1. Apply input bitflip (if configured)
        if not (self.x_p_exp is None and self.x_p_frac is None and self.x_zero_out_t is None):
            x, x_seed_exp, x_seed_frac = random_bitflip_fn(
                x,
                exp_halves=self.x_nearest_exp_halves,
                frac_halves=self.x_nearest_frac_halves,
                seed_exp=self.x_seed_exp,
                seed_frac=self.x_seed_frac,
                zero_out_threshold=self.x_zero_out_t,
            )
            self.x_seed_exp = x_seed_exp
            self.x_seed_frac = x_seed_frac

        # 2. Compute adapted weight: W + B @ A * scaling
        w = self.weight
        if self.r > 0:
            w = w + (self.lora_B @ self.lora_A) * self.scaling

        # 3. Apply weight bitflip
        if self.w_p_exp is None and self.w_p_frac is None and self.w_zero_out_t is None:
            pass
        else:
            w, w_seed_exp, w_seed_frac = random_bitflip_fn(
                w,
                exp_halves=self.w_nearest_exp_halves,
                frac_halves=self.w_nearest_frac_halves,
                seed_exp=self.w_seed_exp,
                seed_frac=self.w_seed_frac,
                zero_out_threshold=self.w_zero_out_t,
            )
            self.w_seed_exp = w_seed_exp
            self.w_seed_frac = w_seed_frac

        # 4. Linear transformation
        return torch.nn.functional.linear(x, w, self.bias)

    @classmethod
    def from_linear(
        cls,
        linear: torch.nn.Linear,
        x_p_exp: float | None,
        x_p_frac: float | None,
        x_zero_out_t: float | None,
        w_p_exp: float | None,
        w_p_frac: float | None,
        w_zero_out_t: float | None,
        x_seed_exp: int = 0,
        x_seed_frac: int = 0,
        w_seed_exp: int = 0,
        w_seed_frac: int = 0,
        r: int = 32,
        lora_alpha: int = 32,
    ) -> "BitFlipLinearLora":
        new_fc = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
            linear.weight.dtype,
            x_p_exp=x_p_exp,
            x_p_frac=x_p_frac,
            x_zero_out_t=x_zero_out_t,
            w_p_exp=w_p_exp,
            w_p_frac=w_p_frac,
            w_zero_out_t=w_zero_out_t,
            x_seed_exp=x_seed_exp,
            x_seed_frac=x_seed_frac,
            w_seed_exp=w_seed_exp,
            w_seed_frac=w_seed_frac,
            r=r,
            lora_alpha=lora_alpha,
        )
        with torch.no_grad():
            if linear.weight.device != torch.device("meta"):
                new_fc.weight.copy_(linear.weight)
                if linear.bias is not None:
                    new_fc.bias.copy_(linear.bias)
        return new_fc

    def merge_lora(self) -> None:
        if self.r > 0:
            with torch.no_grad():
                self.weight += (self.lora_B @ self.lora_A) * self.scaling
                self.lora_A = None
                self.lora_B = None
                self.r = 0
