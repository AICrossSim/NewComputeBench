import math

import torch
from mase_triton.optical_compute.core.optical_transformer import fake
from mase_triton.optical_compute.layers import (
    OpticalTransformerLinear,
    optical_transformer_update_qstats,
)
from torch import Tensor, nn


class OpticalTransformerLinearLora(OpticalTransformerLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        q_levels: int = 256,
        q_lut_min: float | None = 0.020040,
        q_quantiles: tuple[float, float] | None = None,
        q_smooth_factor: float = 0.9,
        q_init_seed: int = 0,
        q_bypass: bool = False,
        r: int = 32,
        lora_alpha: int = 32,
    ):
        super().__init__(
            in_features,
            out_features,
            bias,
            device,
            dtype,
            q_levels,
            q_lut_min,
            q_quantiles,
            q_smooth_factor,
            q_init_seed,
            q_bypass,
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
        if self.bypass:
            return super().forward(x)

        w = self.weight
        # Apply LoRA adaptation outside of no_grad() to maintain gradients
        if self.r > 0:
            w = w + (self.lora_B @ self.lora_A) * self.scaling

        if self.training:
            with torch.no_grad():
                x_min_max = optical_transformer_update_qstats(
                    x, self.x_min_max, self.q_min_max_quantile, self.stat_smooth_factor
                )
                self.x_min_max.copy_(x_min_max)
                w_min_max = optical_transformer_update_qstats(
                    w, self.w_min_max, self.q_min_max_quantile, self.stat_smooth_factor
                )
                self.w_min_max.copy_(w_min_max)
                if self.out_min_max.isinf().any():
                    o_min_max = optical_transformer_update_qstats(
                        x @ w.t(),
                        self.out_min_max,
                        self.q_min_max_quantile,
                        self.stat_smooth_factor,
                    )
                    self.out_min_max.copy_(o_min_max)

        out_q, q_seed = fake.qlinear_fn(
            x,
            w,
            self.bias,
            x_min=self.x_min_max[0].item(),
            x_max=self.x_min_max[1].item(),
            w_min=self.w_min_max[0].item(),
            w_max=self.w_min_max[1].item(),
            w_lut_min=self.q_lut_min,
            o_min=self.out_min_max[0].item(),
            o_max=self.out_min_max[1].item(),
            q_levels=self.quant_levels,
            q_seed=self.seed.item(),
            skip_quantize=False,
        )
        with torch.no_grad():
            self.seed.copy_(q_seed)
            if self.training:
                out_min_max = optical_transformer_update_qstats(
                    out_q,
                    self.out_min_max,
                    self.q_min_max_quantile,
                    self.stat_smooth_factor,
                )
                self.out_min_max.copy_(out_min_max)

        return out_q

    @classmethod
    def from_linear(
        cls,
        linear: torch.nn.Linear,
        q_levels: int = 256,
        q_lut_min: float | None = 0.020040,
        q_quantiles: tuple[float, float] | None = None,
        q_smooth_factor: float = 0.9,
        q_init_seed: int = 0,
        q_bypass: bool = False,
        r: int = 32,
        lora_alpha: int = 32,
    ) -> "OpticalTransformerLinearLora":
        new_fc = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
            linear.weight.dtype,
            q_levels=q_levels,
            q_lut_min=q_lut_min,
            q_quantiles=q_quantiles,
            q_smooth_factor=q_smooth_factor,
            q_init_seed=q_init_seed,
            q_bypass=q_bypass,
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
