from collections import namedtuple
import logging
import math
from typing import Callable, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.models.llama.model import apply_rotary_emb, repeat_kv, ModelArgs, Attention
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer as HFLlamaDecoderLayer,
    LlamaAttention as HFLlamaAttention,
)
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as hf_apply_rotary_pos_emb

from mase_triton.optical_compute import OpticalTransformerFunctions as OTFunctions
from mase_triton.optical_compute import layers as OTLayers
from mase_triton.optical_compute.layers import optical_transformer_update_qstats


logger = logging.getLogger(__name__)


# Efficient implementation equivalent to the following:
def optical_transformer_SDPA(
    query,
    key,
    value,
    query_min_max: Tensor,
    key_min_max: Tensor,
    qk_min_max: Tensor,
    attn_min_max: Tensor,
    value_min_max: Tensor,
    av_min_max: Tensor,
    q_min_max_quantiles: Tensor,
    q_seed: Tensor,
    q_update_stats: bool,
    q_levels: int = 256,
    q_lut_min: float = 0.020040,
    q_smooth_factor: float = 0.9,
    q_bypass: bool = False,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(query.device)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    # attn_weight = query @ key.transpose(-2, -1) * scale_factor
    #
    if q_update_stats:
        with torch.no_grad():
            query_min_max_ = optical_transformer_update_qstats(
                query, query_min_max, q_min_max_quantiles, q_smooth_factor
            )
            query_min_max.copy_(query_min_max_)
            key_min_max_ = optical_transformer_update_qstats(key, key_min_max, q_min_max_quantiles, q_smooth_factor)
            key_min_max.copy_(key_min_max_)
            value_min_max_ = optical_transformer_update_qstats(
                value, value_min_max, q_min_max_quantiles, q_smooth_factor
            )
            value_min_max.copy_(value_min_max_)
            if not qk_min_max.isfinite().all():
                attn_weight_ = query @ key.transpose(-2, -1)
                attn_score_ = torch.softmax(attn_weight_ * scale_factor + attn_bias, dim=-1)
                y_ = attn_score_ @ value
                qk_min_max_ = optical_transformer_update_qstats(
                    attn_weight_, qk_min_max, q_min_max_quantiles, q_smooth_factor
                )
                qk_min_max.copy_(qk_min_max_)
                # attn_min_max_ = _optical_transformer_update_stats(
                #     attn_score_, attn_min_max, query_min_max, q_smooth_factor
                # )
                # attn_min_max.copy_(attn_min_max_)
                av_min_max_ = optical_transformer_update_qstats(y_, av_min_max, q_min_max_quantiles, q_smooth_factor)
                av_min_max.copy_(av_min_max_)

    attn_weight, _ = OTFunctions.quantized_matmul_fn(
        a=query.contiguous(),
        b=key.transpose(-2, -1).contiguous(),
        a_min=query_min_max[0],
        a_max=query_min_max[1],
        b_min=key_min_max[0],
        b_max=key_min_max[1],
        b_lut_min=q_lut_min,
        o_min=qk_min_max[0],
        o_max=qk_min_max[1],
        q_levels=q_levels,
        q_seed=q_seed.item(),
        skip_quantize=q_bypass,
    )
    if q_update_stats:
        with torch.no_grad():
            qk_min_max_ = optical_transformer_update_qstats(
                attn_weight, qk_min_max, q_min_max_quantiles, q_smooth_factor
            )
            qk_min_max.copy_(qk_min_max_)

    attn_weight = attn_weight * scale_factor
    #
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    # y = attn_weight @ value
    #
    y, _ = OTFunctions.quantized_matmul_fn(
        a=attn_weight,
        b=value.contiguous(),
        a_min=attn_min_max[0],
        a_max=attn_min_max[1],
        b_min=value_min_max[0],
        b_max=value_min_max[1],
        b_lut_min=q_lut_min,
        o_min=av_min_max[0],
        o_max=av_min_max[1],
        q_levels=q_levels,
        q_seed=q_seed.item() + 1,
        skip_quantize=q_bypass,
    )
    if q_update_stats:
        with torch.no_grad():
            av_min_max_ = optical_transformer_update_qstats(y, av_min_max, q_min_max_quantiles, q_smooth_factor)
            av_min_max.copy_(av_min_max_)
    #
    return y


FakeModelArgs = namedtuple("FakeModelArgs", ["n_heads", "n_kv_heads", "dim"])


class TTOpticalTransformerLlamaAttention(nn.Module):
    """
    Multi-head optical transformer attention module (torchtitan)

    Args:
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.

    """

    def __init__(
        self,
        model_args: ModelArgs,
        q_levels: int = 256,
        q_lut_min: float = 0.020040,
        q_quantiles: tuple[float, float] | None = None,
        q_smooth_factor: float = 0.9,
        q_init_seed: int = 0,
        q_bypass: bool = False,
    ):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = model_args.n_heads if model_args.n_kv_heads is None else model_args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads

        self.wq = nn.Linear(model_args.dim, model_args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(model_args.n_heads * self.head_dim, model_args.dim, bias=False)

        self.q_levels = q_levels
        self.q_lut_min = q_lut_min
        if q_quantiles is None:
            self.q_min_max_quantiles = None
        else:
            self.register_buffer("q_min_max_quantiles", torch.tensor(q_quantiles))
        self.register_buffer("query_min_max", torch.tensor([float("inf"), float("-inf")]))
        self.register_buffer("key_min_max", torch.tensor([float("inf"), float("-inf")]))
        self.register_buffer("qk_min_max", torch.tensor([float("inf"), float("-inf")]))
        self.register_buffer("attn_min_max", torch.tensor([float(0), float(1)]))
        self.register_buffer("value_min_max", torch.tensor([float("inf"), float("-inf")]))
        self.register_buffer("av_min_max", torch.tensor([float("inf"), float("-inf")]))
        self.register_buffer("seed", torch.tensor(q_init_seed, dtype=torch.int64))
        self.stat_smooth_factor = q_smooth_factor
        self.bypass = q_bypass

        self.query_min_max: Tensor
        self.key_min_max: Tensor
        self.qk_min_max: Tensor
        self.attn_min_max: Tensor
        self.value_min_max: Tensor
        self.av_min_max: Tensor
        self.seed: Tensor

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        # we use casual mask for training
        # output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        if self.bypass:
            output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        else:
            output = optical_transformer_SDPA(
                xq,
                xk,
                xv,
                query_min_max=self.query_min_max,
                key_min_max=self.key_min_max,
                qk_min_max=self.qk_min_max,
                attn_min_max=self.attn_min_max,
                value_min_max=self.value_min_max,
                av_min_max=self.av_min_max,
                q_min_max_quantiles=self.q_min_max_quantiles,
                q_seed=self.seed,
                q_update_stats=self.training,
                q_levels=self.q_levels,
                q_lut_min=self.q_lut_min,
                q_bypass=self.bypass,
                is_causal=True,
            )
            with torch.no_grad():
                self.seed = self.seed + 2
        output = output.transpose(1, 2).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)
        return self.wo(output)

    @classmethod
    def from_pretrained(
        cls,
        attn: Attention,
        q_levels: int = 256,
        q_lut_min: float = 0.020040,
        q_quantiles: tuple[float, float] = (0.001, 0.999),
        q_smooth_factor: float = 0.9,
        q_init_seed: int = 0,
        q_bypass: bool = False,
    ):
        model_args = FakeModelArgs(
            n_heads=attn.n_heads,
            n_kv_heads=attn.n_kv_heads,
            dim=attn.n_heads * attn.head_dim,
        )
        new_attn = cls(
            model_args=model_args,
            q_levels=q_levels,
            q_lut_min=q_lut_min,
            q_quantiles=q_quantiles,
            q_smooth_factor=q_smooth_factor,
            q_init_seed=q_init_seed,
            q_bypass=q_bypass,
        )
        with torch.no_grad():
            if all(
                [
                    w.device != torch.device("meta")
                    for w in [attn.wq.weight, attn.wk.weight, attn.wv.weight, attn.wo.weight]
                ]
            ):
                new_attn.wq.weight.copy_(attn.wq.weight)
                new_attn.wk.weight.copy_(attn.wk.weight)
                new_attn.wv.weight.copy_(attn.wv.weight)
                new_attn.wo.weight.copy_(attn.wo.weight)
            else:
                logger.warning(
                    "Weights are not copied from the original attention layer because some of them are on the meta device."
                )

        return new_attn


class HFOpticalTransformerLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config,
        layer_idx: int,
        q_levels: int = 256,
        q_lut_min: float = 0.020040,
        q_quantiles: tuple[float, float] | None = None,
        q_smooth_factor: float = 0.9,
        q_init_seed: int = 0,
        q_bypass: bool = False,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        self.q_levels = q_levels
        self.q_lut_min = q_lut_min
        if q_quantiles is None:
            self.q_min_max_quantiles = None
        else:
            self.register_buffer("q_min_max_quantiles", torch.tensor(q_quantiles))
        self.register_buffer("query_min_max", torch.tensor([float("inf"), float("-inf")]))
        self.register_buffer("key_min_max", torch.tensor([float("inf"), float("-inf")]))
        self.register_buffer("qk_min_max", torch.tensor([float("inf"), float("-inf")]))
        self.register_buffer("attn_min_max", torch.tensor([float(0), float(1)]))
        self.register_buffer("value_min_max", torch.tensor([float("inf"), float("-inf")]))
        self.register_buffer("av_min_max", torch.tensor([float("inf"), float("-inf")]))
        self.register_buffer("seed", torch.tensor(q_init_seed, dtype=torch.int64))
        self.stat_smooth_factor = q_smooth_factor
        self.bypass = q_bypass

        self.query_min_max: Tensor
        self.key_min_max: Tensor
        self.qk_min_max: Tensor
        self.attn_min_max: Tensor
        self.value_min_max: Tensor
        self.av_min_max: Tensor
        self.seed: Tensor

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = hf_apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_weights = None
        if self.bypass:
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                is_causal=True,
                dropout_p=0.0 if not self.training else self.attention_dropout,
                scale=self.scaling,
                attn_mask=attention_mask,
            )
        else:
            attn_output = optical_transformer_SDPA(
                query_states,
                key_states,
                value_states,
                query_min_max=self.query_min_max,
                key_min_max=self.key_min_max,
                qk_min_max=self.qk_min_max,
                attn_min_max=self.attn_min_max,
                value_min_max=self.value_min_max,
                av_min_max=self.av_min_max,
                q_min_max_quantiles=self.q_min_max_quantiles,
                q_seed=self.seed,
                q_update_stats=self.training,
                q_levels=self.q_levels,
                q_lut_min=self.q_lut_min,
                q_bypass=self.bypass,
                is_causal=True,
                dropout_p=0.0 if not self.training else self.attention_dropout,
                scale=self.scaling,
                attn_mask=attention_mask,
                enable_gqa=self.num_key_value_groups > 1,
            )
            with torch.no_grad():
                self.seed = self.seed + 2

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    @classmethod
    def from_pretrained(
        cls,
        attn: HFLlamaAttention,
        q_levels: int = 256,
        q_lut_min: float = 0.020040,
        q_quantiles: tuple[float, float] | None = None,
        q_smooth_factor: float = 0.9,
        q_init_seed: int = 0,
        q_bypass: bool = False,
    ):
        new_attn = cls(
            attn.config,
            attn.layer_idx,
            q_levels=q_levels,
            q_lut_min=q_lut_min,
            q_quantiles=q_quantiles,
            q_smooth_factor=q_smooth_factor,
            q_init_seed=q_init_seed,
            q_bypass=q_bypass,
        )

        with torch.no_grad():
            if attn.q_proj.weight.device != torch.device("meta"):
                new_attn.load_state_dict(attn.state_dict(), strict=False)

        return new_attn
