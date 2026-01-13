import math
from collections import namedtuple
from typing import Optional, Tuple

import torch
from mase_triton.optical_compute import OpticalTransformerFunctions as OTFunctions
from mase_triton.optical_compute.layers import OpticalTransformerLinear as OTLinear
from mase_triton.optical_compute.layers import optical_transformer_update_qstats
from mase_triton.utils.torch_module import get_layer_name, set_layer_by_name
from torch import Tensor, nn
from transformers.models.roberta.modeling_roberta import (
    RobertaAttention,
    RobertaForSequenceClassification,
    RobertaLayer,
    RobertaSelfAttention,
)

FakeModelArgs = namedtuple(
    "FakeModelArgs",
    [
        "hidden_size",
        "num_attention_heads",
        "position_embedding_type",
        "max_position_embeddings",
        "is_decoder",
        "attention_probs_dropout_prob",
    ],
)


class OTRobertaSelfAttention(torch.nn.Module):
    def __init__(
        self,
        config,
        position_embedding_type=None,
        q_levels: int = 256,
        q_lut_min: float = 0.020040,
        q_quantiles: tuple[float, float] | None = None,
        q_smooth_factor: float = 0.9,
        q_init_seed: int = 0,
        q_bypass: bool = False,
    ):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )

        self.is_decoder = config.is_decoder

        self.q_levels = q_levels
        self.q_lut_min = q_lut_min
        if q_quantiles is None:
            self.q_min_max_quantiles = None
        else:
            self.register_buffer("q_min_max_quantiles", torch.tensor(q_quantiles))
        self.register_buffer(
            "query_min_max", torch.tensor([float("inf"), float("-inf")])
        )
        self.register_buffer("key_min_max", torch.tensor([float("inf"), float("-inf")]))
        self.register_buffer("qk_min_max", torch.tensor([float("inf"), float("-inf")]))
        self.register_buffer("attn_min_max", torch.tensor([float(0), float(1)]))
        self.register_buffer(
            "value_min_max", torch.tensor([float("inf"), float("-inf")])
        )
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

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # *: replace matmul with optical compute
        if not self.bypass:
            with torch.no_grad():
                query_min_max_ = optical_transformer_update_qstats(
                    query_layer,
                    self.query_min_max,
                    self.q_min_max_quantiles,
                    self.stat_smooth_factor,
                )
                self.query_min_max.copy_(query_min_max_)
                key_min_max_ = optical_transformer_update_qstats(
                    key_layer,
                    self.key_min_max,
                    self.q_min_max_quantiles,
                    self.stat_smooth_factor,
                )
                self.key_min_max.copy_(key_min_max_)

                if not self.qk_min_max.isfinite().all():
                    attn_score_ = torch.matmul(query_layer, key_layer.transpose(-1, -2))
                    qk_min_max_ = optical_transformer_update_qstats(
                        attn_score_,
                        self.qk_min_max,
                        self.q_min_max_quantiles,
                        self.stat_smooth_factor,
                    )
                    self.qk_min_max.copy_(qk_min_max_)

            attention_scores, _ = OTFunctions.quantized_matmul_fn(
                a=query_layer.contiguous(),
                b=key_layer.transpose(-1, -2).contiguous(),
                a_min=self.query_min_max[0],
                a_max=self.query_min_max[1],
                b_min=self.key_min_max[0],
                b_max=self.key_min_max[1],
                b_lut_min=self.q_lut_min,
                o_min=self.qk_min_max[0],
                o_max=self.qk_min_max[1],
                q_levels=self.q_levels,
                q_seed=self.seed.item(),
                skip_quantize=self.bypass,
            )
            with torch.no_grad():
                self.seed = self.seed + 1
        else:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(
                    key_length - 1, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            else:
                position_ids_l = torch.arange(
                    query_length, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            position_ids_r = torch.arange(
                key_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # *: replace matmul with optical compute
        if not self.bypass:
            with torch.no_grad():
                attn_min_max_ = optical_transformer_update_qstats(
                    attention_probs,
                    self.attn_min_max,
                    self.q_min_max_quantiles,
                    self.stat_smooth_factor,
                )
                self.attn_min_max.copy_(attn_min_max_)
                value_min_max_ = optical_transformer_update_qstats(
                    value_layer,
                    self.value_min_max,
                    self.q_min_max_quantiles,
                    self.stat_smooth_factor,
                )
                self.value_min_max.copy_(value_min_max_)
                attn_ = torch.matmul(attention_probs, value_layer)
                av_min_max_ = optical_transformer_update_qstats(
                    attn_,
                    self.av_min_max,
                    self.q_min_max_quantiles,
                    self.stat_smooth_factor,
                )
                self.av_min_max.copy_(av_min_max_)
            context_layer, _ = OTFunctions.quantized_matmul_fn(
                a=attention_probs.contiguous(),
                b=value_layer.contiguous(),
                a_min=self.attn_min_max[0],
                a_max=self.attn_min_max[1],
                b_min=self.value_min_max[0],
                b_max=self.value_min_max[1],
                b_lut_min=self.q_lut_min,
                o_min=self.av_min_max[0],
                o_max=self.av_min_max[1],
                q_levels=self.q_levels,
                q_seed=self.seed.item(),
                skip_quantize=self.bypass,
            )
            with torch.no_grad():
                self.seed = self.seed + 1
        else:
            context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

    @classmethod
    def from_pretrained(
        cls,
        attn: RobertaSelfAttention,
        q_levels: int = 256,
        q_lut_min: float = 0.020040,
        q_quantiles: tuple[float, float] | None = None,
        q_smooth_factor: float = 0.9,
        q_init_seed: int = 0,
        q_bypass: bool = False,
    ) -> "OTRobertaSelfAttention":
        model_args = FakeModelArgs(
            hidden_size=attn.query.in_features,
            num_attention_heads=attn.num_attention_heads,
            position_embedding_type=attn.position_embedding_type,
            max_position_embeddings=getattr(attn, "max_position_embeddings", None),
            is_decoder=attn.is_decoder,
            attention_probs_dropout_prob=attn.dropout.p,
        )

        new_attn = cls(
            model_args,
            q_levels=q_levels,
            q_lut_min=q_lut_min,
            q_quantiles=q_quantiles,
            q_smooth_factor=q_smooth_factor,
            q_init_seed=q_init_seed,
            q_bypass=q_bypass,
        )

        new_attn.to(attn.query.weight.dtype)
        state_dict = attn.state_dict()
        new_attn.load_state_dict(state_dict, strict=False)
        return new_attn


def transform_roberta(
    model: RobertaForSequenceClassification, attn_config: dict, fc_config: dict
):
    assert isinstance(model, RobertaForSequenceClassification)
    replaced_layers = []

    for roberta_layer in model.roberta.encoder.layer:
        roberta_layer: RobertaLayer

        self_attn = roberta_layer.attention.self
        assert isinstance(self_attn, RobertaSelfAttention), (
            f"{get_layer_name(self_attn)} is not RobertaSelfAttention, pass '_attn_implementation='eager'"
        )

        new_self_attn = OTRobertaSelfAttention.from_pretrained(
            self_attn,
            **attn_config,
        )
        roberta_layer.attention.self = new_self_attn
        replaced_layers.append(get_layer_name(model, new_self_attn))

        if hasattr(roberta_layer, "crossattention"):
            cross_attn: RobertaAttention = roberta_layer.crossattention
            cross_attn_self = cross_attn.self
            assert isinstance(cross_attn_self, RobertaSelfAttention), (
                f"{get_layer_name(cross_attn_self)} is not RobertaSelfAttention, pass '_attn_implementation='eager'"
            )
            new_cross_attn = OTRobertaSelfAttention.from_pretrained(
                cross_attn_self,
                **attn_config,
            )
            cross_attn.self = new_cross_attn
            replaced_layers.append(get_layer_name(model, new_cross_attn))

    # replace fc
    for name, layer in model.named_modules():
        if not isinstance(layer, nn.Linear):
            continue

        # skip "classifier"
        if name.endswith("classifier"):
            continue

        new_fc = OTLinear.from_linear(
            layer,
            **fc_config,
        )

        set_layer_by_name(model, name, new_fc)
        replaced_layers.append(name)

    return replaced_layers
