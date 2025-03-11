from typing import Literal, Optional
from pathlib import Path
import logging
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

logger = logging.getLogger(__name__)


def convert_torch_to_hf(
    model_arch: Literal["aixsim"],
    model_flavor: Literal["60M", "200M", "400M", "600M", "1.1B"],
    torch_ckpt: Path,
    save_dir: Path,
    tokenizer_name: str = "HuggingFaceTB/cosmo2-tokenizer",
    ori_max_position_embeddings: int = 8192,
    max_position_embeddings: int = 131072,
    push_to_hub: Optional[str] = None,
):
    """Convert a PyTorch checkpoint to Hugging Face format for LLaMA-based models.

    This function converts a model checkpoint from a custom PyTorch format to the Hugging Face
    transformers format, specifically for LLaMA-based architectures. It handles the conversion
    of model weights, configuration, and generation settings.

    Args:
        model_arch (Literal["aixsim"]): The model architecture type.
        model_flavor (Literal["60M", "200M", "400M", "600M", "1.1B"]): Model size variant.
        torch_ckpt (Path): Path to the source PyTorch checkpoint file.
        save_dir (Path): Directory where the converted model will be saved.
        tokenizer_name (str): Name or path of the tokenizer to use.
        ori_max_position_embeddings (int, optional): Original maximum position embeddings.
            Defaults to 8192.
        max_position_embeddings (int, optional): New maximum position embeddings.
            Defaults to 131072.

    Note:
        The function supports both single-file PyTorch checkpoints and distributed checkpoint
        (DCP) formats. It performs necessary weight permutations for rotary embeddings and
        sets up appropriate model configurations.

    Raises:
        AssertionError: If the source checkpoint file does not exist.
    """
    import torch
    import transformers
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM
    from transformers.generation.configuration_utils import GenerationConfig
    from torchtitan.models import models_config
    from torchtitan.models.llama import ModelArgs
    import gc

    assert torch_ckpt.exists(), f"{torch_ckpt} does not exist"
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    if torch_ckpt.is_file():
        loaded = torch.load(torch_ckpt, weights_only=False, map_location="cpu")["model"]
        logger.info(f"Loaded torch checkpoint from {torch_ckpt}")
    else:
        # dcp format
        # from torch.distributed.checkpoint.format_utils import dcp_to_torch_save, torch_save_to_dcp
        raw_pt_bin = save_dir.joinpath("raw_pytorch_model.bin")
        dcp_to_torch_save(torch_ckpt, raw_pt_bin)
        loaded = torch.load(raw_pt_bin, weights_only=False, map_location="cpu")["model"]
        raw_pt_bin.unlink()
        logger.info(f"Loaded dcp checkpoint from {torch_ckpt}")

    new_state_dict = {}

    model_cfg: ModelArgs = models_config[model_arch][model_flavor]

    n_layers = model_cfg.n_layers
    n_heads = model_cfg.n_heads
    dim = model_cfg.dim
    dims_per_head = dim // n_heads
    base = model_cfg.rope_theta
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))

    num_key_value_heads = model_cfg.n_kv_heads
    key_value_dim = dims_per_head * num_key_value_heads

    # permute for sliced rotary
    def permute(w, n_heads, dim1=dim, dim2=dim):
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    for layer_i in range(n_layers):
        new_state_dict |= {
            f"model.layers.{layer_i}.self_attn.q_proj.weight": permute(
                loaded[f"layers.{layer_i}.attention.wq.weight"], n_heads=n_heads
            ),
            f"model.layers.{layer_i}.self_attn.k_proj.weight": permute(
                loaded[f"layers.{layer_i}.attention.wk.weight"],
                n_heads=num_key_value_heads,
                dim1=key_value_dim,
            ),
            f"model.layers.{layer_i}.self_attn.v_proj.weight": loaded[f"layers.{layer_i}.attention.wv.weight"],
            f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded[f"layers.{layer_i}.attention.wo.weight"],
            f"model.layers.{layer_i}.mlp.gate_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w1.weight"],
            f"model.layers.{layer_i}.mlp.down_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w2.weight"],
            f"model.layers.{layer_i}.mlp.up_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w3.weight"],
            f"model.layers.{layer_i}.input_layernorm.weight": loaded[f"layers.{layer_i}.attention_norm.weight"],
            f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[f"layers.{layer_i}.ffn_norm.weight"],
        }

    new_state_dict[f"rotary_emb.inv_freq"] = inv_freq
    new_state_dict |= {
        "model.embed_tokens.weight": loaded["tok_embeddings.weight"],
        "model.norm.weight": loaded["norm.weight"],
        "lm_head.weight": loaded["output.weight"],
    }

    tmp_bin = save_dir / "pytorch_model.bin"
    torch.save(new_state_dict, tmp_bin)

    ffn_dim_multiplier = model_cfg.ffn_dim_multiplier
    multiple_of = model_cfg.multiple_of

    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id

    rope_scaling = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": ori_max_position_embeddings,
        "rope_type": "llama3",
    }
    # rope_scaling = None

    def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
        return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)

    config = LlamaConfig(
        hidden_size=dim,
        intermediate_size=compute_intermediate_size(dim, ffn_dim_multiplier, multiple_of),
        num_attention_heads=n_heads,
        num_hidden_layers=n_layers,
        rms_norm_eps=model_cfg.norm_eps,
        num_key_value_heads=num_key_value_heads,
        vocab_size=tokenizer.vocab_size,
        rope_theta=base,
        rope_scaling=rope_scaling,
        max_position_embeddings=max_position_embeddings,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        tie_word_embeddings=False,
    )
    config.save_pretrained(save_dir)

    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )
    generation_config.save_pretrained(save_dir)

    del new_state_dict
    del loaded
    gc.collect()

    model = LlamaForCausalLM.from_pretrained(save_dir)
    del model.config._name_or_path
    model.config.torch_dtype = torch.float32

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    tmp_bin.unlink()

    logger.info(f"Converted model saved to {save_dir}")

    if push_to_hub:
        model.push_to_hub(push_to_hub)
        tokenizer.push_to_hub(push_to_hub)
        logger.info(f"Model pushed to Hugging Face Hub: {push_to_hub}")
