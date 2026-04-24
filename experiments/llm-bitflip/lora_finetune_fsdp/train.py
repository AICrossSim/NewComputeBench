"""Bitflip-aware LoRA fine-tuning with FSDP2 via torchtitan.

Standalone training script that:
  1. Builds a Llama 3.1 model using torchtitan's model definitions (meta init)
  2. Applies the BitFlipLoRA converter (nn.Linear -> BitFlipLinearLora)
  3. Freezes base model parameters, keeping only LoRA A/B trainable
  4. Applies FSDP2 for memory-efficient distributed training
  5. Loads HuggingFace checkpoint weights (unsloth/Meta-Llama-3.1-70B)
  6. Runs a standard CLM fine-tuning loop

Launch with torchrun:
    torchrun --nproc_per_node=8 train.py --config config.toml
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import tomllib
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

logger = logging.getLogger(__name__)

# Add torchtitan to path
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR / "torchtitan"))

from torchtitan.config import TORCH_DTYPE_MAP  # noqa: E402
from torchtitan.models.llama3 import Llama3Model, llama3_configs  # noqa: E402
from torchtitan.models.llama3.state_dict_adapter import (  # noqa: E402
    Llama3StateDictAdapter,
)

sys.path.insert(0, str(_SCRIPT_DIR))
from bitflip_lora.bitflip_lora_linear import BitFlipLinearLora  # noqa: E402
from bitflip_lora.converter import (  # noqa: E402
    BitFlipLoRAConfig,
    BitFlipLoRAConverter,
    count_parameters,
    freeze_non_lora_params,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Bitflip-aware LoRA fine-tuning with FSDP2")
    parser.add_argument("--config", type=str, required=True, help="Path to TOML config file")
    parser.add_argument("--steps", type=int, default=None, help="Override training.steps from config")
    return parser.parse_args()


def load_config(path: str, args) -> dict:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    # CLI overrides
    if args.steps is not None:
        cfg["training"]["steps"] = args.steps
    return cfg


def setup_distributed():
    """Initialize distributed training and return device mesh."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return device


def build_model(model_flavor: str, dtype: torch.dtype) -> tuple[Llama3Model, Llama3Model.Config]:
    """Build Llama3 model on meta device."""
    model_config = llama3_configs[model_flavor]()
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("meta"):
            model = model_config.build()
    finally:
        torch.set_default_dtype(prev_dtype)
    return model, model_config


def load_hf_weights(
    model: Llama3Model,
    model_config: Llama3Model.Config,
    hf_model_path: str,
    device: torch.device,
):
    """Load HuggingFace checkpoint into an FSDP2-wrapped model.

    Uses torch.distributed.checkpoint (dcp) with HuggingFaceStorageReader
    so that each FSDP rank only loads its own shard - essential for 70B models
    where the full state dict doesn't fit in a single GPU's host memory.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import HuggingFaceStorageReader
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        StateDictOptions,
    )

    rank = dist.get_rank()
    if rank == 0:
        logger.info(f"Loading HF weights from {hf_model_path}")

    # Resolve HF Hub ID to local cache path if needed
    local_path = hf_model_path
    if not Path(hf_model_path).exists():
        from huggingface_hub import snapshot_download

        if rank == 0:
            logger.info(f"Resolving HF model: {hf_model_path}")
        try:
            local_path = snapshot_download(hf_model_path, local_files_only=True)
        except Exception:
            if rank == 0:
                logger.info("Not in local cache, downloading...")
            local_path = snapshot_download(hf_model_path)
        if rank == 0:
            logger.info(f"Resolved to local path: {local_path}")

    adapter = Llama3StateDictAdapter(model_config, local_path)
    hf_storage_reader = adapter.get_hf_storage_reader(local_path)

    # Get the model's sharded state dict (DTensor references to FSDP params).
    # These are live references - dcp.load writes directly into them.
    state_dict = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=False, strict=False),
    )

    # Filter out LoRA/bitflip keys not present in the HF checkpoint.
    extra_key_patterns = ("lora_A", "lora_B", "_step")
    base_state_dict = {
        k: v for k, v in state_dict.items()
        if not any(pat in k for pat in extra_key_patterns)
    }

    # Convert to HF key names, load shards via dcp, convert back.
    # This follows torchtitan's exact checkpoint loading pattern:
    # to_hf -> dcp.load -> from_hf -> model.load_state_dict
    hf_state_dict = adapter.to_hf(base_state_dict)
    dcp.load(hf_state_dict, storage_reader=hf_storage_reader)
    tt_state_dict = adapter.from_hf(hf_state_dict)

    # Use model.load_state_dict directly (not set_model_state_dict which
    # can materialize full tensors and OOM on large models).
    model.load_state_dict(tt_state_dict, strict=False)

    if rank == 0:
        logger.info("HF weights loaded successfully via distributed checkpoint")


def apply_fsdp2(
    model: Llama3Model,
    dp_mesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
):
    """Apply activation checkpointing and FSDP2 to the model."""
    from torch.distributed._composable.fsdp import FSDPModule
    from torch.utils.checkpoint import checkpoint

    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    # Apply activation checkpointing to each transformer block.
    # This recomputes activations during backward instead of storing them,
    # reducing memory from ~67 GB to ~2.7 GB for 70B.
    for block in model.layers.values():
        original_forward = block.forward

        def make_ac_forward(fwd):
            def ac_forward(*args, **kwargs):
                return checkpoint(fwd, *args, use_reentrant=False, **kwargs)
            return ac_forward

        block.forward = make_ac_forward(original_forward)

    # Shard embeddings
    if model.tok_embeddings is not None:
        fully_shard(model.tok_embeddings, **fsdp_config, reshard_after_forward=True)

    # Shard each transformer layer
    for layer_id, block in model.layers.items():
        fully_shard(block, **fsdp_config, reshard_after_forward=True)

    # Shard norm + output together
    if model.norm is not None and model.output is not None:
        fully_shard([model.norm, model.output], **fsdp_config, reshard_after_forward=False)

    # Shard root
    fully_shard(model, **fsdp_config)

    # Disable FSDP's automatic gradient division (we handle it manually)
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(1.0)


def build_dataloader(cfg: dict, rank: int, world_size: int, tokenizer):
    """Build a streaming dataloader for CLM fine-tuning."""
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    dataset_name = cfg["dataset"]["name"]
    dataset_config = cfg["dataset"].get("config", None)
    split = cfg["dataset"].get("split", "train")
    seq_len = cfg["training"]["seq_len"]
    batch_size = cfg["training"]["local_batch_size"]

    logger.info(f"Loading dataset: {dataset_name} (config={dataset_config}, split={split})")
    # Local path (parquet directory) vs HF Hub name
    if Path(dataset_name).exists():
        ds = load_dataset(dataset_name, split=split, streaming=True)
    else:
        ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True)

    # Shard the streaming dataset across ranks
    ds = ds.shard(num_shards=world_size, index=rank)

    def tokenize_and_chunk(examples):
        """Tokenize and concatenate into fixed-length chunks."""
        text_key = cfg["dataset"].get("text_field", "text")
        tokenized = tokenizer(
            examples[text_key],
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        all_ids = []
        for ids in tokenized["input_ids"]:
            all_ids.extend(ids)

        # Create chunks of seq_len + 1 (input + label)
        chunks = []
        for i in range(0, len(all_ids) - seq_len, seq_len):
            chunks.append(all_ids[i : i + seq_len + 1])
        return {"input_ids": chunks}

    # For streaming datasets, column_names is available as a property
    cols = ds.column_names if hasattr(ds, "column_names") and ds.column_names else None
    ds = ds.map(tokenize_and_chunk, batched=True, remove_columns=cols)

    def collate_fn(batch):
        input_ids = torch.stack([torch.tensor(item["input_ids"], dtype=torch.long) for item in batch])
        labels = input_ids[:, 1:].clone()
        input_ids = input_ids[:, :-1]
        return input_ids, labels

    return DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=cfg["training"].get("num_workers", 2),
        pin_memory=True,
    )


def save_lora_checkpoint(model, cfg, save_dir, tag, rank):
    """Save only LoRA weights using FSDP2-aware sharded state dict.

    Uses full_state_dict=False + CPU offload to avoid GPU OOM on large models.
    Only LoRA parameters are gathered to rank 0 for saving.
    """
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        StateDictOptions,
    )

    # Get sharded state dict (low memory) and filter to LoRA keys
    sharded_sd = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=False),
    )
    lora_sharded = {
        k: v for k, v in sharded_sd.items() if "lora_A" in k or "lora_B" in k
    }

    # Gather only LoRA params to full tensors on CPU (tiny: ~0.8 GB for 70B r=32)
    lora_full = {}
    for k, v in lora_sharded.items():
        if hasattr(v, "full_tensor"):
            lora_full[k] = v.full_tensor().cpu()
        else:
            lora_full[k] = v.cpu()

    if rank == 0:
        ckpt_path = os.path.join(save_dir, tag)
        os.makedirs(ckpt_path, exist_ok=True)
        torch.save(lora_full, os.path.join(ckpt_path, "lora_weights.pt"))
        with open(os.path.join(ckpt_path, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2, default=str)
        logger.info(f"Saved LoRA checkpoint to {ckpt_path}")

    dist.barrier()


def train(cfg: dict):
    """Main training function."""
    device = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    # ----------------------------------------------------------------
    # 1. Build model on meta device
    # ----------------------------------------------------------------
    model_flavor = cfg["model"]["flavor"]
    dtype = TORCH_DTYPE_MAP[cfg["training"].get("dtype", "bfloat16")]

    if rank == 0:
        logger.info(f"Building Llama3 {model_flavor} on meta device (dtype={dtype})")

    model, model_config = build_model(model_flavor, dtype)

    # ----------------------------------------------------------------
    # 2. Apply BitFlipLoRA converter (before FSDP, while on meta device)
    # ----------------------------------------------------------------
    bf_cfg = cfg.get("bitflip", {})
    lora_cfg = cfg.get("lora", {})
    converter_config = BitFlipLoRAConfig(
        x_p_exp=bf_cfg.get("x_p_exp"),
        x_p_frac=bf_cfg.get("x_p_frac"),
        x_zero_out_t=bf_cfg.get("x_zero_out_t"),
        w_p_exp=bf_cfg.get("w_p_exp"),
        w_p_frac=bf_cfg.get("w_p_frac"),
        w_zero_out_t=bf_cfg.get("w_zero_out_t"),
        r=lora_cfg.get("r", 32),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        base_seed=bf_cfg.get("base_seed", 42),
        skip_patterns=tuple(bf_cfg.get("skip_patterns", ["output"])),
    )
    converter = BitFlipLoRAConverter(converter_config)
    replaced = converter.convert(model)

    if rank == 0:
        logger.info(f"Replaced {len(replaced)} Linear layers with BitFlipLinearLora")

    # Freeze base model parameters
    freeze_non_lora_params(model)

    total_params, trainable_params = count_parameters(model)
    if rank == 0:
        logger.info(
            f"Total params: {total_params:,} | Trainable (LoRA): {trainable_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

    # ----------------------------------------------------------------
    # 3. Materialize model on GPU and apply FSDP2
    # ----------------------------------------------------------------
    # Create device mesh for FSDP
    fsdp_mesh_dim = cfg["parallelism"].get("fsdp_degree", world_size)
    if fsdp_mesh_dim > world_size:
        if rank == 0:
            logger.warning(f"fsdp_degree={fsdp_mesh_dim} > world_size={world_size}, using world_size")
        fsdp_mesh_dim = world_size
    dp_mesh = init_device_mesh("cuda", (fsdp_mesh_dim,), mesh_dim_names=("fsdp",))

    param_dtype = TORCH_DTYPE_MAP[cfg["training"].get("mixed_precision_param", "bfloat16")]
    reduce_dtype = TORCH_DTYPE_MAP[cfg["training"].get("mixed_precision_reduce", "float32")]

    apply_fsdp2(model, dp_mesh, param_dtype, reduce_dtype)

    # Materialize from meta device to real device
    model.to_empty(device=device)

    # Initialize LoRA weights (after materializing from meta)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, BitFlipLinearLora):
                module.reset_lora_parameters()
        # Initialize model's built-in states (RoPE freqs, norms, etc.)
        model.init_states(buffer_device=None)

    model.train()

    if rank == 0:
        alloc = torch.cuda.memory_allocated() / 1e9
        logger.info(f"Model materialized on GPU with FSDP2 ({alloc:.1f} GB allocated)")

    # ----------------------------------------------------------------
    # 4. Load pretrained weights from HF checkpoint
    # ----------------------------------------------------------------
    hf_model_path = cfg["model"]["hf_model_path"]
    load_hf_weights(model, model_config, hf_model_path, device)
    if rank == 0:
        alloc = torch.cuda.memory_allocated() / 1e9
        logger.info(f"After weight loading: {alloc:.1f} GB allocated")

    # ----------------------------------------------------------------
    # 5. Build tokenizer and dataloader
    # ----------------------------------------------------------------
    from transformers import AutoTokenizer

    tokenizer_path = cfg["model"].get("tokenizer_path", hf_model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataloader = build_dataloader(cfg, rank, world_size, tokenizer)
    data_iter = iter(dataloader)

    # ----------------------------------------------------------------
    # 6. Build optimizer and LR scheduler (LoRA params only)
    # ----------------------------------------------------------------
    lr = cfg["training"].get("lr", 2e-4)
    weight_decay = cfg["training"].get("weight_decay", 0.0)
    lora_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)

    num_steps = cfg["training"]["steps"]
    warmup_steps = cfg["training"].get("warmup_steps", 100)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, num_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ----------------------------------------------------------------
    # 7. Training loop
    # ----------------------------------------------------------------
    seq_len = cfg["training"]["seq_len"]
    log_freq = cfg["training"].get("log_freq", 10)
    save_freq = cfg["training"].get("save_freq", 500)
    save_dir = cfg["training"].get("save_dir", "./checkpoints")
    max_norm = cfg["training"].get("max_norm", 1.0)
    grad_accum_steps = cfg["training"].get("gradient_accumulation_steps", 1)

    if rank == 0:
        logger.info(
            f"Training for {num_steps} steps, seq_len={seq_len}, "
            f"lr={lr}, warmup={warmup_steps}, grad_accum={grad_accum_steps}"
        )
        os.makedirs(save_dir, exist_ok=True)

    model.train()
    start_time = time.time()
    running_loss = 0.0

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        accumulated_loss = 0.0

        for _micro in range(grad_accum_steps):
            try:
                input_ids, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                input_ids, labels = next(data_iter)

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(input_ids)

            # Compute cross-entropy loss
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            loss = loss / grad_accum_steps
            loss.backward()
            accumulated_loss += loss.detach().item()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(lora_params, max_norm)

        # Optimizer step
        optimizer.step()
        scheduler.step()

        # Increment bitflip step counters
        converter.post_optimizer_hook(model)

        running_loss += accumulated_loss

        # Logging
        if rank == 0 and step % log_freq == 0:
            avg_loss = running_loss / log_freq
            elapsed = time.time() - start_time
            tokens_per_sec = (
                step * seq_len * cfg["training"]["local_batch_size"] * world_size * grad_accum_steps
            ) / elapsed
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Step {step}/{num_steps} | Loss: {avg_loss:.4f} | "
                f"LR: {current_lr:.2e} | Tokens/s: {tokens_per_sec:.0f} | "
                f"Elapsed: {elapsed:.1f}s"
            )
            running_loss = 0.0

        # Save checkpoint (all ranks participate for FSDP2 gathering)
        if step % save_freq == 0:
            save_lora_checkpoint(model, cfg, save_dir, f"step_{step}", rank)

    # Final save
    save_lora_checkpoint(model, cfg, save_dir, "final", rank)
    if rank == 0:
        logger.info("Training complete.")

    dist.destroy_process_group()


def main():
    args = parse_args()
    cfg = load_config(args.config, args)
    train(cfg)


if __name__ == "__main__":
    main()
