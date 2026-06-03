"""Evaluate the ORIGINAL pretrained model (no LoRA, no training).

Builds a plain torchtitan Llama3 model, loads the HuggingFace checkpoint weights,
and measures cross-entropy loss / perplexity on the first N sequence chunks of
the training dataset. A baseline reference for the bitflip-LoRA runs.

Modes:
  * default:    clean base model, no Linear layers replaced.
  * --bitflip:  apply the [bitflip] config to the weights via BitFlipLoRAConverter
                with r=0, so bitflip noise IS injected but NO LoRA adapters are
                added and nothing is trained.
  * --profile:  clean model + histogram the magnitudes of every weight and every
                Linear-input activation, to choose safe zero-out thresholds.

Launch with torchrun (FSDP2 is still needed to fit 70B across GPUs):
    torchrun --nproc_per_node=4 eval.py --config config_70b.toml --num-samples 256
    torchrun --nproc_per_node=4 eval.py --config config_70b.toml --num-samples 256 --bitflip
    torchrun --nproc_per_node=4 eval.py --config config_70b.toml --num-samples 64 --profile
"""

import argparse
import logging
import math
import sys
from pathlib import Path

import tomllib
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

# Reuse train.py's building blocks (importing it is side-effect free; train()
# only runs under its own __main__ guard). Also mirror train.py's sys.path
# setup so eval.py can be imported / run directly without `pip install -e .`.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_SRC = _SCRIPT_DIR.parents[2] / "src"
sys.path.insert(0, str(_SCRIPT_DIR / "torchtitan"))
sys.path.insert(0, str(_REPO_SRC))
sys.path.insert(0, str(_SCRIPT_DIR))
from train import (  # noqa: E402
    apply_fsdp2,
    build_model,
    load_hf_weights,
    setup_distributed,
)
from torchtitan.config import TORCH_DTYPE_MAP  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline eval of the original model")
    parser.add_argument("--config", type=str, required=True, help="Path to TOML config file")
    parser.add_argument(
        "--num-samples", type=int, default=256,
        help="Number of seq_len-token sequences to evaluate (default: 256)",
    )
    parser.add_argument(
        "--bitflip", action="store_true",
        help="Apply weight/activation bitflip from the [bitflip] config section "
             "(r=0, so NO LoRA adapters are added)",
    )
    parser.add_argument(
        "--bitflip-seed", type=int, default=None,
        help="Override bitflip base_seed (default: [bitflip].base_seed from config)",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Profile weight/activation magnitudes of the clean model "
             "(overrides --bitflip) to choose safe zero-out thresholds",
    )
    return parser.parse_args()


def collect_eval_chunks(cfg, tokenizer, num_chunks):
    """Stream the dataset and return the first `num_chunks` fixed-length chunks.

    Mirrors train.py's tokenize-and-chunk logic: tokens are concatenated and
    sliced into windows of `seq_len + 1` (input + shifted label) with stride
    `seq_len`. Deterministic, so every rank gets an identical list.
    """
    from datasets import load_dataset

    dataset_name = cfg["dataset"]["name"]
    dataset_config = cfg["dataset"].get("config", None)
    split = cfg["dataset"].get("split", "train")
    seq_len = cfg["training"]["seq_len"]
    text_field = cfg["dataset"].get("text_field", "text")

    if Path(dataset_name).exists():
        ds = load_dataset(dataset_name, split=split, streaming=True)
    else:
        ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True)

    chunks = []
    buffer = []
    for example in ds:
        ids = tokenizer(
            example[text_field],
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]
        buffer.extend(ids)
        while len(buffer) >= seq_len + 1:
            chunks.append(buffer[: seq_len + 1])
            buffer = buffer[seq_len:]  # stride == seq_len, matches train.py
            if len(chunks) >= num_chunks:
                return chunks
    return chunks


# ---------------------------------------------------------------------------
# Magnitude profiling: base-2 log-magnitude histograms of weights / activations
# ---------------------------------------------------------------------------
_HIST_LO, _HIST_HI = -40, 40        # one histogram bin per integer power of two
_NBINS = _HIST_HI - _HIST_LO + 1


def _accumulate_log2_hist(tensor, hist):
    """Bin |tensor| into a base-2 log-magnitude histogram (in-place add to hist)."""
    a = tensor.detach().abs().float().flatten()
    a = torch.nan_to_num(a, nan=0.0, posinf=2.0 ** _HIST_HI, neginf=0.0)
    e = torch.log2(a.clamp_min(2.0 ** _HIST_LO))
    idx = e.floor().clamp(_HIST_LO, _HIST_HI).to(torch.long) - _HIST_LO
    hist += torch.bincount(idx, minlength=_NBINS)[:_NBINS]


def _hist_percentile(hist, q):
    """Upper-bound magnitude of the q-quantile, read off the cumulative histogram."""
    total = hist.sum()
    if total.item() == 0:
        return float("nan")
    cum = torch.cumsum(hist, 0).double() / total.double()
    idx = min(int((cum < q).sum().item()), _NBINS - 1)
    return 2.0 ** (idx + _HIST_LO + 1)


class MagnitudeProfiler:
    """Profiles |weight| and |Linear-input activation| for the bitflip-eligible
    Linear layers (same selection the converter uses), so the zero-out
    thresholds can be set above the clean model's legitimate range."""

    def __init__(self, model, cfg, device):
        skip = tuple(cfg.get("bitflip", {}).get("skip_patterns", ["output"]))
        self.eligible = [
            (n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and not any(p in n for p in skip)
        ]
        n = len(self.eligible)
        self.act_hist = torch.zeros(_NBINS, dtype=torch.long, device=device)
        self.w_hist = torch.zeros(_NBINS, dtype=torch.long, device=device)
        self.act_max = torch.full((n,), -1.0, device=device)
        self.w_max = torch.full((n,), -1.0, device=device)
        self._handles = []

    def scan_weights(self):
        """One-shot pass over weights (static, so no forward needed)."""
        for i, (_, m) in enumerate(self.eligible):
            w = m.weight
            wl = w.to_local() if hasattr(w, "to_local") else w  # FSDP shard
            if wl.numel() > 0:
                _accumulate_log2_hist(wl, self.w_hist)
                self.w_max[i] = wl.detach().abs().float().amax()

    def register_hooks(self):
        """Pre-hooks record the input activation of each eligible Linear."""
        for i, (_, m) in enumerate(self.eligible):
            self._handles.append(m.register_forward_pre_hook(self._make_hook(i)))

    def _make_hook(self, i):
        def hook(_module, args):
            x = args[0]
            if torch.is_tensor(x):
                _accumulate_log2_hist(x, self.act_hist)
                self.act_max[i] = torch.maximum(
                    self.act_max[i], x.detach().abs().float().amax()
                )
        return hook

    def finalize(self):
        """Remove hooks and all-reduce stats across ranks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        dist.all_reduce(self.act_hist, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.w_hist, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.act_max, op=dist.ReduceOp.MAX)
        dist.all_reduce(self.w_max, op=dist.ReduceOp.MAX)

    def report(self, cfg):
        bf = cfg.get("bitflip", {})
        names = [n for n, _ in self.eligible]
        logger.info("=" * 60)
        logger.info("MAGNITUDE PROFILE — clean Llama3-70B (no bitflip)")
        self._report_one("activations (inputs to Linear)", self.act_hist,
                         self.act_max, names, bf.get("x_zero_out_t"), "x_zero_out_t")
        self._report_one("weights (Linear .weight)", self.w_hist,
                         self.w_max, names, bf.get("w_zero_out_t"), "w_zero_out_t")
        logger.info("=" * 60)

    @staticmethod
    def _report_one(title, hist, max_vec, names, cur, key):
        gmax = max_vec.max().item()
        q = {p: _hist_percentile(hist, p) for p in (0.5, 0.99, 0.999, 0.9999)}
        topk = torch.topk(max_vec, min(5, len(names)))
        sugg = 2.0 ** math.ceil(math.log2(max(gmax, 1e-9)) + 1.0)  # next pow2 >= 2*max
        verdict = "OK" if (cur is not None and cur >= gmax) else "TOO LOW"
        logger.info(f"  [{title}]")
        logger.info(f"    p50/p99/p99.9/p99.99 (<=): "
                    f"{q[0.5]:.3g} / {q[0.99]:.3g} / {q[0.999]:.3g} / {q[0.9999]:.3g}")
        logger.info(f"    global max |.|: {gmax:.4g}")
        logger.info("    top layers by max:")
        for v, idx in zip(topk.values.tolist(), topk.indices.tolist()):
            logger.info(f"      {v:>11.4g}  {names[idx]}")
        logger.info(f"    {key}: current={cur}  [{verdict}]  "
                    f"min-safe={gmax:.4g}  suggested={sugg:g}")


def evaluate(cfg, num_samples, use_bitflip=False, bitflip_seed=None, profile=False):
    device = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    # --profile measures the clean model; never inject bitflip in that mode.
    if profile and use_bitflip:
        use_bitflip = False
        if rank == 0:
            logger.warning("--profile overrides --bitflip: profiling the CLEAN model")

    # ---- 1. Build plain Llama3 model on meta device ----
    model_flavor = cfg["model"]["flavor"]
    dtype = TORCH_DTYPE_MAP[cfg["training"].get("dtype", "bfloat16")]
    if rank == 0:
        logger.info(f"Building plain Llama3 {model_flavor} on meta device (dtype={dtype})")
    model, model_config = build_model(model_flavor, dtype)

    # ---- 1b. Optionally inject bitflip (r=0 -> NO LoRA adapters) ----
    # Done on the meta-device model, before FSDP, mirroring train.py's order.
    # With r=0 BitFlipLinearLora creates no lora_A/lora_B, so forward reduces
    # to: linear(x, bitflip(weight)) -- the clean base model plus bitflip.
    bf_label = "no bitflip, no LoRA (clean baseline)"
    if use_bitflip:
        from aixsim_models.bitflip.lora_finetune_fsdp.converter import BitFlipLoRAConfig, BitFlipLoRAConverter

        bf = cfg.get("bitflip", {})
        seed = bitflip_seed if bitflip_seed is not None else bf.get("base_seed", 42)
        converter_config = BitFlipLoRAConfig(
            x_p_exp=bf.get("x_p_exp"),
            x_p_frac=bf.get("x_p_frac"),
            x_zero_out_t=bf.get("x_zero_out_t"),
            w_p_exp=bf.get("w_p_exp"),
            w_p_frac=bf.get("w_p_frac"),
            w_zero_out_t=bf.get("w_zero_out_t"),
            r=0,  # r=0 -> no LoRA parameters created at all
            base_seed=seed,
            skip_patterns=tuple(bf.get("skip_patterns", ["output"])),
        )
        replaced = BitFlipLoRAConverter(converter_config).convert(model)
        bf_label = (
            f"bitflip ON, no LoRA (r=0) | w_p_frac={bf.get('w_p_frac')} "
            f"x_p_frac={bf.get('x_p_frac')} seed={seed} | {len(replaced)} layers"
        )
        if rank == 0:
            logger.info(f"Applied {bf_label}")

    # ---- 2. Apply FSDP2 (needed to fit 70B across GPUs) ----
    fsdp_mesh_dim = cfg["parallelism"].get("fsdp_degree", world_size)
    if fsdp_mesh_dim > world_size:
        if rank == 0:
            logger.warning(f"fsdp_degree={fsdp_mesh_dim} > world_size={world_size}, using world_size")
        fsdp_mesh_dim = world_size
    dp_mesh = init_device_mesh("cuda", (fsdp_mesh_dim,), mesh_dim_names=("fsdp",))

    param_dtype = TORCH_DTYPE_MAP[cfg["training"].get("mixed_precision_param", "bfloat16")]
    reduce_dtype = TORCH_DTYPE_MAP[cfg["training"].get("mixed_precision_reduce", "float32")]
    apply_fsdp2(model, dp_mesh, param_dtype, reduce_dtype)

    # Materialize from meta device, then initialize built-in states (RoPE, etc.)
    model.to_empty(device=device)
    with torch.no_grad():
        model.init_states(buffer_device=None)
    model.eval()

    # ---- 3. Load pretrained HF checkpoint weights ----
    hf_model_path = cfg["model"]["hf_model_path"]
    load_hf_weights(model, model_config, hf_model_path, device)
    if rank == 0:
        alloc = torch.cuda.memory_allocated() / 1e9
        logger.info(f"Model ready ({alloc:.1f} GB allocated on rank 0)")

    # ---- 3b. Profiling setup: scan weights now, hook activations for the loop ----
    profiler = None
    if profile:
        profiler = MagnitudeProfiler(model, cfg, device)
        profiler.scan_weights()
        profiler.register_hooks()
        if rank == 0:
            logger.info(f"Profiling {len(profiler.eligible)} Linear layers "
                        f"(weights + input activations)")

    # ---- 4. Tokenizer + collect the first num_samples chunks ----
    from transformers import AutoTokenizer

    tokenizer_path = cfg["model"].get("tokenizer_path", hf_model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Drop any remainder so every rank evaluates an equal share (FSDP needs the
    # same number of collective forward calls on every rank).
    per_rank = num_samples // world_size
    eff_total = per_rank * world_size
    if rank == 0 and eff_total != num_samples:
        logger.warning(
            f"num_samples={num_samples} not divisible by world_size={world_size}; "
            f"evaluating {eff_total} samples instead"
        )
    if rank == 0:
        logger.info(f"Collecting first {eff_total} sequences (seq_len={cfg['training']['seq_len']})...")
    chunks = collect_eval_chunks(cfg, tokenizer, eff_total)
    if len(chunks) < eff_total:
        eff_total = (len(chunks) // world_size) * world_size
        if rank == 0:
            logger.warning(f"Dataset exhausted early; evaluating {eff_total} sequences")
    my_chunks = chunks[rank:eff_total:world_size]  # disjoint slice per rank

    # ---- 5. Forward-only pass; accumulate token-weighted loss ----
    batch_size = cfg["training"]["local_batch_size"]
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    tok_count = torch.zeros((), dtype=torch.float64, device=device)

    with torch.no_grad():
        for i in range(0, len(my_chunks), batch_size):
            batch = my_chunks[i : i + batch_size]
            ids = torch.tensor(batch, dtype=torch.long, device=device)
            input_ids = ids[:, :-1]
            labels = ids[:, 1:]

            logits = model(input_ids).float()  # float32 for a stable loss
            n_tok = labels.numel()
            # mean reduction is numerically stable; re-weight by token count
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )
            loss_sum += loss.double() * n_tok
            tok_count += n_tok

    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(tok_count, op=dist.ReduceOp.SUM)

    mean_loss = (loss_sum / tok_count).item()
    perplexity = math.exp(mean_loss)

    if rank == 0:
        logger.info("=" * 60)
        logger.info("EVAL — original Llama3-70B (no retraining)")
        logger.info(f"  condition:   {bf_label}")
        logger.info(f"  model:       {hf_model_path}")
        logger.info(f"  sequences:   {eff_total}  (seq_len={cfg['training']['seq_len']})")
        logger.info(f"  tokens:      {int(tok_count.item()):,}")
        logger.info(f"  loss (CE):   {mean_loss:.4f}")
        logger.info(f"  perplexity:  {perplexity:.4f}")
        logger.info("=" * 60)

    if profiler is not None:
        profiler.finalize()
        if rank == 0:
            profiler.report(cfg)

    dist.destroy_process_group()


def main():
    args = parse_args()
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    evaluate(cfg, args.num_samples, use_bitflip=args.bitflip,
             bitflip_seed=args.bitflip_seed, profile=args.profile)


if __name__ == "__main__":
    main()
