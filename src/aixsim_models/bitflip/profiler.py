from typing import Literal, Optional, Union
import logging
from pathlib import Path
import yaml
from pprint import pformat

import torch
import torch.nn as nn
import transformers

logger = logging.getLogger(__name__)


class MagnitudeForwardHookManager:
    SUPPORTED_MODULES = (nn.Linear,)

    def __init__(
        self,
        target_modules=(nn.Linear,),
        percentile: float | None = 0.001,
        avg_topk: bool = True,
        alpha: float | None = 0.98,
    ):
        self.hook_handles = {}
        self.act_stats = {}
        self.weight_stats = {}
        self.meta = {}
        assert 0 <= percentile <= 1
        self.percentile = percentile
        self.avg_topk = avg_topk
        assert 0 <= alpha <= 1
        self.alpha = alpha
        assert all(issubclass(module, self.SUPPORTED_MODULES) for module in target_modules)
        self.target_modules = target_modules

    def get_forward_hook(self, name, module):
        if isinstance(module, nn.Linear):

            def hook(module: nn.Linear, input, output):
                x: torch.Tensor = input[0].flatten()

                if self.percentile is not None:
                    n_elements = x.numel()
                    if self.avg_topk:
                        max_val = x.abs().topk(int(n_elements * self.percentile)).values.mean().item()
                    else:
                        top_k = x.abs().topk(int(n_elements * self.percentile)).values
                        max_val = top_k.max().item()
                else:
                    max_val = x.abs().max().item()

                if self.alpha is not None:
                    if name in self.act_stats:
                        max_val = self.alpha * self.act_stats[name] + (1 - self.alpha) * max_val
                else:
                    if name in self.act_stats:
                        max_val = max(self.act_stats[name], max_val)
                self.act_stats[name] = max_val

                if name not in self.weight_stats:
                    self.weight_stats[name] = {}
                    for w_name, w in module.named_parameters():
                        w = w.flatten()
                        if self.percentile is not None:
                            n_elements = w.numel()
                            if self.avg_topk:
                                max_val = w.abs().topk(int(n_elements * self.percentile)).values.mean().item()
                            else:
                                top_k = w.abs().topk(int(n_elements * self.percentile)).values
                                max_val = top_k.max().item()
                        else:
                            max_val = w.abs().max().item()

                        if self.alpha is not None:
                            if w_name in self.weight_stats[name]:
                                max_val = self.alpha * self.weight_stats[name][w_name] + (1 - self.alpha) * max_val
                        else:
                            if w_name in self.weight_stats:
                                max_val = max(self.weight_stats[name][w_name], max_val)

                    self.weight_stats[name][w_name] = max_val
                if name not in self.meta:
                    self.meta[name] = {"module_type": "Linear", "weight_entries": tuple(module.named_parameters())}

        else:
            raise ValueError(f"Unsupported module type {type(module)}")

        return hook

    def register_hooks(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, self.target_modules):
                self.hook_handles[name] = module.register_forward_hook(self.get_forward_hook(name, module))

    def remove_hooks(self):
        for handle in self.hook_handles.values():
            handle.remove()
        self.hook_handles = {}

    def get_full_stats(self) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, dict[str, str]]]:
        return self.act_stats, self.weight_stats, self.meta

    def get_simple_stats(self) -> dict[str, float]:
        act_stats_by_module_type = {}
        for name, val in self.act_stats.items():
            module_type = self.meta[name]["module_type"]
            if module_type not in act_stats_by_module_type:
                act_stats_by_module_type[module_type] = []
            act_stats_by_module_type[module_type].append(val)
        act_stats_by_module_type = {k: max(v) for k, v in act_stats_by_module_type.items()}

        w_stats_by_module_type = {}
        for name, w_stats in self.weight_stats.items():
            module_type = self.meta[name]["module_type"]
            if module_type not in w_stats_by_module_type:
                w_stats_by_module_type[module_type] = {}
            for w_name, val in w_stats.items():
                if w_name not in w_stats_by_module_type[module_type]:
                    w_stats_by_module_type[module_type][w_name] = []
                w_stats_by_module_type[module_type][w_name].append(val)

        w_stats_by_module_type = {k: {kk: max(vv) for kk, vv in v.items()} for k, v in w_stats_by_module_type.items()}
        return act_stats_by_module_type, w_stats_by_module_type, self.meta


def profile_stats_hf(
    model_name_or_path: Union[str, transformers.PreTrainedModel],
    dtype: Literal["float32", "float16"] = "float16",
    dataset_name: str = "fineweb",
    dataset_subset: str = "HuggingFaceFW/fineweb",
    batch_size: int = 32,
    num_batches: int = 32,
    seq_len: int = 2048,
    target_modules: tuple[str] = ("Linear",),
    percentile: float = 0.001,
    avg_topk: bool = True,
    alpha: float = 0.98,
    collect_full_stats: bool = False,
    save_dir: Optional[Path] = None,
):
    """Calculate perplexity of a Hugging Face model on a given dataset.
    This function loads a pre-trained Hugging Face model and evaluates its perplexity
    on a specified dataset. The perplexity is calculated as exp(average negative log likelihood).
    Args:
        model_name_or_path (str): Name or path of the Hugging Face model to evaluate
        dtype (Literal["float32", "float16", "bfloat16"]): Data type for model computation.
            Defaults to "bfloat16".
        dataset_name (str): Name of the dataset to use. Defaults to "fineweb".
        dataset_subset (str): Specific subset of the dataset. Defaults to "HuggingFaceFW/fineweb".
        batch_size (int): Number of sequences per batch. Defaults to 32.
        num_batches (int): Number of batches to process. Defaults to 32.
        seq_len (int): Length of input sequences. Defaults to 2048.
    Returns:
        None: Prints the calculated perplexity to the logger.
    """
    from tqdm import tqdm
    from aixsim_models.llm.tokenizer import HFTokenizer
    from torchtitan.datasets import build_hf_data_loader

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = HFTokenizer(model_name_or_path)
    if isinstance(model_name_or_path, str):
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=getattr(torch, dtype))
    elif isinstance(model_name_or_path, transformers.PreTrainedModel):
        model = model_name_or_path
    else:
        raise ValueError("Invalid model_name_or_path")
    model.eval().to(device)

    target_modules = tuple(getattr(nn, module) for module in target_modules)
    hook_manager = MagnitudeForwardHookManager(
        target_modules=target_modules, percentile=percentile, avg_topk=avg_topk, alpha=alpha
    )
    hook_manager.register_hooks(model)

    dp_degree, dp_rank = 1, 0
    data_loader = build_hf_data_loader(
        dataset_name=dataset_name,
        dataset_path=dataset_subset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        seq_len=seq_len,
        world_size=dp_degree,
        rank=dp_rank,
        infinite=False,
    )

    nll_sum = 0.0
    n_tokens = 0

    def loss_fn(pred, labels):
        return torch.nn.functional.cross_entropy(pred.flatten(0, 1).float(), labels.flatten(0, 1))

    for i, batch in tqdm(enumerate(data_loader), total=num_batches):
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

        num_valid_tokens = (labels != -100).sum().item()
        batch_size = labels.size(0)
        num_loss_tokens = num_valid_tokens - batch_size
        nll_sum += loss * num_loss_tokens
        n_tokens += num_loss_tokens

        if i >= num_batches:
            break

    avg_nll = nll_sum / n_tokens
    ppl = torch.exp(avg_nll).item()
    logger.info(f"Perplexity: {ppl:.2f}")

    hook_manager.remove_hooks()
    if collect_full_stats:
        act_stats, w_stats, meta = hook_manager.get_full_stats()
    else:
        act_stats, w_stats, meta = hook_manager.get_simple_stats()
        logger.info(f"Activation stats:\n{pformat(act_stats)}")
        logger.info(f"Weight stats:\n{pformat(w_stats)}")
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "stats.yaml", "w") as f:
            yaml.dump(
                {
                    "activation_stats": act_stats,
                    "weight_stats": w_stats,
                    "meta": meta,
                },
                f,
            )
        with open(save_dir / "config.yaml", "w") as f:
            yaml.dump(
                {
                    "model_name_or_path": model_name_or_path,
                    "dtype": dtype,
                    "dataset_name": dataset_name,
                    "dataset_subset": dataset_subset,
                    "batch_size": batch_size,
                    "num_batches": num_batches,
                    "seq_len": seq_len,
                    "target_modules": target_modules,
                    "percentile": percentile,
                    "avg_topk": avg_topk,
                    "alpha": alpha,
                },
                f,
            )
