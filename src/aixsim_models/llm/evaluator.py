from pathlib import Path
from typing import Literal, Union
import logging

import torch

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed._tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from tqdm import tqdm
import transformers

from torchtitan.datasets import build_hf_data_loader
from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config

from aixsim_models.llm import build_tokenizer

logger = logging.getLogger(__name__)


def apply_tp_minus_sp(model: nn.Module, tp_mesh: DeviceMesh):
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(input_layouts=Replicate()),
            "output": ColwiseParallel(output_layouts=Replicate()),
        },
    )

    for _, transformer_block in model.layers.items():
        layer_plan = {
            "attention.wq": ColwiseParallel(),
            "attention.wk": ColwiseParallel(),
            "attention.wv": ColwiseParallel(),
            "attention.wo": RowwiseParallel(),
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(),
            "feed_forward.w3": ColwiseParallel(),
        }

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )


def evaluate_ppl(
    model_arch: Literal["aixsim", "llama"],
    model_flavor: str,
    tokenizer_path: Path,
    checkpoint_path: Path,
    dataset_name: str = "fineweb",
    dataset_subset: str = "HuggingFaceFW/fineweb",
    batch_size: int = 32,
    num_batches: int = 32,
    seq_len: int = 2048,
):
    """
    Evaluate the perplexity of a language model.

    Args:
        model_arch (Literal["aixsim", "llama"]): The architecture of the model.
        model_flavor (str): The specific flavor or variant of the model.
        tokenizer_path (Path): Path to the tokenizer.
        checkpoint_path (Path): Path to the model checkpoint.
        dataset_name (str, optional): Name of the dataset. Defaults to "fineweb".
        dataset_subset (str, optional): Subset of the dataset. Defaults to "HuggingFaceFW/fineweb".
        batch_size (int, optional): Batch size for evaluation. Defaults to 32.
        num_batches (int, optional): Number of batches to evaluate. Defaults to 32.
        seq_len (int, optional): Sequence length for evaluation. Defaults to 2048.

    Raises:
        AssertionError: If the checkpoint path does not exist.

    Returns:
        None
    """
    assert (
        checkpoint_path.exists()
    ), f"Checkpoint path {checkpoint_path} does not exist."

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = build_tokenizer(model_name_to_tokenizer[model_arch], tokenizer_path)

    model_config = models_config[model_arch][model_flavor]
    model_config.vocab_size = tokenizer.n_words
    model_config.max_seq_len = seq_len
    model_cls = model_name_to_cls[model_arch]
    model = model_cls.from_model_args(model_config)
    if checkpoint_path.is_file():
        state_dict = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(state_dict["model"])
        logger.info(f"Loaded bin checkpoint from {checkpoint_path}")
    else:
        state_dict = {"model": model.state_dict()}
        dcp.load(state_dict, checkpoint_id=checkpoint_path)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    model.eval().to(device)

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
        return torch.nn.functional.cross_entropy(
            pred.flatten(0, 1).float(), labels.flatten(0, 1)
        )

    for i, batch in tqdm(enumerate(data_loader), total=num_batches):
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits = model(input_ids)
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


def check_hf_ppl(
    model_name_or_path: Union[str, transformers.PreTrainedModel],
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16",
    dataset_name: str = "fineweb",
    dataset_subset: str = "HuggingFaceFW/fineweb",
    batch_size: int = 32,
    num_batches: int = 32,
    seq_len: int = 2048,
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
    from aixsim_models.llm.tokenizer import HFTokenizer

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = HFTokenizer(model_name_or_path)
    if isinstance(model_name_or_path, str):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=getattr(torch, dtype)
        )
    elif isinstance(model_name_or_path, transformers.PreTrainedModel):
        model = model_name_or_path
    else:
        raise ValueError("Invalid model_name_or_path")
    model.eval().to(device)

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
        return torch.nn.functional.cross_entropy(
            pred.flatten(0, 1).float(), labels.flatten(0, 1)
        )

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


def hf_generate(
    model_name: str,
    prompt: str,
    max_new_tokens: int = 128,
    dtype: Literal["float32", "float16", "bfloat16"] = "float32",
    seed: int = 0,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float = 0.9,
):
    """
    Generate text using a Hugging Face model.

    Args:
        model_name (str): The name of the pre-trained model to use.
        prompt (str): The initial text prompt to generate text from.
        max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 128.
        dtype (Literal["float32", "float16", "bfloat16"], optional): The data type for model weights. Defaults to "float32".
        seed (int, optional): The random seed for reproducibility. Defaults to 0.
        do_sample (bool, optional): Whether to use sampling; if False, uses greedy decoding. Defaults to True.
        temperature (float, optional): The sampling temperature; higher values mean more random generations. Defaults to 1.0.
        top_p (float, optional): The cumulative probability for nucleus sampling. Defaults to 0.9.

    Returns:
        None
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=getattr(torch, dtype)
    )
    device = next(model.parameters()).device
    set_seed(seed)
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    input_length = model_inputs.input_ids.shape[1]
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    print(f"🔣\tPrompt:\n{prompt}")
    response = tokenizer.batch_decode(
        generated_ids[:, input_length:], skip_special_tokens=True
    )[0]

    print(f"🔮\tPrompt + Response:\n{prompt}{response}")
