import os
from pathlib import Path
from typing import Optional
import logging
from requests.exceptions import HTTPError

from huggingface_hub import snapshot_download, hf_hub_download

logger = logging.getLogger(__name__)


def download_hf_tokenizer(repo_id: str, local_dir: str, hf_token: Optional[str] = None):
    # !: Most tokenizers on HuggingFace are sentencepiece tokenizer
    # !: However, only tiktoken tokenizer is supported by TorchTitan
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        token=hf_token,
        local_dir=local_dir,
        cache_dir=os.environ.get("HF_HUB_CACHE", None),
        allow_patterns=["tokenizer*", "generation_config.json", "config.json"],
    )
    logger.info(f"Tokenizer {repo_id} downloaded and stored in {local_dir}")


def download_tiktoken_tokenizer(
    repo_id: str, tokenizer_path: str, local_dir: str, hf_token: Optional[str] = None
) -> None:

    tokenizer_path = (
        f"{tokenizer_path}/tokenizer.model" if tokenizer_path else "tokenizer.model"
    )

    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=tokenizer_path,
            local_dir=local_dir,
            token=hf_token,
        )
    except HTTPError as e:
        if e.response.status_code == 401:
            print(
                "You need to pass a valid `--hf_token=...` to download private checkpoints."
            )
        else:
            raise e
    logger.info(f"TikTokenizer {repo_id} downloaded and stored in {local_dir}")


def download_dataset(
    dataset_name: str,
    cache_dir: Optional[Path] = None,
    allow_pattern: Optional[str] = None,
    symlink_to: Optional[Path] = None,
    hf_token: Optional[str] = None,
) -> None:
    """Download a dataset from HuggingFace Hub and optionally create a symlink.

    This function downloads a dataset from the HuggingFace Hub and stores it locally.
    It can optionally create a symlink to the downloaded dataset directory.

    Args:
        dataset_name (str): The name of the dataset to download from HuggingFace Hub
        cache_dir (Path, optional): Directory to cache the downloaded files. If None,
            uses HF_HUB_CACHE environment variable or default directory.
        allow_pattern (str, optional): If specified, only files matching this pattern
            will be downloaded.
        symlink_to (Path, optional): If specified, creates a symbolic link from the
            downloaded directory to this location.
        hf_token (str, optional): HuggingFace authentication token for private datasets.

    Returns:
        None
    """
    if cache_dir is None:
        cache_dir = os.environ.get("HF_HUB_CACHE", None)
        if cache_dir is None:
            logger.warning(
                "HF_HUB_CACHE is not set. Dataset will be stored in the default directory."
            )

    local_dir = snapshot_download(
        repo_id=dataset_name,
        repo_type="dataset",
        cache_dir=cache_dir,
        allow_patterns=allow_pattern,
        token=hf_token,
    )
    logger.info(f"Dataset downloaded and stored in {local_dir}")
    local_dir = Path(local_dir)
    if symlink_to is not None:
        local_dir.symlink_to(symlink_to)
        logger.info(f"Symlink created from {local_dir} to {symlink_to}")
