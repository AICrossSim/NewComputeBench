import os
import logging
from typing import Optional, Any
import datasets as hf_datasets


logger = logging.getLogger(__name__)


def _load_wikitext(dataset_path: str = "Salesforce/wikitext"):
    cache_dir = os.environ.get("HF_HUB_CACHE", None)
    if cache_dir is None:
        logger.warning(
            "HF_HUB_CACHE is not set. Dataset will be stored in the default directory."
        )
    dataset = hf_datasets.load_dataset(
        dataset_path,
        "wikitext-2-raw-v1",
        split="train",
        streaming=True,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    return dataset


def _process_wikipedia_text(sample: dict[str, Any]) -> str:
    """Process Wikipedia dataset sample text."""
    return sample["text"]


def _load_fineweb(
    dataset_path: str = "HuggingFaceFW/fineweb",
    subset_name: Optional[str] = "sample-100BT",
):
    cache_dir = os.environ.get("HF_HUB_CACHE", None)
    if cache_dir is None:
        logger.warning(
            "HF_HUB_CACHE is not set. Dataset will be stored in the default directory."
        )
    dataset = hf_datasets.load_dataset(
        dataset_path,
        name=subset_name,
        split="train",
        streaming=True,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    return dataset


def _process_fineweb_text(sample: dict[str, Any]) -> str:
    return sample["text"]


def register_pretrain_dataset():
    from torchtitan.datasets.hf_datasets import DATASETS, DatasetConfig

    DATASETS["fineweb"] = DatasetConfig(
        path="HuggingFaceFW/fineweb",
        loader=_load_fineweb,
        text_processor=_process_fineweb_text,
    )
    logger.info("FineWeb dataset registered.")
    DATASETS["wikitext"] = DatasetConfig(
        path="Salesforce/wikitext",
        loader=_load_wikitext,
        text_processor=_process_wikipedia_text,
    )
    logger.info("Wikitext dataset registered.")
