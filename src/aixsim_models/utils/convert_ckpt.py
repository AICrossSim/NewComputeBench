from pathlib import Path
import logging

from torch.distributed.checkpoint.format_utils import (
    dcp_to_torch_save,
    torch_save_to_dcp,
)

logger = logging.getLogger(__name__)


def dcp_to_torch(dpc_ckpt: Path, torch_ckpt: Path):
    assert dpc_ckpt.exists(), f"{dpc_ckpt} does not exist"
    torch_ckpt.parent.mkdir(parents=True, exist_ok=True)
    dcp_to_torch_save(dpc_ckpt, torch_ckpt)
    logger.info(f"Converted {dpc_ckpt} to {torch_ckpt}")


def torch_to_dcp(torch_ckpt: Path, dpc_ckpt: Path):
    assert torch_ckpt.exists(), f"{torch_ckpt} does not exist"
    torch_save_to_dcp(torch_ckpt, dpc_ckpt)
    logger.info(f"Converted {torch_ckpt} to {dpc_ckpt}")
