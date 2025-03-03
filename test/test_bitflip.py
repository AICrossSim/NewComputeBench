import triton
import triton.language as tl

import logging
from difflib import SequenceMatcher

import torch

from aixsim_kernels.random_bitflip import (
    find_nearest_prob_halves,
    _random_bitflip_forward,
    RandomBitFlip,
)
from aixsim_kernels.utils.bit_repr import get_binary_repr
from aixsim_kernels.logging import set_logging_verbosity

logger = logging.getLogger(f"aixsim_kernels.test.{__name__}")

DEVICE = "cuda"


@triton.jit
def bit_flip_kernel(
    x_ptr,
    output_ptr,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
    BIN_DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # load x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(BIN_DTYPE, bitcast=True)
    x = ~x
    # store x
    tl.store(output_ptr + offsets, x, mask=mask)


def bit_flip(x, BLOCK_SIZE, BIN_DTYPE):
    n_elements = x.numel()
    output = torch.zeros_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    bit_flip_kernel[grid](x, output, n_elements, BLOCK_SIZE, BIN_DTYPE)
    return output


def test_bitflip():
    x = torch.zeros(4, device=DEVICE, dtype=torch.float32)
    out = bit_flip(x, 32, tl.uint32)
    logger.info(f"x: {x}")
    logger.info(f"out: {out}")


if __name__ == "__main__":
    set_logging_verbosity(logging.DEBUG)
    test_bitflip()
