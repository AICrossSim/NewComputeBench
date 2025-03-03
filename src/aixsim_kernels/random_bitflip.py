import math
from typing import Optional, Union

import torch
from torch import Tensor
import triton
import triton.language as tl
from .dtype import TORCH_DTYPE_TO_TRITON


def calculate_flip_probability(prob_halves) -> float:
    return 0.5**prob_halves


def find_nearest_prob_halves(prob: float) -> int:
    return math.ceil(math.log2(1 / prob))


@triton.jit
def _get_four_randints(seed, offsets, bin_dtype):
    rint1, rint2, rint3, rint4 = tl.randint4x(seed, offsets)
    rint1 = rint1.to(tl.uint32, bitcast=True).to(bin_dtype)
    rint2 = rint2.to(tl.uint32, bitcast=True).to(bin_dtype)
    rint3 = rint3.to(tl.uint32, bitcast=True).to(bin_dtype)
    rint4 = rint4.to(tl.uint32, bitcast=True).to(bin_dtype)
    return rint1, rint2, rint3, rint4


@triton.jit
def _cta_random_flip(
    set_bits, offsets, prob_halves: int, seed: int, BIN_DTYPE: tl.constexpr
):
    q = prob_halves // 4
    r = prob_halves % 4
    for i in range(q):
        rint1, rint2, rint3, rint4 = _get_four_randints(seed + i, offsets, BIN_DTYPE)
        set_bits = set_bits & rint1 & rint2 & rint3 & rint4
    rint1, rint2, rint3, _ = _get_four_randints(seed + q, offsets, BIN_DTYPE)
    if r >= 1:
        set_bits = set_bits & rint1
    if r >= 2:
        set_bits = set_bits & rint2
    if r >= 3:
        set_bits = set_bits & rint3
    return set_bits


@triton.jit
def _create_sign_exp_mask(INPUT_DTYPE: tl.constexpr):
    if INPUT_DTYPE == tl.float16:
        exp_mask = 0xFC00  # bin = 1111_1100_0000_0000
        exp_mask = tl.full((1,), exp_mask, dtype=tl.uint16)
    elif INPUT_DTYPE == tl.bfloat16:
        exp_mask = 0xFF80  # bin = 1111_1111_1000_0000
        exp_mask = tl.full((1,), exp_mask, dtype=tl.uint16)
    elif INPUT_DTYPE == tl.float32:
        exp_mask = 0xFF800000  # bin = 1111_1111_1000_0000_0000_0000_0000_0000
        exp_mask = tl.full((1,), exp_mask, dtype=tl.uint32)
    else:
        # this branch should not be reached
        exp_mask = 0
        exp_mask = tl.full((1,), exp_mask, dtype=tl.uint32)
    return exp_mask


@triton.jit
def create_frac_mask(INPUT_DTYPE: tl.constexpr):
    if INPUT_DTYPE == tl.float16:
        frac_mask = 0x3FF  # bin = 0000_0011_1111_1111
        frac_mask = tl.full((1,), frac_mask, dtype=tl.uint16)
    elif INPUT_DTYPE == tl.bfloat16:
        frac_mask = 0x7F  # bin = 0000_0000_0111_1111
        frac_mask = tl.full((1,), frac_mask, dtype=tl.uint16)
    elif INPUT_DTYPE == tl.float32:
        frac_mask = 0x7FFFFF  # bin = 0000_0000_0111_1111_1111_1111_1111_1111
        frac_mask = tl.full((1,), frac_mask, dtype=tl.uint32)
    else:
        # this branch should not be reached
        frac_mask = 0xFFFFFFFF
        frac_mask = tl.full((1,), frac_mask, dtype=tl.uint32)
    return frac_mask


def _get_autotune_configs():
    # small batch, not sure what is the right default cnnfig here.
    block_sizes = [128, 256, 512, 1024]
    configs = []
    for bs in block_sizes:
        configs.append(triton.Config({"BLOCK_SIZE": bs}))
    return configs


# @triton.autotune(
#     configs=_get_autotune_configs(),
#     key=["BLOCK_SIZE"],
#     use_cuda_graph=False,
# )
@triton.jit
def _random_bitflip_kernel(
    x_ptr,
    output_ptr,
    n_elements: int,
    exp_n_halves: int,  # 0.5 ** exp_n_halves for exponent bits,
    frac_n_halves: int,  # 0.5 ** frac_n_halves for fraction bits
    seed_exp: int,
    seed_frac: int,
    zero_out_threshold: float,
    SKIP_EXP_FLIP: tl.constexpr,
    SKIP_FRAC_FLIP: tl.constexpr,
    ENABLE_ZERO_OUT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    BIN_DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # load x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(BIN_DTYPE, bitcast=True)

    # flip exp bits
    # random flip using mask: https://stackoverflow.com/a/35796081
    if not SKIP_EXP_FLIP:
        bits_to_flip = tl.zeros(x.shape, dtype=BIN_DTYPE) - 1  # all bits set to 1
        bits_to_flip = _cta_random_flip(
            bits_to_flip, offsets, exp_n_halves, seed_exp, BIN_DTYPE
        )
        exp_mask = _create_sign_exp_mask(INPUT_DTYPE)
        x = x ^ (bits_to_flip & exp_mask)

    # flip frac bits
    if not SKIP_FRAC_FLIP:
        bits_to_flip = tl.zeros(x.shape, dtype=BIN_DTYPE) - 1  # all bits set to 1
        bits_to_flip = _cta_random_flip(
            bits_to_flip, offsets, frac_n_halves, seed_frac, BIN_DTYPE
        )
        frac_mask = create_frac_mask(INPUT_DTYPE)
        x = x ^ (bits_to_flip & frac_mask)

    x = x.to(INPUT_DTYPE, bitcast=True)

    if ENABLE_ZERO_OUT:
        # threshold = tl.full((1,), zero_out_threshold, dtype=INPUT_DTYPE)
        x = tl.where(x.abs() < zero_out_threshold, x, 0.0)

    # store x
    tl.store(output_ptr + offsets, x, mask=mask)


BIT_FLIP_DTYPE_MAP = {
    torch.float32: tl.uint32,
    torch.float16: tl.uint16,
    torch.bfloat16: tl.uint16,
    torch.int32: tl.uint32,
    torch.uint32: tl.uint32,
    torch.int16: tl.uint16,
    torch.uint16: tl.uint16,
    torch.int8: tl.uint8,
    torch.uint8: tl.uint8,
}


def _random_bitflip_forward(
    x: Tensor,
    exp_n_halves: int | None,
    frac_n_halves: int | None,
    seed_exp: int,
    seed_frac: int,
    zero_out_threshold: float | None,
    train: bool,
) -> tuple[Tensor, int, int]:
    if (exp_n_halves is None and frac_n_halves is None) or (not train):
        return x.clone(), seed_exp, seed_frac
    else:
        x = x.contiguous()
        output = torch.empty_like(x)
        num_elements = x.numel()
        grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)
        _random_bitflip_kernel[grid](
            x,
            output,
            n_elements=num_elements,
            exp_n_halves=exp_n_halves,
            frac_n_halves=frac_n_halves,
            seed_exp=seed_exp,
            seed_frac=seed_frac,
            zero_out_threshold=(
                zero_out_threshold if zero_out_threshold is not None else 0.0
            ),
            SKIP_EXP_FLIP=exp_n_halves is None,
            SKIP_FRAC_FLIP=frac_n_halves is None,
            ENABLE_ZERO_OUT=zero_out_threshold is not None,
            BLOCK_SIZE=1024,
            INPUT_DTYPE=TORCH_DTYPE_TO_TRITON[x.dtype],
            BIN_DTYPE=BIT_FLIP_DTYPE_MAP[x.dtype],
        )
        if exp_n_halves is not None:
            seed_exp += math.ceil(exp_n_halves / 4)
        if frac_n_halves is not None:
            seed_frac += math.ceil(frac_n_halves / 4)

        return output, seed_exp, seed_frac


class _RandomBitFlipFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, prob_halves: float, train: bool, seed: int) -> Tensor:
        return _random_bitflip_forward(x, prob_halves, train, seed)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None, None


def random_bitflip(x: Tensor, prob_halves: float, train: bool, seed: int) -> Tensor:
    return _RandomBitFlipFn.apply(x, prob_halves, train, seed)


class RandomBitFlip(torch.nn.Module):
    def __init__(self, p: float, seed: int, zero_out_invalid: bool):
        super().__init__()
        self.p = p
        self.prob_halves = find_nearest_prob_halves(p)
        self.p_real = calculate_flip_probability(self.prob_halves)
        self.seed = seed
        self.zero_out_invalid = zero_out_invalid

    def forward(self, input: Tensor) -> Tensor:
        output = random_bitflip(input, self.prob_halves, self.training, self.seed)
        self.seed += self.prob_halves // 4 + self.prob_halves % 4
        if self.zero_out_invalid:
            output = torch.where(torch.isfinite(output), output, 0.0)
        return output

    def extra_repr(self):
        return (
            f"p={self.p}, p_real={self.p_real}, seed={self.seed}, train={self.training}"
        )


def zero_out(x: Tensor, abs_threshold: Optional[float] = None) -> Tensor:
    if abs_threshold is None:
        return torch.where(torch.isfinite(x), x, 0.0)
    else:
        return torch.where(torch.abs(x) < abs_threshold, x, 0.0)
