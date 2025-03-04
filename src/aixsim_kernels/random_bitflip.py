import math
from typing import Optional, Union

import torch
from torch import Tensor
import triton
import triton.language as tl
from .dtype import TORCH_DTYPE_TO_TRITON
from .utils.constants import PACKAGE_NAME


def calculate_flip_probability(prob_halves) -> float:
    return 0.5**prob_halves


def find_nearest_prob_n_halves(prob: float) -> int:
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
    else:
        # tl.float32
        exp_mask = 0xFF800000  # bin = 1111_1111_1000_0000_0000_0000_0000_0000
        exp_mask = tl.full((1,), exp_mask, dtype=tl.uint32)
    exp_mask = tl.constexpr(exp_mask)
    return exp_mask


@triton.jit
def create_frac_mask(INPUT_DTYPE: tl.constexpr):
    if INPUT_DTYPE == tl.float16:
        frac_mask = 0x3FF  # bin = 0000_0011_1111_1111
        frac_mask = tl.full((1,), frac_mask, dtype=tl.uint16)
    elif INPUT_DTYPE == tl.bfloat16:
        frac_mask = 0x7F  # bin = 0000_0000_0111_1111
        frac_mask = tl.full((1,), frac_mask, dtype=tl.uint16)
    else:
        # tl.float32
        frac_mask = 0x7FFFFF  # bin = 0000_0000_0111_1111_1111_1111_1111_1111
        frac_mask = tl.full((1,), frac_mask, dtype=tl.uint32)
    frac_mask = tl.constexpr(frac_mask)
    return frac_mask


def _get_autotune_configs_forward():
    # small batch, not sure what is the right default cnnfig here.
    block_sizes = [128, 256, 512, 1024]
    stages = [1, 2, 3, 4]

    configs = []
    for bs in block_sizes:
        for s in stages:
            configs.append(triton.Config({"BLOCK_SIZE": bs}, num_stages=s))
    return configs


@triton.autotune(
    configs=_get_autotune_configs_forward(),
    key=["n_elements"],
    use_cuda_graph=False,
)
@triton.jit
def _random_bitflip_forward_kernel(
    x_ptr,
    output_ptr,
    n_elements: int,
    exp_halves: int,  # 0.5 ** exp_halves for exponent bits,
    frac_halves: int,  # 0.5 ** frac_halves for fraction bits
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
        bits_to_flip = ~tl.zeros(x.shape, dtype=BIN_DTYPE)  # all bits set to 1
        bits_to_flip = _cta_random_flip(
            bits_to_flip, offsets, exp_halves, seed_exp, BIN_DTYPE
        )
        exp_mask = _create_sign_exp_mask(INPUT_DTYPE)
        x = x ^ (bits_to_flip & exp_mask)

    # flip frac bits
    if not SKIP_FRAC_FLIP:
        bits_to_flip = ~tl.zeros(x.shape, dtype=BIN_DTYPE)  # all bits set to 1
        bits_to_flip = _cta_random_flip(
            bits_to_flip, offsets, frac_halves, seed_frac, BIN_DTYPE
        )
        frac_mask = create_frac_mask(INPUT_DTYPE)
        x = x ^ (bits_to_flip & frac_mask)

    x = x.to(INPUT_DTYPE, bitcast=True)

    if ENABLE_ZERO_OUT:
        activated = x.abs() < zero_out_threshold
        x = tl.where(activated, x, 0.0)

    # store x
    tl.store(output_ptr + offsets, x, mask=mask)


BIT_FLIP_DTYPE_MAP = {
    torch.float32: tl.uint32,
    torch.float16: tl.uint16,
    torch.bfloat16: tl.uint16,
}


@torch.library.custom_op(
    f"{PACKAGE_NAME}::random_bitflip_forward",
    mutates_args={},
)
def random_bitflip_fn(
    x: Tensor,
    exp_halves: int | None,
    frac_halves: int | None,
    seed_exp: int,
    seed_frac: int,
    zero_out_threshold: float | None,
    train: bool,
) -> tuple[Tensor, int, int]:
    assert x.dtype in BIT_FLIP_DTYPE_MAP
    skip_exp_flip = exp_halves is None
    skip_frac_flip = frac_halves is None
    enable_zero_out = zero_out_threshold is not None
    if (skip_exp_flip and skip_exp_flip) or (not train):
        if enable_zero_out:
            output = torch.where(x.abs() < zero_out_threshold, x, 0.0)
        return output, seed_exp, seed_frac
    else:
        x = x.contiguous()
        output = torch.empty_like(x)
        num_elements = x.numel()
        grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)
        _random_bitflip_forward_kernel[grid](
            x,
            output,
            n_elements=num_elements,
            exp_halves=exp_halves,
            frac_halves=frac_halves,
            seed_exp=seed_exp,
            seed_frac=seed_frac,
            zero_out_threshold=zero_out_threshold if enable_zero_out else 0.0,
            SKIP_EXP_FLIP=skip_exp_flip,
            SKIP_FRAC_FLIP=skip_frac_flip,
            ENABLE_ZERO_OUT=enable_zero_out,
            INPUT_DTYPE=TORCH_DTYPE_TO_TRITON[x.dtype],
            BIN_DTYPE=BIT_FLIP_DTYPE_MAP[x.dtype],
        )
        if not skip_exp_flip:
            seed_exp += math.ceil(exp_halves / 4)
        if not skip_frac_flip:
            seed_frac += math.ceil(frac_halves / 4)

        return output, seed_exp, seed_frac


@torch.library.register_fake(f"{PACKAGE_NAME}::random_bitflip_forward")
def _random_bitflip_forward_fake(
    x: Tensor,
    exp_halves: int | None,
    frac_halves: int | None,
    seed_exp: int,
    seed_frac: int,
    zero_out_threshold: float | None,
    train: bool,
) -> tuple[Tensor, int, int]:
    output = torch.empty_like(x, dtype=x.dtype)
    return output, seed_exp, seed_frac


def _get_autotune_configs_backward():
    block_sizes = [128, 256, 512, 1024]
    stages = [1, 2, 3, 4]

    configs = []
    for bs in block_sizes:
        for s in stages:
            configs.append(triton.Config({"BLOCK_SIZE": bs}, num_stages=s))
    return configs


@triton.autotune(
    configs=_get_autotune_configs_backward(),
    key=["n_elements"],
    use_cuda_graph=False,
)
@triton.jit
def _random_bitflip_zero_outed_backward_kernel(
    x_ptr,
    grad_y_ptr,
    grad_x_ptr,
    n_elements: int,
    exp_halves: int,  # 0.5 ** exp_halves for exponent bits,
    frac_halves: int,  # 0.5 ** frac_halves for fraction bits
    seed_exp: int,
    seed_frac: int,
    zero_out_threshold: float,
    SKIP_EXP_FLIP: tl.constexpr,
    SKIP_FRAC_FLIP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    BIN_DTYPE: tl.constexpr,
    GRAD_DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # load x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(BIN_DTYPE, bitcast=True)

    # flip exp bits
    # random flip using mask: https://stackoverflow.com/a/35796081
    if not SKIP_EXP_FLIP:
        bits_to_flip = ~tl.zeros(x.shape, dtype=BIN_DTYPE)  # all bits set to 1
        bits_to_flip = _cta_random_flip(
            bits_to_flip, offsets, exp_halves, seed_exp, BIN_DTYPE
        )
        exp_mask = _create_sign_exp_mask(INPUT_DTYPE)
        x = x ^ (bits_to_flip & exp_mask)

    # flip frac bits
    if not SKIP_FRAC_FLIP:
        bits_to_flip = ~tl.zeros(x.shape, dtype=BIN_DTYPE)  # all bits set to 1
        bits_to_flip = _cta_random_flip(
            bits_to_flip, offsets, frac_halves, seed_frac, BIN_DTYPE
        )
        frac_mask = create_frac_mask(INPUT_DTYPE)
        x = x ^ (bits_to_flip & frac_mask)

    x = x.to(INPUT_DTYPE, bitcast=True)

    # zero out mask
    activated = x.abs() < zero_out_threshold

    grad_y = tl.load(grad_y_ptr + offsets, mask=mask)
    grad_x = tl.where(activated, grad_y, 0.0).to(GRAD_DTYPE)

    # store grad_x
    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)


@torch.library.custom_op(
    f"{PACKAGE_NAME}::random_bitflip_backward",
    mutates_args={},
)
def _random_bitflip_backward(
    x: Tensor,
    grad_y: Tensor,
    exp_halves: int | None,
    frac_halves: int | None,
    seed_exp: int,
    seed_frac: int,
    zero_out_threshold: float | None,
) -> Tensor:
    assert x.dtype in BIT_FLIP_DTYPE_MAP
    skip_exp_flip = exp_halves is None
    skip_frac_flip = frac_halves is None
    enable_zero_out = zero_out_threshold is not None

    if skip_exp_flip and skip_frac_flip:
        if enable_zero_out:
            grad_x = torch.where(x.abs() < zero_out_threshold, grad_y, 0.0)
        return grad_x
    else:
        if enable_zero_out:
            x = x.contiguous()
            grad_y = grad_y.contiguous()
            grad_x = torch.empty_like(x)
            num_elements = x.numel()
            grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)
            _random_bitflip_zero_outed_backward_kernel[grid](
                x,
                grad_y,
                grad_x,
                n_elements=num_elements,
                exp_halves=exp_halves,
                frac_halves=frac_halves,
                seed_exp=seed_exp,
                seed_frac=seed_frac,
                zero_out_threshold=zero_out_threshold,
                SKIP_EXP_FLIP=skip_exp_flip,
                SKIP_FRAC_FLIP=skip_frac_flip,
                INPUT_DTYPE=TORCH_DTYPE_TO_TRITON[x.dtype],
                BIN_DTYPE=BIT_FLIP_DTYPE_MAP[x.dtype],
                GRAD_DTYPE=TORCH_DTYPE_TO_TRITON[grad_y.dtype],
            )
        else:
            grad_x = grad_y.clone()
        return grad_x


@torch.library.register_fake(f"{PACKAGE_NAME}::random_bitflip_backward")
def _random_bitflip_backward_fake(
    x: Tensor,
    grad_y: Tensor,
    exp_halves: int | None,
    frac_halves: int | None,
    seed_exp: int,
    seed_frac: int,
    zero_out_threshold: float | None,
) -> Tensor:
    grad_x = torch.empty_like(grad_y)
    return grad_x


def _random_bitflip_backward_wrapper(ctx, *grad_outputs):
    exp_halves = ctx.exp_halves
    frac_halves = ctx.frac_halves
    seed_exp = ctx.seed_exp
    seed_frac = ctx.seed_frac
    zero_out_threshold = ctx.zero_out_threshold

    x = ctx.saved_tensors[0]
    grad_input = _random_bitflip_backward(
        x=x,
        grad_y=grad_outputs[0],
        exp_halves=exp_halves,
        frac_halves=frac_halves,
        seed_exp=seed_exp,
        seed_frac=seed_frac,
        zero_out_threshold=zero_out_threshold,
    )
    return grad_input, None, None, None, None, None, None


def _random_bitflip_setup_context(ctx, inputs, output):
    ctx.save_for_backward(inputs[0])
    ctx.exp_halves = inputs[1]
    ctx.frac_halves = inputs[2]
    ctx.seed_exp = inputs[3]
    ctx.seed_frac = inputs[4]
    ctx.zero_out_threshold = inputs[5]


random_bitflip_fn.register_autograd(
    _random_bitflip_backward_wrapper, setup_context=_random_bitflip_setup_context
)


class RandomBitFlip(torch.nn.Module):
    def __init__(
        self,
        p_exp: float | None,
        p_frac: float | None,
        zero_out_threshold: float | None,
        seed_exp: int,
        seed_frac: int,
    ):
        super().__init__()
        self.p_exp = p_exp
        self.p_frac = p_frac
        self.nearest_exp_halves = find_nearest_prob_n_halves(p_exp)
        self.nearest_frac_halves = find_nearest_prob_n_halves(p_frac)
        self.seed_exp = seed_exp
        self.seed_frac = seed_frac
        self.zero_out_threshold = zero_out_threshold

    def forward(self, x: Tensor) -> Tensor:
        out, seed_exp, seed_frac = random_bitflip_fn(
            x,
            exp_halves=self.nearest_exp_halves,
            frac_halves=self.nearest_frac_halves,
            seed_exp=self.seed_exp,
            seed_frac=self.seed_frac,
            zero_out_threshold=self.zero_out_threshold,
            train=self.training,
        )
        self.seed_exp = seed_exp
        self.seed_frac = seed_frac
        return out

    def extra_repr(self) -> str:
        return (
            f"nearest_p_exp={calculate_flip_probability(self.nearest_exp_halves)}, "
            f"nearest_p_frac={calculate_flip_probability(self.nearest_frac_halves)}, "
            f"zero_out_threshold={self.zero_out_threshold}, "
            f"seed_exp={self.seed_exp}, seed_frac={self.seed_frac}"
        )


__all__ = [
    "random_bitflip_fn",
    "RandomBitFlip",
    "find_nearest_prob_n_halves",
    "calculate_flip_probability",
]
