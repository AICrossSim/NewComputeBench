"""Compile-safe random bitflip Triton kernels.

Replicates the mase-triton bitflip logic but with stateless seed management
so the kernels work under torch.compile (no mutable module attributes).

Seeds are passed in explicitly and updated seeds are returned; the caller
is responsible for deriving per-step seeds from a global step counter.
"""

import math

import torch
import triton
import triton.language as tl
from torch import Tensor

# ---------------------------------------------------------------------------
# Triton dtype mapping
# ---------------------------------------------------------------------------
TORCH_DTYPE_TO_TRITON = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}

BIT_FLIP_DTYPE_MAP = {
    torch.float32: tl.uint32,
    torch.float16: tl.uint16,
    torch.bfloat16: tl.uint16,
}


# ---------------------------------------------------------------------------
# Triton JIT helpers (same logic as mase-triton)
# ---------------------------------------------------------------------------
@triton.jit
def _get_four_randints(seed, offsets, BIN_DTYPE: tl.constexpr, N_ROUNDS: tl.constexpr):
    rint1, rint2, rint3, rint4 = tl.randint4x(seed, offsets, n_rounds=N_ROUNDS)
    rint1 = rint1.to(tl.uint32, bitcast=True).to(BIN_DTYPE)
    rint2 = rint2.to(tl.uint32, bitcast=True).to(BIN_DTYPE)
    rint3 = rint3.to(tl.uint32, bitcast=True).to(BIN_DTYPE)
    rint4 = rint4.to(tl.uint32, bitcast=True).to(BIN_DTYPE)
    return rint1, rint2, rint3, rint4


@triton.jit
def _cta_random_flip(
    set_bits,
    offsets,
    prob_halves: int,
    seed: int,
    BIN_DTYPE: tl.constexpr,
    PHILOX_N_ROUNDS: tl.constexpr,
):
    q = prob_halves // 4
    r = prob_halves % 4
    for i in range(q):
        rint1, rint2, rint3, rint4 = _get_four_randints(
            seed + i, offsets, BIN_DTYPE, PHILOX_N_ROUNDS
        )
        set_bits = set_bits & rint1 & rint2 & rint3 & rint4
    rint1, rint2, rint3, _ = _get_four_randints(
        seed + q, offsets, BIN_DTYPE, PHILOX_N_ROUNDS
    )
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
        exp_mask = 0xFC00
        exp_mask = tl.full((1,), exp_mask, dtype=tl.uint16)
    elif INPUT_DTYPE == tl.bfloat16:
        exp_mask = 0xFF80
        exp_mask = tl.full((1,), exp_mask, dtype=tl.uint16)
    else:
        exp_mask = 0xFF800000
        exp_mask = tl.full((1,), exp_mask, dtype=tl.uint32)
    exp_mask = tl.constexpr(exp_mask)
    return exp_mask


@triton.jit
def _create_frac_mask(INPUT_DTYPE: tl.constexpr):
    if INPUT_DTYPE == tl.float16:
        frac_mask = 0x3FF
        frac_mask = tl.full((1,), frac_mask, dtype=tl.uint16)
    elif INPUT_DTYPE == tl.bfloat16:
        frac_mask = 0x7F
        frac_mask = tl.full((1,), frac_mask, dtype=tl.uint16)
    else:
        frac_mask = 0x7FFFFF
        frac_mask = tl.full((1,), frac_mask, dtype=tl.uint32)
    frac_mask = tl.constexpr(frac_mask)
    return frac_mask


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------
def _get_autotune_configs():
    block_sizes = [128, 256, 512, 1024]
    stages = [1, 2, 3, 4]
    configs = []
    for bs in block_sizes:
        for s in stages:
            configs.append(triton.Config({"BLOCK_SIZE": bs}, num_stages=s))
    return configs


@triton.autotune(configs=_get_autotune_configs(), key=["n_elements"], use_cuda_graph=False)
@triton.jit
def _random_bitflip_forward_kernel(
    x_ptr,
    output_ptr,
    n_elements: int,
    exp_halves: int,
    frac_halves: int,
    seed_exp: int,
    seed_frac: int,
    zero_out_threshold: float,
    SKIP_EXP_FLIP: tl.constexpr,
    SKIP_FRAC_FLIP: tl.constexpr,
    ENABLE_ZERO_OUT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    BIN_DTYPE: tl.constexpr,
    EXP_PHILOX_N_ROUNDS: tl.constexpr,
    FRAC_PHILOX_N_ROUNDS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(BIN_DTYPE, bitcast=True)

    if not SKIP_EXP_FLIP:
        bits_to_flip = ~tl.zeros(x.shape, dtype=BIN_DTYPE)
        bits_to_flip = _cta_random_flip(
            bits_to_flip, offsets, exp_halves, seed_exp, BIN_DTYPE, EXP_PHILOX_N_ROUNDS
        )
        exp_mask = _create_sign_exp_mask(INPUT_DTYPE)
        x = x ^ (bits_to_flip & exp_mask)

    if not SKIP_FRAC_FLIP:
        bits_to_flip = ~tl.zeros(x.shape, dtype=BIN_DTYPE)
        bits_to_flip = _cta_random_flip(
            bits_to_flip, offsets, frac_halves, seed_frac, BIN_DTYPE, FRAC_PHILOX_N_ROUNDS
        )
        frac_mask = _create_frac_mask(INPUT_DTYPE)
        x = x ^ (bits_to_flip & frac_mask)

    x = x.to(INPUT_DTYPE, bitcast=True)

    if ENABLE_ZERO_OUT:
        activated = x.abs() < zero_out_threshold
        x = tl.where(activated, x, 0.0)

    tl.store(output_ptr + offsets, x, mask=mask)


# ---------------------------------------------------------------------------
# Backward kernel (zero-out gradient masking)
# ---------------------------------------------------------------------------
@triton.autotune(configs=_get_autotune_configs(), key=["n_elements"], use_cuda_graph=False)
@triton.jit
def _random_bitflip_zero_outed_backward_kernel(
    x_ptr,
    grad_y_ptr,
    grad_x_ptr,
    n_elements: int,
    exp_halves: int,
    frac_halves: int,
    seed_exp: int,
    seed_frac: int,
    zero_out_threshold: float,
    SKIP_EXP_FLIP: tl.constexpr,
    SKIP_FRAC_FLIP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    BIN_DTYPE: tl.constexpr,
    GRAD_DTYPE: tl.constexpr,
    EXP_PHILOX_N_ROUNDS: tl.constexpr,
    FRAC_PHILOX_N_ROUNDS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(BIN_DTYPE, bitcast=True)

    if not SKIP_EXP_FLIP:
        bits_to_flip = ~tl.zeros(x.shape, dtype=BIN_DTYPE)
        bits_to_flip = _cta_random_flip(
            bits_to_flip, offsets, exp_halves, seed_exp, BIN_DTYPE, EXP_PHILOX_N_ROUNDS
        )
        exp_mask = _create_sign_exp_mask(INPUT_DTYPE)
        x = x ^ (bits_to_flip & exp_mask)

    if not SKIP_FRAC_FLIP:
        bits_to_flip = ~tl.zeros(x.shape, dtype=BIN_DTYPE)
        bits_to_flip = _cta_random_flip(
            bits_to_flip, offsets, frac_halves, seed_frac, BIN_DTYPE, FRAC_PHILOX_N_ROUNDS
        )
        frac_mask = _create_frac_mask(INPUT_DTYPE)
        x = x ^ (bits_to_flip & frac_mask)

    x = x.to(INPUT_DTYPE, bitcast=True)
    activated = x.abs() < zero_out_threshold

    grad_y = tl.load(grad_y_ptr + offsets, mask=mask)
    grad_x = tl.where(activated, grad_y, 0.0).to(GRAD_DTYPE)
    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)


# ---------------------------------------------------------------------------
# Probability / seed helpers
# ---------------------------------------------------------------------------
def find_nearest_prob_n_halves(prob: float | None) -> int | None:
    if prob is None:
        return None
    assert 0 < prob < 1
    return math.ceil(math.log2(1 / prob))


def calculate_flip_probability(prob_halves: int | None) -> float | None:
    if prob_halves is None:
        return None
    assert prob_halves > 0
    return 0.5**prob_halves


def _get_philox_n_rounds(n_halves: int | None) -> int:
    if n_halves is None:
        return 0
    if n_halves < 13:
        return 10
    elif n_halves < 19:
        return 12
    elif n_halves < 25:
        return 16
    else:
        return 30


# ---------------------------------------------------------------------------
# custom_op wrappers (torch.compile compatible)
# ---------------------------------------------------------------------------
@torch.library.custom_op("bitflip_lora::random_bitflip_forward", mutates_args={})
def random_bitflip_fn(
    x: Tensor,
    exp_halves: int,
    frac_halves: int,
    seed_exp: int,
    seed_frac: int,
    zero_out_threshold: float,
    skip_exp: bool,
    skip_frac: bool,
    enable_zero_out: bool,
) -> Tensor:
    """Stateless random bitflip forward.

    Unlike the mase-triton version, seeds are passed in and NOT mutated.
    The caller derives seeds from (base_seed + step * stride + layer_offset).
    """
    if skip_exp and skip_frac:
        if enable_zero_out:
            return torch.where(x.abs() < zero_out_threshold, x, 0.0)
        return x.clone()

    x = x.contiguous()
    output = torch.empty_like(x)
    n = x.numel()

    def grid(meta):
        return (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    with torch.cuda.device(x.device.index):
        _random_bitflip_forward_kernel[grid](
            x,
            output,
            n_elements=n,
            exp_halves=exp_halves if not skip_exp else 0,
            frac_halves=frac_halves if not skip_frac else 0,
            seed_exp=seed_exp,
            seed_frac=seed_frac,
            zero_out_threshold=zero_out_threshold if enable_zero_out else 0.0,
            SKIP_EXP_FLIP=skip_exp,
            SKIP_FRAC_FLIP=skip_frac,
            ENABLE_ZERO_OUT=enable_zero_out,
            INPUT_DTYPE=TORCH_DTYPE_TO_TRITON[x.dtype],
            BIN_DTYPE=BIT_FLIP_DTYPE_MAP[x.dtype],
            EXP_PHILOX_N_ROUNDS=_get_philox_n_rounds(exp_halves if not skip_exp else None),
            FRAC_PHILOX_N_ROUNDS=_get_philox_n_rounds(frac_halves if not skip_frac else None),
        )
    return output


@random_bitflip_fn.register_fake
def _random_bitflip_forward_fake(
    x: Tensor,
    exp_halves: int,
    frac_halves: int,
    seed_exp: int,
    seed_frac: int,
    zero_out_threshold: float,
    skip_exp: bool,
    skip_frac: bool,
    enable_zero_out: bool,
) -> Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("bitflip_lora::random_bitflip_backward", mutates_args={})
def _random_bitflip_backward(
    x: Tensor,
    grad_y: Tensor,
    exp_halves: int,
    frac_halves: int,
    seed_exp: int,
    seed_frac: int,
    zero_out_threshold: float,
    skip_exp: bool,
    skip_frac: bool,
    enable_zero_out: bool,
) -> Tensor:
    if skip_exp and skip_frac:
        if enable_zero_out:
            return torch.where(x.abs() < zero_out_threshold, grad_y, 0.0)
        return grad_y.clone()

    if enable_zero_out:
        x = x.contiguous()
        grad_y = grad_y.contiguous()
        grad_x = torch.empty_like(x)
        n = x.numel()

        def grid(meta):
            return (triton.cdiv(n, meta["BLOCK_SIZE"]),)

        with torch.cuda.device(x.device.index):
            _random_bitflip_zero_outed_backward_kernel[grid](
                x,
                grad_y,
                grad_x,
                n_elements=n,
                exp_halves=exp_halves if not skip_exp else 0,
                frac_halves=frac_halves if not skip_frac else 0,
                seed_exp=seed_exp,
                seed_frac=seed_frac,
                zero_out_threshold=zero_out_threshold,
                SKIP_EXP_FLIP=skip_exp,
                SKIP_FRAC_FLIP=skip_frac,
                INPUT_DTYPE=TORCH_DTYPE_TO_TRITON[x.dtype],
                BIN_DTYPE=BIT_FLIP_DTYPE_MAP[x.dtype],
                GRAD_DTYPE=TORCH_DTYPE_TO_TRITON[grad_y.dtype],
                EXP_PHILOX_N_ROUNDS=_get_philox_n_rounds(exp_halves if not skip_exp else None),
                FRAC_PHILOX_N_ROUNDS=_get_philox_n_rounds(frac_halves if not skip_frac else None),
            )
        return grad_x
    else:
        return grad_y.clone()


@_random_bitflip_backward.register_fake
def _random_bitflip_backward_fake(
    x: Tensor,
    grad_y: Tensor,
    exp_halves: int,
    frac_halves: int,
    seed_exp: int,
    seed_frac: int,
    zero_out_threshold: float,
    skip_exp: bool,
    skip_frac: bool,
    enable_zero_out: bool,
) -> Tensor:
    return torch.empty_like(grad_y)


def _backward_wrapper(ctx, grad_output):
    x = ctx.saved_tensors[0]
    grad_input = _random_bitflip_backward(
        x=x,
        grad_y=grad_output,
        exp_halves=ctx.exp_halves,
        frac_halves=ctx.frac_halves,
        seed_exp=ctx.seed_exp,
        seed_frac=ctx.seed_frac,
        zero_out_threshold=ctx.zero_out_threshold,
        skip_exp=ctx.skip_exp,
        skip_frac=ctx.skip_frac,
        enable_zero_out=ctx.enable_zero_out,
    )
    return grad_input, None, None, None, None, None, None, None, None


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(inputs[0])  # x
    ctx.exp_halves = inputs[1]
    ctx.frac_halves = inputs[2]
    ctx.seed_exp = inputs[3]
    ctx.seed_frac = inputs[4]
    ctx.zero_out_threshold = inputs[5]
    ctx.skip_exp = inputs[6]
    ctx.skip_frac = inputs[7]
    ctx.enable_zero_out = inputs[8]


random_bitflip_fn.register_autograd(_backward_wrapper, setup_context=_setup_context)
