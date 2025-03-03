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


def _count_matched_bits(a: str, b: str):
    assert len(a) == len(b)
    return sum([1 for i in range(len(a)) if a[i] == b[i]])


def test_random_bitflip_forward_simple():
    x = torch.zeros(4, device=DEVICE, dtype=torch.bfloat16)
    exp_halves = 4
    frac_halves = 1
    seed_exp, seed_frac = 0, 0
    out, seed_exp, seed_frac = _random_bitflip_forward(
        x,
        exp_n_halves=exp_halves,
        frac_n_halves=frac_halves,
        seed_exp=seed_exp,
        seed_frac=seed_frac,
        zero_out_threshold=None,
        train=True,
    )
    logger.info(f"binary x:\n{get_binary_repr(x, splitter='')}")
    logger.info(f"binary out:\n{get_binary_repr(out, splitter='')}")
    logger.info(f"seed_exp: {seed_exp}, seed_frac: {seed_frac}")


def test_random_bitflip_forward_bitstring():
    input_dtypes = [torch.float32, torch.float16, torch.bfloat16]
    dtype_to_bit_split = {
        torch.float32: (8, 23),
        torch.float16: (5, 10),
        torch.bfloat16: (8, 7),
    }
    s_exp_halves_frac_halves = [(0.5, 0.5), (0.5**4, 0.5**2), (0.5**3.8, 0.5**10)]
    max_tries = 1000

    for input_dtype in input_dtypes:
        x = torch.randn(1024, 1024, device=DEVICE, dtype=input_dtype)
        cur_try = 0
        for exp_p, frac_p in s_exp_halves_frac_halves:
            exp_n_halves = find_nearest_prob_halves(exp_p)
            frac_n_halves = find_nearest_prob_halves(frac_p)
            seed_exp, seed_frac = 0, 0
            logger.info(
                f"====== input_dtype = {input_dtype}, exp_p = {exp_p}, frac_p = {frac_p}, exp_n_halves = {exp_n_halves}, frac_n_halves = {frac_n_halves}, train = True ====="
            )
            while True:
                out, seed_exp, seed_frac = _random_bitflip_forward(
                    x,
                    exp_n_halves=exp_n_halves,
                    frac_n_halves=frac_n_halves,
                    seed_exp=seed_exp,
                    seed_frac=seed_frac,
                    zero_out_threshold=None,
                    train=True,
                )
                find_bitflip = not torch.equal(x, out)
                if find_bitflip:
                    x_bin = get_binary_repr(x, splitter="").tolist()
                    out_bin = get_binary_repr(out, splitter="").tolist()
                    first_n_bits = dtype_to_bit_split[input_dtype][0]
                    last_n_bits = dtype_to_bit_split[input_dtype][1]
                    x_s_exp_bin = []
                    x_frac_bin = []
                    out_s_exp_bin = []
                    out_frac_bin = []
                    for i in range(len(x_bin)):
                        x_s_exp_bin += [el[:first_n_bits] for el in x_bin[i]]
                        x_frac_bin += [el[-last_n_bits:] for el in x_bin[i]]
                        out_s_exp_bin += [el[:first_n_bits] for el in out_bin[i]]
                        out_frac_bin += [el[-last_n_bits:] for el in out_bin[i]]
                    x_s_exp_bin = "".join(x_s_exp_bin)
                    x_frac_bin = "".join(x_frac_bin)
                    out_s_exp_bin = "".join(out_s_exp_bin)
                    out_frac_bin = "".join(out_frac_bin)

                    s_exp_match_ratio = _count_matched_bits(x_s_exp_bin, out_s_exp_bin)
                    s_exp_match_ratio /= len(x_s_exp_bin)
                    s_exp_mismatch_ratio = 1 - s_exp_match_ratio
                    frac_match_ratio = _count_matched_bits(x_frac_bin, out_frac_bin)
                    frac_match_ratio /= len(x_frac_bin)
                    frac_mismatch_ratio = 1 - frac_match_ratio
                    logger.info(f"Flip found in {cur_try} tries")
                    logger.info(f"sign_exp mismatch ratio: {s_exp_mismatch_ratio}")
                    logger.info(f"frac mismatch ratio: {frac_mismatch_ratio}")
                    break
                cur_try += 1
                if cur_try >= max_tries:
                    logger.error(f"Could not find a bitflip in {max_tries} tries")
                    break


# def test_random_bitflip_layer():
#     n_repeats = 100
#     layer = RandomBitFlip(p=0.5, seed=0, zero_out_invalid=True)
#     layer.to(DEVICE)

#     x = torch.randn(1024, 1024, device=DEVICE, dtype=torch.float32)
#     for i in range(n_repeats):
#         out = layer(x)
#         assert torch.all(torch.isfinite(x))
#         assert torch.all(torch.isfinite(out))
#         logger.info(f"x.abs().mean(): {torch.mean(torch.abs(x))}")
#         logger.info(f"out.abs().mean(): {torch.mean(torch.abs(out))}")
#         logger.info(f"Error: {torch.mean(torch.abs(x - out))}")
#     logger.info(f"Layer: {layer}")


if __name__ == "__main__":
    set_logging_verbosity("info")
    torch.set_printoptions(linewidth=120)
    # test_random_bitflip_forward_simple()
    test_random_bitflip_forward_bitstring()
    # test_random_bitflip_layer()
