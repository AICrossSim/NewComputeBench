import logging
from difflib import SequenceMatcher

import torch

from aixsim_kernels.random_bitflip import (
    find_nearest_prob_n_halves,
    random_bitflip_fn,
    RandomBitFlip,
)
from aixsim_kernels.utils.bit_repr import get_binary_repr
from aixsim_kernels.logging import set_logging_verbosity

logger = logging.getLogger(f"aixsim_kernels.test.{__name__}")

DEVICE = "cuda"


def _count_matched_bits(a: str, b: str):
    assert len(a) == len(b)
    return sum([1 for i in range(len(a)) if a[i] == b[i]])


@torch.no_grad()
def test_random_bitflip_forward_simple():
    x = torch.zeros(4, device=DEVICE, dtype=torch.bfloat16)
    exp_halves = 4
    frac_halves = 1
    seed_exp, seed_frac = 0, 0
    out, seed_exp, seed_frac = random_bitflip_fn(
        x,
        exp_halves=exp_halves,
        frac_halves=frac_halves,
        seed_exp=seed_exp,
        seed_frac=seed_frac,
        zero_out_threshold=None,
        train=True,
    )
    logger.info(f"binary x:\n{get_binary_repr(x, splitter='')}")
    logger.info(f"binary out:\n{get_binary_repr(out, splitter='')}")
    logger.info(f"seed_exp: {seed_exp}, seed_frac: {seed_frac}")


@torch.no_grad()
def test_random_bitflip_forward_fully_activated():
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
            exp_halves = find_nearest_prob_n_halves(exp_p)
            frac_halves = find_nearest_prob_n_halves(frac_p)
            seed_exp, seed_frac = 0, 0
            logger.info(
                f"====== input_dtype = {input_dtype}, exp_p = {exp_p}, frac_p = {frac_p}, exp_halves = {exp_halves}, frac_halves = {frac_halves}, train = True ====="
            )
            while True:
                out, seed_exp, seed_frac = random_bitflip_fn(
                    x,
                    exp_halves=exp_halves,
                    frac_halves=frac_halves,
                    seed_exp=seed_exp,
                    seed_frac=seed_frac,
                    zero_out_threshold=None,
                    train=True,
                )
                assert out.dtype == input_dtype
                assert out.shape == x.shape
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


@torch.no_grad()
def test_random_bitflip_forward_zero_outed():
    for exp_halves in [1, 2, 3, 4, 5, 6]:
        x = torch.randn(2048, 2048, device=DEVICE, dtype=torch.float32)
        frac_halves = 2
        seed_exp, seed_frac = 0, 0
        zero_out_threshold = 200.0
        out, seed_exp, seed_frac = random_bitflip_fn(
            x,
            exp_halves=exp_halves,
            frac_halves=frac_halves,
            seed_exp=seed_exp,
            seed_frac=seed_frac,
            zero_out_threshold=zero_out_threshold,
            train=True,
        )
        assert torch.all(torch.isfinite(x))
        zero_out_ratio = (out == 0.0).sum() / out.numel()
        logger.info(
            f"===== exp_halves = {exp_halves}, frac_halves = {frac_halves}, exp_p = {0.5**exp_halves}, frac_p = {0.5**frac_halves} ====="
        )
        logger.info(f"zero_out_ratio: {zero_out_ratio}")


def test_random_bitflip_fn_backward():
    n_repeats = 10
    for exp_halves in [1, 2, 3]:
        for _ in range(n_repeats):
            x = torch.rand(8, device=DEVICE, dtype=torch.float32)
            x = x + 0.1
            x.requires_grad_()
            frac_halves = 2
            seed_exp, seed_frac = 0, 0
            zero_out_threshold = 200.0
            out, seed_exp, seed_frac = random_bitflip_fn(
                x,
                exp_halves=exp_halves,
                frac_halves=frac_halves,
                seed_exp=seed_exp,
                seed_frac=seed_frac,
                zero_out_threshold=zero_out_threshold,
                train=True,
            )

            loss = torch.sum(out)
            loss.backward()
            assert torch.all(torch.isfinite(x))
            assert torch.all((out != 0) == (x.grad == 1.0))
        logger.info(f"exp_halves = {exp_halves} passed")


def test_random_bitflip_layer():
    n_passes = 4
    p_exp = 0.5**4
    p_frac = 0.5**5
    zero_out_threshold = 2
    seed_exp, seed_frac = 1, 1
    bitflip = RandomBitFlip(
        p_exp=p_exp,
        p_frac=p_frac,
        zero_out_threshold=zero_out_threshold,
        seed_exp=seed_exp,
        seed_frac=seed_frac,
    )
    logger.info(bitflip)

    for i in range(n_passes):
        x = torch.randn(8, device=DEVICE, dtype=torch.float32)
        x = x + 0.1
        x.requires_grad_()

        out = bitflip(x)
        loss = torch.sum(out)
        loss.backward()

        assert torch.all(torch.isfinite(x))
        assert torch.all((out != 0) == (x.grad == 1.0))
        logger.info(f"{i}-th pass, {bitflip}")


if __name__ == "__main__":
    set_logging_verbosity("info")
    torch.set_printoptions(linewidth=120)
    # test_random_bitflip_forward_simple()
    # test_random_bitflip_forward_fully_activated()
    # test_random_bitflip_forward_zero_outed()
    # test_random_bitflip_fn_backward()
    test_random_bitflip_layer()
