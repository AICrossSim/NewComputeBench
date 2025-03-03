import numpy as np
import torch


def get_binary_repr(
    x: torch.Tensor, split_every: int = 4, splitter: str = " "
) -> np.ndarray:
    assert isinstance(x, torch.Tensor), "Only support torch.Tensor"
    if x.dtype in [torch.bool]:
        int_type = torch.bool
        num_bits = 1
    elif x.dtype in [torch.uint8, torch.int8]:
        int_type = torch.uint8
        num_bits = 8
    elif x.dtype in [torch.uint16, torch.int16]:
        int_type = torch.uint16
        num_bits = 16
    elif x.dtype in [torch.uint32, torch.int32]:
        int_type = torch.uint32
        num_bits = 32
    elif x.dtype in [torch.uint64, torch.int64]:
        int_type = torch.uint64
        num_bits = 64
    elif x.dtype in [torch.float16, torch.bfloat16]:
        int_type = torch.uint16
        num_bits = 16
    elif x.dtype in [torch.float32]:
        int_type = torch.uint32
        num_bits = 32
    elif x.dtype in [torch.float64]:
        int_type = torch.uint64
        num_bits = 64
    else:
        raise NotImplementedError(f"Unsupported dtype: {x.dtype}")

    x_int = x.view(dtype=int_type).contiguous().cpu().numpy()

    def formatted_bin(i):
        if num_bits == 1:
            return "1" if i else "0"
        if split_every is None:
            return format(i, f"0{num_bits}b")
        else:
            bin_str = format(i, f"0{num_bits}b")
            bin_str = splitter.join(
                [
                    bin_str[i : i + split_every]
                    for i in range(0, len(bin_str), split_every)
                ]
            )
            return bin_str

    bin_repr = np.array(list(map(formatted_bin, x_int.flatten()))).reshape(x_int.shape)
    return bin_repr


def get_binary_repr_bf16(x: torch.Tensor) -> str:
    assert isinstance(x, torch.Tensor), "Only support torch.Tensor"
    assert x.dtype == torch.bfloat16, "Only support torch.bfloat16 dtype"
    x_int = x.view(dtype=torch.uint16).contiguous().cpu().numpy()

    def formatted_bin(i):
        bin_str = format(i, "016b")
        bin_str = "{sign} {exponent} {mantissa}".format(
            sign=bin_str[0],
            exponent=bin_str[1:9],
            mantissa=bin_str[9:],
        )
        return bin_str

    bin_repr = np.array(list(map(formatted_bin, x_int.flatten()))).reshape(x_int.shape)
    return bin_repr


def get_hex_repr(x: torch.Tensor) -> str:
    assert isinstance(x, torch.Tensor), "Only support torch.Tensor"

    if x.dtype in [torch.bool]:
        int_type = torch.bool
        num_bits = 1
    elif x.dtype in [torch.uint8, torch.int8]:
        int_type = torch.uint8
        num_bits = 8
    elif x.dtype in [torch.uint16, torch.int16]:
        int_type = torch.uint16
        num_bits = 16
    elif x.dtype in [torch.uint32, torch.int32]:
        int_type = torch.uint32
        num_bits = 32
    elif x.dtype in [torch.uint64, torch.int64]:
        int_type = torch.uint64
        num_bits = 64
    elif x.dtype in [torch.float16, torch.bfloat16]:
        int_type = torch.uint16
        num_bits = 16
    elif x.dtype in [torch.float32]:
        int_type = torch.uint32
        num_bits = 32
    elif x.dtype in [torch.float64]:
        int_type = torch.uint64
        num_bits = 64
    else:
        raise NotImplementedError(f"Unsupported dtype: {x.dtype}")

    x_int: np.ndarray = x.view(dtype=int_type).contiguous().cpu().numpy()

    def formatted_hex(i):
        return format(i, f"0{num_bits//4}X")

    # hex_repr = np.array(list(map(formatted_hex, x_int.flatten()))).reshape(x_int.shape)

    formatted_hex = np.vectorize(formatted_hex)
    hex_repr = formatted_hex(x_int.flatten()).reshape(x_int.shape)
    return hex_repr
