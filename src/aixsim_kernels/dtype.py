import torch
import triton.language as tl
from enum import IntEnum


class DType(IntEnum):
    FP32 = 0
    FP16 = 1
    BF16 = 2
    FP8 = 3
    INT8 = 4
    UINT8 = 5
    INT32 = 6
    UINT32 = 7
    FP8e5 = 8


ENUM_DTYPE_TO_TORCH = {
    DType.FP32: torch.float32,
    DType.FP16: torch.float16,
    DType.BF16: torch.bfloat16,
    DType.FP8: torch.float8_e4m3fn,
    DType.INT8: torch.int8,
    DType.UINT8: torch.uint8,
    DType.INT32: torch.int32,
    DType.UINT32: torch.uint32,
    DType.FP8e5: torch.float8_e5m2,
}

TORCH_DTYPE_TO_ENUM = {v: k for k, v in ENUM_DTYPE_TO_TORCH.items()}

TORCH_DTYPE_TO_TRITON = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
    torch.int8: tl.int8,
    torch.uint8: tl.uint8,
    torch.int16: tl.int16,
    torch.uint16: tl.uint16,
    torch.int32: tl.int32,
    torch.uint32: tl.uint32,
    torch.float8_e4m3fn: tl.float8e4nv,
    torch.float8_e5m2: tl.float8e5,
}

ENUM_DTYPE_TO_TRITON = {
    k: TORCH_DTYPE_TO_TRITON[d] for k, d in ENUM_DTYPE_TO_TORCH.items()
}
