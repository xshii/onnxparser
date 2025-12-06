# -*- coding: utf-8 -*-
"""Data type definitions and conversions"""

import torch
import numpy as np
from typing import Union, Dict
from enum import IntEnum


class ONNXDataType(IntEnum):
    """ONNX TensorProto data types"""
    UNDEFINED = 0
    FLOAT = 1
    UINT8 = 2
    INT8 = 3
    UINT16 = 4
    INT16 = 5
    INT32 = 6
    INT64 = 7
    STRING = 8
    BOOL = 9
    FLOAT16 = 10
    DOUBLE = 11
    UINT32 = 12
    UINT64 = 13
    COMPLEX64 = 14
    COMPLEX128 = 15
    BFLOAT16 = 16
    FLOAT8E4M3FN = 17
    FLOAT8E4M3FNUZ = 18
    FLOAT8E5M2 = 19
    FLOAT8E5M2FNUZ = 20


# ONNX dtype -> Torch dtype
ONNX_TO_TORCH_DTYPE: Dict[int, torch.dtype] = {
    ONNXDataType.FLOAT: torch.float32,
    ONNXDataType.FLOAT16: torch.float16,
    ONNXDataType.DOUBLE: torch.float64,
    ONNXDataType.BFLOAT16: torch.bfloat16,
    ONNXDataType.INT8: torch.int8,
    ONNXDataType.INT16: torch.int16,
    ONNXDataType.INT32: torch.int32,
    ONNXDataType.INT64: torch.int64,
    ONNXDataType.UINT8: torch.uint8,
    ONNXDataType.BOOL: torch.bool,
    ONNXDataType.COMPLEX64: torch.complex64,
    ONNXDataType.COMPLEX128: torch.complex128,
}

# Torch dtype -> ONNX dtype
TORCH_TO_ONNX_DTYPE: Dict[torch.dtype, int] = {v: k for k, v in ONNX_TO_TORCH_DTYPE.items()}

# ONNX dtype -> NumPy dtype
ONNX_TO_NUMPY_DTYPE: Dict[int, np.dtype] = {
    ONNXDataType.FLOAT: np.float32,
    ONNXDataType.FLOAT16: np.float16,
    ONNXDataType.DOUBLE: np.float64,
    ONNXDataType.INT8: np.int8,
    ONNXDataType.INT16: np.int16,
    ONNXDataType.INT32: np.int32,
    ONNXDataType.INT64: np.int64,
    ONNXDataType.UINT8: np.uint8,
    ONNXDataType.UINT16: np.uint16,
    ONNXDataType.UINT32: np.uint32,
    ONNXDataType.UINT64: np.uint64,
    ONNXDataType.BOOL: np.bool_,
    ONNXDataType.COMPLEX64: np.complex64,
    ONNXDataType.COMPLEX128: np.complex128,
}

# String -> Torch dtype
STR_TO_TORCH_DTYPE: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "half": torch.float16,
    "float64": torch.float64,
    "double": torch.float64,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int": torch.int32,
    "int64": torch.int64,
    "long": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}


def to_torch_dtype(dtype: Union[int, str, torch.dtype, np.dtype]) -> torch.dtype:
    """Convert to Torch dtype"""
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, int):
        return ONNX_TO_TORCH_DTYPE.get(dtype, torch.float32)
    if isinstance(dtype, str):
        return STR_TO_TORCH_DTYPE.get(dtype.lower(), torch.float32)
    if isinstance(dtype, np.dtype):
        np_to_torch = {
            np.float32: torch.float32,
            np.float16: torch.float16,
            np.float64: torch.float64,
            np.int8: torch.int8,
            np.int16: torch.int16,
            np.int32: torch.int32,
            np.int64: torch.int64,
            np.uint8: torch.uint8,
            np.bool_: torch.bool,
        }
        return np_to_torch.get(dtype.type, torch.float32)
    return torch.float32


def to_onnx_dtype(dtype: Union[torch.dtype, str]) -> int:
    """Convert to ONNX dtype"""
    if isinstance(dtype, str):
        dtype = to_torch_dtype(dtype)
    return TORCH_TO_ONNX_DTYPE.get(dtype, ONNXDataType.FLOAT)


def dtype_size(dtype: Union[int, torch.dtype]) -> int:
    """Get byte size of data type"""
    if isinstance(dtype, int):
        dtype = ONNX_TO_TORCH_DTYPE.get(dtype, torch.float32)

    size_map = {
        torch.float32: 4,
        torch.float16: 2,
        torch.float64: 8,
        torch.bfloat16: 2,
        torch.int8: 1,
        torch.int16: 2,
        torch.int32: 4,
        torch.int64: 8,
        torch.uint8: 1,
        torch.bool: 1,
        torch.complex64: 8,
        torch.complex128: 16,
    }
    return size_map.get(dtype, 4)
