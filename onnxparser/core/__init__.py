# -*- coding: utf-8 -*-
"""Core data structures and mappings"""

from onnxparser.core.ops import ONNX_TO_TORCH, TORCH_TO_ONNX, get_torch_op, is_special_op
from onnxparser.core.dtypes import (
    ONNXDataType,
    to_torch_dtype,
    to_onnx_dtype,
    dtype_size,
)

__all__ = [
    "ONNX_TO_TORCH",
    "TORCH_TO_ONNX",
    "get_torch_op",
    "is_special_op",
    "ONNXDataType",
    "to_torch_dtype",
    "to_onnx_dtype",
    "dtype_size",
]
