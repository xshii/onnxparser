# -*- coding: utf-8 -*-
"""Hardware trace data loading and comparison utilities."""

from .format import TraceFormat, TensorHeader, DTYPE_MAP
from .loader import TraceLoader, load_trace, merge_traces
from .codegen import generate_trace_code
from .comparator import compare_traces, compare_tensors, print_compare_report
from .dtypes import (
    HWDType,
    DTypeConverter,
    convert_to_fp32,
    bfp_to_fp32,
    fp8_e4m3_to_fp32,
    fp8_e5m2_to_fp32,
    fxp_to_fp32,
    BFPConfig,
)

__all__ = [
    # Format
    "TraceFormat",
    "TensorHeader",
    "DTYPE_MAP",
    # Loader
    "TraceLoader",
    "load_trace",
    "merge_traces",
    # Codegen
    "generate_trace_code",
    # Comparator
    "compare_traces",
    "compare_tensors",
    "print_compare_report",
    # Data types
    "HWDType",
    "DTypeConverter",
    "convert_to_fp32",
    "bfp_to_fp32",
    "fp8_e4m3_to_fp32",
    "fp8_e5m2_to_fp32",
    "fxp_to_fp32",
    "BFPConfig",
]
