# -*- coding: utf-8 -*-
"""ONNX to Torch operator mapping"""

import torch
import torch.nn.functional as F
from typing import Dict, Callable

# ONNX op -> Torch function mapping
ONNX_TO_TORCH: Dict[str, Callable] = {
    # Basic math
    "Add": torch.add,
    "Sub": torch.sub,
    "Mul": torch.mul,
    "Div": torch.div,
    "Neg": torch.neg,
    "Abs": torch.abs,
    "Sqrt": torch.sqrt,
    "Exp": torch.exp,
    "Log": torch.log,
    "Pow": torch.pow,
    "Floor": torch.floor,
    "Ceil": torch.ceil,
    "Round": torch.round,
    "Sign": torch.sign,
    "Reciprocal": torch.reciprocal,

    # Matrix ops
    "MatMul": torch.matmul,
    "Gemm": torch.mm,

    # Activations
    "Relu": torch.relu,
    "Sigmoid": torch.sigmoid,
    "Tanh": torch.tanh,
    "LeakyRelu": F.leaky_relu,
    "Elu": F.elu,
    "Selu": F.selu,
    "Softplus": F.softplus,
    "Softsign": F.softsign,
    "Gelu": F.gelu,
    "HardSigmoid": F.hardsigmoid,
    "HardSwish": F.hardswish,
    "Mish": F.mish,

    # Softmax
    "Softmax": F.softmax,
    "LogSoftmax": F.log_softmax,

    # Normalization
    "BatchNormalization": F.batch_norm,
    "LayerNormalization": F.layer_norm,
    "InstanceNormalization": F.instance_norm,
    "GroupNormalization": F.group_norm,

    # Convolution
    "Conv": F.conv2d,
    "ConvTranspose": F.conv_transpose2d,

    # Pooling
    "MaxPool": F.max_pool2d,
    "AveragePool": F.avg_pool2d,
    "GlobalAveragePool": F.adaptive_avg_pool2d,
    "GlobalMaxPool": F.adaptive_max_pool2d,

    # Shape ops
    "Reshape": torch.reshape,
    "Transpose": torch.transpose,
    "Squeeze": torch.squeeze,
    "Unsqueeze": torch.unsqueeze,
    "Flatten": torch.flatten,
    "Concat": torch.cat,
    "Split": torch.split,
    "Slice": None,
    "Gather": torch.gather,
    "Scatter": torch.scatter,

    # Comparison
    "Equal": torch.eq,
    "Less": torch.lt,
    "LessOrEqual": torch.le,
    "Greater": torch.gt,
    "GreaterOrEqual": torch.ge,
    "Not": torch.logical_not,
    "And": torch.logical_and,
    "Or": torch.logical_or,

    # Reduction
    "ReduceSum": torch.sum,
    "ReduceMean": torch.mean,
    "ReduceMax": torch.max,
    "ReduceMin": torch.min,
    "ReduceProd": torch.prod,

    # Other
    "Clip": torch.clamp,
    "Cast": None,
    "Dropout": F.dropout,
    "Identity": lambda x: x,
    "Pad": F.pad,
    "Resize": F.interpolate,

    # Transformer
    "Attention": None,
    "Erf": torch.erf,
}

# Special ops that need custom handling
SPECIAL_OPS = {
    "Conv",
    "MaxPool",
    "AveragePool",
    "Gemm",
    "Softmax",
    "Slice",
    "Cast",
    "Attention",
}


def get_torch_op(onnx_op: str) -> Callable:
    """Get Torch function for ONNX op"""
    if onnx_op not in ONNX_TO_TORCH:
        raise NotImplementedError(f"Unsupported ONNX op: {onnx_op}")

    op = ONNX_TO_TORCH[onnx_op]
    if op is None:
        raise NotImplementedError(f"Op {onnx_op} requires special handling")

    return op


def is_special_op(onnx_op: str) -> bool:
    """Check if op needs special handling"""
    return onnx_op in SPECIAL_OPS


# Reverse mapping for export
TORCH_TO_ONNX: Dict[Callable, str] = {v: k for k, v in ONNX_TO_TORCH.items() if v is not None}
