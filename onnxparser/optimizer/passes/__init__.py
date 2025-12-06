# -*- coding: utf-8 -*-
"""Optimizer Passes - Categorized graph optimization passes"""

from onnxparser.optimizer.passes.dead_code import eliminate_dead_code
from onnxparser.optimizer.passes.folding import constant_folding
from onnxparser.optimizer.passes.fusion import (
    fuse_linear_relu,
    fuse_bn_into_conv,
    fuse_consecutive_transpose,
    fuse_attention,
)
from onnxparser.optimizer.passes.simplify import (
    remove_identity,
    simplify_reshape,
    remove_redundant_cast,
)

__all__ = [
    # Dead code
    "eliminate_dead_code",
    # Folding
    "constant_folding",
    # Fusion
    "fuse_linear_relu",
    "fuse_bn_into_conv",
    "fuse_consecutive_transpose",
    "fuse_attention",
    # Simplify
    "remove_identity",
    "simplify_reshape",
    "remove_redundant_cast",
]
