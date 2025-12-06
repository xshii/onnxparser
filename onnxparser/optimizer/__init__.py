# -*- coding: utf-8 -*-
"""Optimizer - Graph optimization passes"""

from onnxparser.optimizer.pass_manager import PassManager, PassLevel, PassResult, optimize
from onnxparser.optimizer.passes import (
    eliminate_dead_code,
    constant_folding,
    fuse_linear_relu,
    fuse_bn_into_conv,
    fuse_consecutive_transpose,
    fuse_attention,
    remove_identity,
    simplify_reshape,
    remove_redundant_cast,
)

__all__ = [
    # Pass Manager
    "PassManager",
    "PassLevel",
    "PassResult",
    "optimize",
    # Individual passes
    "eliminate_dead_code",
    "constant_folding",
    "fuse_linear_relu",
    "fuse_bn_into_conv",
    "fuse_consecutive_transpose",
    "fuse_attention",
    "remove_identity",
    "simplify_reshape",
    "remove_redundant_cast",
]
