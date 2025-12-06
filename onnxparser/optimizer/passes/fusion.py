# -*- coding: utf-8 -*-
"""Operator Fusion Passes"""

import torch
import torch.nn.functional as F
import torch.fx as fx
from typing import List, Optional, Callable


def fuse_linear_relu(gm: fx.GraphModule) -> fx.GraphModule:
    """Fuse Linear + ReLU into single operation"""

    for node in list(gm.graph.nodes):
        # Look for relu nodes
        if node.op != "call_function":
            continue
        if node.target not in (torch.relu, F.relu):
            continue

        # Check if input is linear/matmul + add
        input_node = node.args[0]
        if not isinstance(input_node, fx.Node):
            continue

        if input_node.op == "call_function" and input_node.target == torch.add:
            # Pattern: add -> relu, check if add's input is matmul
            add_node = input_node
            if len(add_node.users) != 1:  # Only used by relu
                continue

            matmul_node = add_node.args[0]
            if isinstance(matmul_node, fx.Node) and matmul_node.op == "call_function":
                if matmul_node.target == torch.matmul and len(matmul_node.users) == 1:
                    # Found pattern: matmul -> add -> relu
                    # Mark for fused execution
                    node.meta["fused_pattern"] = "linear_relu"
                    node.meta["fused_nodes"] = [matmul_node.name, add_node.name, node.name]

    gm.graph.lint()
    gm.recompile()
    return gm


def fuse_bn_into_conv(gm: fx.GraphModule) -> fx.GraphModule:
    """Fuse BatchNorm into Conv (for inference only)

    Conv: y = W * x + b
    BN:  z = gamma * (y - mean) / sqrt(var + eps) + beta

    Fused: z = W' * x + b'
    where W' = gamma * W / sqrt(var + eps)
          b' = gamma * (b - mean) / sqrt(var + eps) + beta
    """

    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue

        # Look for batch_norm
        if node.target != F.batch_norm:
            continue

        input_node = node.args[0]
        if not isinstance(input_node, fx.Node):
            continue

        # Check if input is conv2d
        if input_node.op != "call_function":
            continue
        if input_node.target not in (F.conv2d, F.conv1d):
            continue
        if len(input_node.users) != 1:  # Conv only used by BN
            continue

        # Mark as fused pattern (actual weight fusion done at export/compile time)
        node.meta["fused_pattern"] = "conv_bn"
        node.meta["conv_node"] = input_node.name

    gm.graph.lint()
    gm.recompile()
    return gm


def fuse_consecutive_transpose(gm: fx.GraphModule) -> fx.GraphModule:
    """Remove or simplify consecutive transpose operations

    transpose(transpose(x, a, b), a, b) = x
    """

    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        if node.target not in (torch.transpose, torch.permute):
            continue

        input_node = node.args[0]
        if not isinstance(input_node, fx.Node):
            continue

        if input_node.op != "call_function":
            continue
        if input_node.target != node.target:
            continue

        # Check if dimensions match (inverse transpose)
        if node.target == torch.transpose:
            if len(node.args) >= 3 and len(input_node.args) >= 3:
                dim0, dim1 = node.args[1], node.args[2]
                in_dim0, in_dim1 = input_node.args[1], input_node.args[2]

                if (dim0, dim1) == (in_dim0, in_dim1) or (dim0, dim1) == (in_dim1, in_dim0):
                    # Consecutive transpose with same dims = identity
                    if len(input_node.users) == 1:
                        original_input = input_node.args[0]
                        node.replace_all_uses_with(original_input)
                        gm.graph.erase_node(node)
                        gm.graph.erase_node(input_node)

    gm.graph.lint()
    gm.recompile()
    return gm


def fuse_attention(gm: fx.GraphModule) -> fx.GraphModule:
    """Detect and mark Multi-Head Attention pattern for optimized execution

    Pattern: Q @ K.T -> softmax -> @ V
    """

    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue

        # Look for final matmul (attn @ V)
        if node.target != torch.matmul:
            continue

        attn_node = node.args[0]
        if not isinstance(attn_node, fx.Node):
            continue

        # Check if attn comes from softmax
        if attn_node.op != "call_function":
            continue
        if attn_node.target not in (F.softmax, torch.softmax):
            continue

        qk_node = attn_node.args[0]
        if not isinstance(qk_node, fx.Node):
            continue

        # Check if qk comes from matmul (Q @ K.T)
        if qk_node.op != "call_function":
            continue
        if qk_node.target != torch.matmul:
            continue

        # Found attention pattern
        node.meta["fused_pattern"] = "attention"
        node.meta["attention_nodes"] = {
            "qk_matmul": qk_node.name,
            "softmax": attn_node.name,
            "av_matmul": node.name,
        }

    gm.graph.lint()
    gm.recompile()
    return gm
