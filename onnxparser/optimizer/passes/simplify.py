# -*- coding: utf-8 -*-
"""Graph Simplification Passes"""

import torch
import torch.fx as fx


def remove_identity(gm: fx.GraphModule) -> fx.GraphModule:
    """Remove identity operations that don't change the tensor

    Examples:
    - x + 0 = x
    - x * 1 = x
    - x @ I = x (identity matrix)
    - reshape to same shape
    """

    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue

        replacement = None

        # x + 0 = x
        if node.target == torch.add:
            if len(node.args) >= 2:
                if _is_zero(node.args[1], gm):
                    replacement = node.args[0]
                elif _is_zero(node.args[0], gm):
                    replacement = node.args[1]

        # x * 1 = x
        elif node.target == torch.mul:
            if len(node.args) >= 2:
                if _is_one(node.args[1], gm):
                    replacement = node.args[0]
                elif _is_one(node.args[0], gm):
                    replacement = node.args[1]

        # Replace if found identity
        if replacement is not None and isinstance(replacement, fx.Node):
            node.replace_all_uses_with(replacement)
            gm.graph.erase_node(node)

    gm.graph.lint()
    gm.recompile()
    return gm


def simplify_reshape(gm: fx.GraphModule) -> fx.GraphModule:
    """Simplify reshape operations

    - Remove reshape to same shape
    - Combine consecutive reshapes
    """

    for node in list(gm.graph.nodes):
        # Check for reshape operations
        is_reshape = (
            (node.op == "call_function" and node.target == torch.reshape) or
            (node.op == "call_method" and node.target == "view")
        )
        if not is_reshape:
            continue

        input_node = node.args[0]
        if not isinstance(input_node, fx.Node):
            continue

        # Check for consecutive reshapes
        if input_node.op == "call_function":
            if input_node.target in (torch.reshape, torch.view):
                if len(input_node.users) == 1:
                    # Skip the intermediate reshape
                    original_input = input_node.args[0]
                    new_args = (original_input,) + node.args[1:]
                    node.args = new_args
                    gm.graph.erase_node(input_node)

        # Check if reshape is to same shape (using meta info)
        if "tensor_meta" in node.meta and "tensor_meta" in input_node.meta:
            in_shape = input_node.meta["tensor_meta"].shape
            out_shape = node.meta["tensor_meta"].shape
            if in_shape == out_shape:
                node.replace_all_uses_with(input_node)
                gm.graph.erase_node(node)

    gm.graph.lint()
    gm.recompile()
    return gm


def remove_redundant_cast(gm: fx.GraphModule) -> fx.GraphModule:
    """Remove redundant type cast operations

    - cast(cast(x, dtype), dtype) -> cast(x, dtype)
    - cast(x, same_dtype) -> x
    """

    for node in list(gm.graph.nodes):
        if node.op != "call_method":
            continue
        if node.target != "to":
            continue

        input_node = node.args[0]
        if not isinstance(input_node, fx.Node):
            continue

        # Get target dtype
        target_dtype = None
        if len(node.args) > 1:
            target_dtype = node.args[1]
        elif "dtype" in node.kwargs:
            target_dtype = node.kwargs["dtype"]

        if target_dtype is None:
            continue

        # Check for consecutive casts
        if input_node.op == "call_method" and input_node.target == "to":
            if len(input_node.users) == 1:
                # Skip the intermediate cast
                original_input = input_node.args[0]
                node.args = (original_input,) + node.args[1:]
                gm.graph.erase_node(input_node)

        # Check if cast to same dtype (using meta info)
        if "tensor_meta" in input_node.meta:
            in_dtype = input_node.meta["tensor_meta"].dtype
            if in_dtype == target_dtype:
                node.replace_all_uses_with(input_node)
                gm.graph.erase_node(node)

    gm.graph.lint()
    gm.recompile()
    return gm


def _is_zero(arg, gm: fx.GraphModule) -> bool:
    """Check if argument is zero constant"""
    if isinstance(arg, (int, float)) and arg == 0:
        return True
    if isinstance(arg, fx.Node) and arg.op == "get_attr":
        try:
            val = _get_attr(gm, arg.target)
            if isinstance(val, torch.Tensor):
                return torch.all(val == 0).item()
        except (AttributeError, RuntimeError):
            pass
    return False


def _is_one(arg, gm: fx.GraphModule) -> bool:
    """Check if argument is one constant"""
    if isinstance(arg, (int, float)) and arg == 1:
        return True
    if isinstance(arg, fx.Node) and arg.op == "get_attr":
        try:
            val = _get_attr(gm, arg.target)
            if isinstance(val, torch.Tensor):
                return torch.all(val == 1).item()
        except (AttributeError, RuntimeError):
            pass
    return False


def _get_attr(gm: fx.GraphModule, target: str):
    """Get attribute from module by dotted path"""
    atoms = target.split(".")
    attr = gm
    for atom in atoms:
        attr = getattr(attr, atom)
    return attr
