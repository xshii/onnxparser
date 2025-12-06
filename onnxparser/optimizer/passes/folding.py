# -*- coding: utf-8 -*-
"""Constant Folding Pass"""

import torch
import torch.fx as fx
from typing import Dict, Any, Optional


def constant_folding(gm: fx.GraphModule) -> fx.GraphModule:
    """Fold constant expressions at compile time"""

    # Find all get_attr nodes (constants)
    constants: Dict[str, torch.Tensor] = {}
    for node in gm.graph.nodes:
        if node.op == "get_attr":
            try:
                val = _get_attr(gm, node.target)
                if isinstance(val, torch.Tensor):
                    constants[node.name] = val
            except AttributeError:
                pass

    changed = True
    while changed:
        changed = False

        for node in list(gm.graph.nodes):
            if node.op != "call_function":
                continue

            # Check if all inputs are constants
            all_const = True
            args = []
            for arg in node.args:
                if isinstance(arg, fx.Node):
                    if arg.name in constants:
                        args.append(constants[arg.name])
                    else:
                        all_const = False
                        break
                else:
                    args.append(arg)

            if not all_const:
                continue

            # Try to evaluate
            try:
                with torch.no_grad():
                    result = node.target(*args, **node.kwargs)

                if isinstance(result, torch.Tensor):
                    # Create new buffer for the result
                    buf_name = f"_folded_{node.name}"
                    gm.register_buffer(buf_name, result)

                    # Replace with get_attr
                    with gm.graph.inserting_before(node):
                        new_node = gm.graph.get_attr(buf_name)
                        node.replace_all_uses_with(new_node)

                    constants[new_node.name] = result
                    gm.graph.erase_node(node)
                    changed = True

            except Exception:
                # Cannot fold this node
                pass

    gm.graph.lint()
    gm.recompile()
    return gm


def _get_attr(gm: fx.GraphModule, target: str) -> Any:
    """Get attribute from module by dotted path"""
    atoms = target.split(".")
    attr = gm
    for atom in atoms:
        attr = getattr(attr, atom)
    return attr
