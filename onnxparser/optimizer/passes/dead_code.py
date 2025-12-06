# -*- coding: utf-8 -*-
"""Dead Code Elimination Pass"""

import torch.fx as fx


def eliminate_dead_code(gm: fx.GraphModule) -> fx.GraphModule:
    """Remove nodes that have no users (dead code elimination)"""
    changed = True

    while changed:
        changed = False
        nodes_to_remove = []

        for node in reversed(list(gm.graph.nodes)):
            # Skip placeholders and outputs
            if node.op in ("placeholder", "output"):
                continue

            # If node has no users, mark for removal
            if len(node.users) == 0:
                nodes_to_remove.append(node)
                changed = True

        for node in nodes_to_remove:
            gm.graph.erase_node(node)

    gm.graph.lint()
    gm.recompile()
    return gm
