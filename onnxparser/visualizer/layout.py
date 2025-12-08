# -*- coding: utf-8 -*-
"""Graph layout computation for visualization"""

from typing import Dict, List


class GraphLayoutEngine:
    """Backend graph layout computation using layered layout algorithm"""

    def __init__(self, node_width: int = 120, node_height: int = 38,
                 layer_gap: int = 160, node_gap: int = 14):
        self.node_width = node_width
        self.node_height = node_height
        self.layer_gap = layer_gap
        self.node_gap = node_gap
        self.weight_width = 80
        self.weight_height = 28

    def compute_layout(self, nodes: List[Dict], edges: List[Dict],
                       canvas_width: int = 1200, canvas_height: int = 800) -> List[Dict]:
        """Compute node positions using layered layout algorithm"""
        node_by_id = {n["id"]: n for n in nodes}

        # Build adjacency
        incoming = {n["id"]: [] for n in nodes}
        outgoing = {n["id"]: [] for n in nodes}
        for e in edges:
            incoming[e["target"]].append(e["source"])
            outgoing[e["source"]].append(e["target"])

        # Separate weight nodes
        weight_nodes = [n for n in nodes if n["type"] == "weight"]
        compute_nodes = [n for n in nodes if n["type"] != "weight"]

        # Assign layers to compute nodes
        layers = {}

        def get_layer(node_id: str) -> int:
            if node_id in layers:
                return layers[node_id]
            node = node_by_id[node_id]
            if node["type"] == "weight":
                return -1
            deps = [src for src in incoming[node_id] if node_by_id[src]["type"] != "weight"]
            if not deps:
                layers[node_id] = 0
                return 0
            layers[node_id] = max(get_layer(d) for d in deps) + 1
            return layers[node_id]

        for n in compute_nodes:
            get_layer(n["id"])

        max_layer = max(layers.values()) if layers else 0

        # Group by layer
        layer_nodes: Dict[int, List[Dict]] = {}
        for n in compute_nodes:
            layer = layers[n["id"]]
            if layer not in layer_nodes:
                layer_nodes[layer] = []
            layer_nodes[layer].append(n)

        # Position compute nodes
        total_width = (max_layer + 1) * (self.node_width + self.layer_gap)
        start_x = max(60, (canvas_width - total_width) // 2)

        for layer, l_nodes in layer_nodes.items():
            total_height = len(l_nodes) * (self.node_height + self.node_gap) - self.node_gap
            start_y = (canvas_height - total_height) // 2

            for idx, n in enumerate(l_nodes):
                n["x"] = start_x + layer * (self.node_width + self.layer_gap)
                n["y"] = start_y + idx * (self.node_height + self.node_gap)
                n["width"] = self.node_width
                n["height"] = self.node_height

        # Position weight nodes
        for n in weight_nodes:
            consumers = [node_by_id[cid] for cid in outgoing[n["id"]]
                         if "x" in node_by_id[cid]]
            if consumers:
                consumer = consumers[0]
                sibling_weights = [node_by_id[sid] for sid in incoming[consumer["id"]]
                                   if node_by_id[sid]["type"] == "weight"]
                sib_idx = sibling_weights.index(n) if n in sibling_weights else 0
                n["x"] = consumer["x"] - self.weight_width - 25
                n["y"] = consumer["y"] - 15 + sib_idx * (self.weight_height + 6)
            else:
                n["x"] = 20
                n["y"] = 50 + weight_nodes.index(n) * (self.weight_height + 6)

            n["width"] = self.weight_width
            n["height"] = self.weight_height
            n["is_weight"] = True

        return nodes
