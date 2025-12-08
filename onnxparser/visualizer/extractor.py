# -*- coding: utf-8 -*-
"""Graph data extraction and tensor tracing"""

import math
from typing import Dict, Optional, Any, List

import numpy as np
import torch
import torch.fx as fx


def analyze_tensor(value: torch.Tensor, op: str, target: str = "", node_name: str = "") -> List[Dict]:
    """Analyze tensor and return diagnostic insights"""
    insights = []

    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return insights

    target_lower = target.lower()
    name_lower = node_name.lower()

    # Skip inf/nan warning for known mask tensors (they intentionally contain -inf)
    # Also skip nodes that are in the attention score path before softmax
    is_mask_related = any(kw in target_lower or kw in name_lower for kw in
                         ["mask", "causal", "attention_mask", "attn_mask"])

    finite_mask = torch.isfinite(value)
    if not finite_mask.all() and not is_mask_related:
        non_finite = value[~finite_mask]
        num_nan = torch.isnan(value).sum().item()
        num_pos_inf = (non_finite == float('inf')).sum().item()
        num_neg_inf = (non_finite == float('-inf')).sum().item()

        # Negative inf is typically from attention masks - not a problem
        # Only warn for NaN or positive inf (signs of numerical issues)
        if num_nan > 0 or num_pos_inf > 0:
            parts = []
            if num_nan > 0:
                parts.append(f"{num_nan} NaN")
            if num_pos_inf > 0:
                parts.append(f"{num_pos_inf} +Inf")
            insights.append({
                "type": "warning" if num_pos_inf > 0 else "error",
                "title": "Numerical Issues",
                "detail": f"Contains {', '.join(parts)}. May indicate overflow or invalid operations"
            })

    if not finite_mask.any():
        return insights

    finite_values = value[finite_mask]
    mean_val = finite_values.mean().item()
    std_val = finite_values.std().item() if finite_values.numel() > 1 else 0
    max_val = finite_values.max().item()
    min_val = finite_values.min().item()

    # Check for vanishing values
    if abs(mean_val) < 1e-6 and std_val < 1e-6:
        insights.append({
            "type": "warning",
            "title": "Vanishing Values",
            "detail": f"Mean≈0, Std≈0. May indicate vanishing gradients or dead neurons"
        })

    # Check for exploding values
    if abs(max_val) > 1e6 or abs(min_val) > 1e6:
        insights.append({
            "type": "error",
            "title": "Exploding Values",
            "detail": f"Values exceed 1e6 (max={max_val:.2e}). May indicate exploding gradients"
        })

    # Activation-specific checks
    target_lower = target.lower()

    # ReLU analysis
    if "relu" in target_lower or "relu" in op.lower():
        zero_pct = (finite_values == 0).sum().item() / finite_values.numel() * 100
        if zero_pct > 90:
            insights.append({
                "type": "warning",
                "title": "Dead ReLU",
                "detail": f"{zero_pct:.1f}% outputs are zero. Neurons may be dead"
            })
        elif zero_pct > 50:
            insights.append({
                "type": "info",
                "title": "ReLU Sparsity",
                "detail": f"{zero_pct:.1f}% outputs are zero (normal sparsity)"
            })

    # Softmax analysis
    if "softmax" in target_lower:
        if value.dim() >= 1:
            # Check if distribution is too peaked
            max_prob = finite_values.max().item()
            if max_prob > 0.99:
                insights.append({
                    "type": "info",
                    "title": "Sharp Attention",
                    "detail": f"Max probability {max_prob:.3f}. Attention is very focused"
                })
            # Check if distribution is too uniform
            if value.dim() >= 2:
                last_dim = value.shape[-1]
                if abs(max_prob - 1.0/last_dim) < 0.1:
                    insights.append({
                        "type": "warning",
                        "title": "Uniform Attention",
                        "detail": "Attention weights nearly uniform. May not be learning meaningful patterns"
                    })

    # LayerNorm output check
    if "layer_norm" in target_lower or "ln" in target_lower:
        if abs(mean_val) > 0.1:
            insights.append({
                "type": "warning",
                "title": "LayerNorm Shift",
                "detail": f"Mean={mean_val:.3f}, expected ~0. Check normalization"
            })
        if abs(std_val - 1.0) > 0.3:
            insights.append({
                "type": "warning",
                "title": "LayerNorm Scale",
                "detail": f"Std={std_val:.3f}, expected ~1. Check normalization"
            })
        if abs(mean_val) < 0.1 and abs(std_val - 1.0) < 0.2:
            insights.append({
                "type": "success",
                "title": "Healthy Normalization",
                "detail": f"Mean≈0 ({mean_val:.3f}), Std≈1 ({std_val:.3f})"
            })

    # General distribution health
    if not insights and finite_values.numel() > 10:
        skewness = ((finite_values - mean_val) ** 3).mean().item() / (std_val ** 3 + 1e-8)
        if abs(skewness) > 2:
            insights.append({
                "type": "info",
                "title": "Skewed Distribution",
                "detail": f"Skewness={skewness:.2f}. Distribution is asymmetric"
            })

    return insights


def safe_float(v) -> Optional[float]:
    """Convert to JSON-safe float (handle inf/nan)"""
    f = float(v)
    if math.isnan(f):
        return None
    if math.isinf(f):
        return 1e38 if f > 0 else -1e38
    return f


def downsample_2d(tensor: torch.Tensor, max_size: int = 64) -> torch.Tensor:
    """Downsample 2D tensor to max size"""
    h, w = tensor.shape
    step_h = max(1, h // max_size)
    step_w = max(1, w // max_size)
    return tensor[::step_h, ::step_w]


def clean_tensor(t: torch.Tensor) -> torch.Tensor:
    """Replace inf/nan with 0"""
    return torch.where(torch.isfinite(t), t, torch.zeros_like(t))


class GraphExtractor:
    """Extract graph structure from FX GraphModule with tensor data"""

    def __init__(self, gm: fx.GraphModule, input_data: Optional[Dict[str, torch.Tensor]] = None):
        self.gm = gm
        self.input_data = input_data or {}
        self.execution_trace: List[Dict] = []

        if input_data:
            self._trace_execution()

    def _trace_execution(self):
        """Trace execution to capture intermediate tensor values"""
        env = {}

        for node in self.gm.graph.nodes:
            self._execute_node(node, env)
            value = env.get(node.name)
            target = str(node.target) if node.target else ""
            # Pass node name for better mask detection
            trace_info = self._extract_tensor_info(node.name, node.op, value, target, node.name)
            self.execution_trace.append(trace_info)

    def _execute_node(self, node: fx.Node, env: Dict):
        """Execute a single node and store result in env"""
        if node.op == "placeholder":
            if node.name in self.input_data:
                env[node.name] = self.input_data[node.name]
            else:
                # Try to match by partial name
                for k, v in self.input_data.items():
                    if k in node.name or node.name in k:
                        env[node.name] = v
                        break

        elif node.op == "get_attr":
            attrs = node.target.split(".")
            obj = self.gm
            for attr in attrs:
                obj = getattr(obj, attr)
            env[node.name] = obj

        elif node.op == "call_function":
            args = [env.get(a.name) if isinstance(a, fx.Node) else a for a in node.args]
            kwargs = {k: env.get(v.name) if isinstance(v, fx.Node) else v
                      for k, v in node.kwargs.items()}
            try:
                env[node.name] = node.target(*args, **kwargs)
            except Exception:
                env[node.name] = None

        elif node.op == "call_method":
            obj = env.get(node.args[0].name) if node.args else None
            if obj is not None:
                args = [env.get(a.name) if isinstance(a, fx.Node) else a
                        for a in node.args[1:]]
                kwargs = {k: env.get(v.name) if isinstance(v, fx.Node) else v
                          for k, v in node.kwargs.items()}
                try:
                    env[node.name] = getattr(obj, node.target)(*args, **kwargs)
                except Exception:
                    env[node.name] = None

        elif node.op == "call_module":
            submod = self.gm.get_submodule(node.target)
            args = [env.get(a.name) if isinstance(a, fx.Node) else a for a in node.args]
            kwargs = {k: env.get(v.name) if isinstance(v, fx.Node) else v
                      for k, v in node.kwargs.items()}
            try:
                env[node.name] = submod(*args, **kwargs)
            except Exception:
                env[node.name] = None

        elif node.op == "output":
            if node.args:
                arg = node.args[0]
                if isinstance(arg, fx.Node):
                    env[node.name] = env.get(arg.name)
                elif isinstance(arg, (tuple, list)):
                    env[node.name] = tuple(
                        env.get(a.name) if isinstance(a, fx.Node) else a
                        for a in arg
                    )

    def _extract_tensor_info(self, name: str, op: str, value: Any, target: str = "", node_name: str = "") -> Dict:
        """Extract statistics and visualization data from a tensor"""
        trace_info = {"name": name, "op": op}

        if not isinstance(value, torch.Tensor):
            return trace_info

        # Analyze tensor and add insights
        insights = analyze_tensor(value, op, target, node_name)
        if insights:
            trace_info["insights"] = insights

        # Basic info
        trace_info["shape"] = list(value.shape)
        trace_info["dtype"] = str(value.dtype)

        # Statistics (from finite values only)
        finite_mask = torch.isfinite(value)
        if finite_mask.any():
            finite_values = value[finite_mask]
            trace_info["min"] = safe_float(finite_values.min().item())
            trace_info["max"] = safe_float(finite_values.max().item())
            trace_info["mean"] = safe_float(finite_values.mean().item())
            trace_info["std"] = safe_float(finite_values.std().item()) if finite_values.numel() > 1 else 0.0
        else:
            trace_info["min"] = 0.0
            trace_info["max"] = 0.0
            trace_info["mean"] = 0.0
            trace_info["std"] = 0.0

        # Raw values preview
        trace_info["values"] = self._extract_values_preview(value)

        # Histogram (finite values only)
        trace_info["histogram"] = self._extract_histogram(value)

        # Heatmap with slices
        if value.dim() >= 2:
            trace_info["heatmap"] = self._extract_heatmap(value)

        return trace_info

    def _extract_values_preview(self, value: torch.Tensor) -> Optional[Dict]:
        """Extract a preview of raw values with slices for high-dim tensors"""
        try:
            result = {"slices": [], "dims": [], "shape": list(value.shape)}

            if value.dim() == 0:
                # Scalar
                v = value.item()
                result["slices"].append([[v if np.isfinite(v) else 0]])

            elif value.dim() == 1:
                # 1D: show first 20 values
                preview = value[:min(20, value.shape[0])]
                preview_np = preview.detach().cpu().numpy()
                result["slices"].append(np.where(np.isfinite(preview_np), preview_np, 0).tolist())

            elif value.dim() == 2:
                # 2D: show first 8x8
                preview = value[:min(8, value.shape[0]), :min(8, value.shape[1])]
                preview_np = preview.detach().cpu().numpy()
                result["slices"].append(np.where(np.isfinite(preview_np), preview_np, 0).tolist())

            elif value.dim() == 3:
                # 3D: allow browsing first dimension
                num_slices = min(value.shape[0], 16)
                result["dims"] = [{"name": "dim0", "size": value.shape[0], "shown": num_slices}]
                for i in range(num_slices):
                    v = value[i, :min(8, value.shape[1]), :min(8, value.shape[2])]
                    preview_np = v.detach().cpu().numpy()
                    result["slices"].append(np.where(np.isfinite(preview_np), preview_np, 0).tolist())

            elif value.dim() == 4:
                # 4D: allow browsing first two dimensions
                dim0_size = min(value.shape[0], 4)
                dim1_size = min(value.shape[1], 4)
                result["dims"] = [
                    {"name": "batch", "size": value.shape[0], "shown": dim0_size},
                    {"name": "channel", "size": value.shape[1], "shown": dim1_size}
                ]
                for i in range(dim0_size):
                    for j in range(dim1_size):
                        v = value[i, j, :min(8, value.shape[2]), :min(8, value.shape[3])]
                        preview_np = v.detach().cpu().numpy()
                        result["slices"].append(np.where(np.isfinite(preview_np), preview_np, 0).tolist())

            else:
                # Higher dims: flatten leading dims to get 2D slice
                v = value
                while v.dim() > 2:
                    v = v[0]
                preview = v[:min(8, v.shape[0]), :min(8, v.shape[1])]
                preview_np = preview.detach().cpu().numpy()
                result["slices"].append(np.where(np.isfinite(preview_np), preview_np, 0).tolist())

            return result
        except Exception:
            return None

    def _extract_histogram(self, value: torch.Tensor) -> List:
        """Extract histogram data from finite values"""
        flat = value.flatten().detach().cpu()
        finite_flat = flat[torch.isfinite(flat)].numpy()
        if len(finite_flat) > 0:
            return finite_flat[:min(10000, len(finite_flat))].tolist()
        return []

    def _extract_heatmap(self, value: torch.Tensor) -> Dict:
        """Extract heatmap slices for visualization with inf/nan tracking"""
        heatmap_data = {"slices": [], "dims": [], "inf_masks": []}

        def process_slice(t: torch.Tensor):
            """Process a 2D slice, returning data and inf mask"""
            hm = downsample_2d(t)
            # Track inf positions before cleaning
            inf_mask = ~torch.isfinite(hm)
            has_inf = inf_mask.any().item()
            # Clean for display (replace inf/nan with 0)
            clean_hm = torch.where(torch.isfinite(hm), hm, torch.zeros_like(hm))
            data = clean_hm.detach().cpu().numpy().tolist()
            mask = inf_mask.detach().cpu().numpy().tolist() if has_inf else None
            return data, mask

        if value.dim() == 2:
            data, mask = process_slice(value)
            heatmap_data["slices"].append(data)
            heatmap_data["inf_masks"].append(mask)

        elif value.dim() == 3:
            num_slices = min(value.shape[0], 16)
            heatmap_data["dims"] = [{"name": "dim0", "size": value.shape[0], "shown": num_slices}]
            for i in range(num_slices):
                data, mask = process_slice(value[i])
                heatmap_data["slices"].append(data)
                heatmap_data["inf_masks"].append(mask)

        elif value.dim() == 4:
            dim0_size = min(value.shape[0], 4)
            dim1_size = min(value.shape[1], 4)
            heatmap_data["dims"] = [
                {"name": "batch", "size": value.shape[0], "shown": dim0_size},
                {"name": "head", "size": value.shape[1], "shown": dim1_size}
            ]
            for i in range(dim0_size):
                for j in range(dim1_size):
                    data, mask = process_slice(value[i, j])
                    heatmap_data["slices"].append(data)
                    heatmap_data["inf_masks"].append(mask)

        else:
            v = value
            while v.dim() > 2:
                v = v[0]
            data, mask = process_slice(v)
            heatmap_data["slices"].append(data)
            heatmap_data["inf_masks"].append(mask)

        # Remove inf_masks if all empty
        if not any(heatmap_data["inf_masks"]):
            del heatmap_data["inf_masks"]

        return heatmap_data

    def _get_node_type(self, node: fx.Node) -> str:
        """Determine the visualization type for a node"""
        if node.op == "placeholder":
            return "input"
        elif node.op == "output":
            return "output"
        elif node.op == "get_attr":
            return "weight"
        elif node.op == "call_function":
            target_str = str(node.target).lower()
            if "matmul" in target_str or "linear" in target_str or "mm" in target_str:
                return "matmul"
            elif "softmax" in target_str or "relu" in target_str or "gelu" in target_str or "silu" in target_str:
                return "activation"
            elif "add" in target_str or "mul" in target_str or "sub" in target_str:
                return "elementwise"
            elif "layer_norm" in target_str or "norm" in target_str or "batch_norm" in target_str:
                return "norm"
            elif "transpose" in target_str or "reshape" in target_str or "view" in target_str or "permute" in target_str:
                return "reshape"
            elif "conv" in target_str:
                return "matmul"
            return "function"
        elif node.op == "call_module":
            target_str = str(node.target).lower()
            if "linear" in target_str or "conv" in target_str:
                return "matmul"
            elif "relu" in target_str or "gelu" in target_str or "silu" in target_str or "activation" in target_str:
                return "activation"
            elif "norm" in target_str:
                return "norm"
            elif "attention" in target_str or "attn" in target_str:
                return "matmul"
            return "function"
        return "other"

    def extract(self) -> Dict[str, Any]:
        """Extract graph structure with node positions"""
        nodes = []
        edges = []
        node_id_map = {}
        trace_lookup = {t["name"]: t for t in self.execution_trace}

        for idx, node in enumerate(self.gm.graph.nodes):
            node_id = f"node_{idx}"
            node_id_map[node.name] = node_id

            node_info = {
                "id": node_id,
                "name": node.name,
                "op": node.op,
                "target": str(node.target) if node.target else "",
                "step": idx,
                "type": self._get_node_type(node),
            }

            # Add trace data if available
            if node.name in trace_lookup:
                trace = trace_lookup[node.name]
                for key in ["shape", "dtype", "min", "max", "mean", "std", "histogram", "heatmap", "values", "insights"]:
                    if key in trace:
                        node_info[key] = trace[key]

            nodes.append(node_info)

        # Extract edges
        for node in self.gm.graph.nodes:
            target_id = node_id_map.get(node.name)
            if not target_id:
                continue
            for arg in node.args:
                if isinstance(arg, fx.Node):
                    source_id = node_id_map.get(arg.name)
                    if source_id:
                        edges.append({"source": source_id, "target": target_id})
                elif isinstance(arg, (list, tuple)):
                    for a in arg:
                        if isinstance(a, fx.Node):
                            source_id = node_id_map.get(a.name)
                            if source_id:
                                edges.append({"source": source_id, "target": target_id})

        return {
            "nodes": nodes,
            "edges": edges,
            "has_data": len(self.execution_trace) > 0
        }


def main():
    """Test extractor with a demo model"""
    import json
    import argparse
    import torch.nn as nn
    import torch.fx as fx

    parser = argparse.ArgumentParser(description="Test GraphExtractor")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    # Build a simple model
    print("Building test model...")

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 16)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.fc(x)
            x = self.relu(x)
            return x

    model = SimpleModel()
    gm = fx.symbolic_trace(model)

    # Create input data
    input_data = {"x": torch.randn(2, 4, 8)}

    # Extract graph
    print("Extracting graph with tensor data...")
    extractor = GraphExtractor(gm, input_data)
    data = extractor.extract()

    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print(f"\nNodes: {len(data['nodes'])}")
        print(f"Edges: {len(data['edges'])}")
        print(f"Has data: {data['has_data']}")

        print("\nNodes with insights:")
        for node in data['nodes']:
            if 'insights' in node and node['insights']:
                print(f"  {node['name']}: {[i['title'] for i in node['insights']]}")


if __name__ == "__main__":
    main()
