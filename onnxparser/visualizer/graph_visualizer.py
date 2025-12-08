# -*- coding: utf-8 -*-
"""Graph visualization with integrated memory analysis"""

import json
import http.server
import os
import signal
import socket
import socketserver
import subprocess
import sys
import webbrowser
from typing import Dict, Optional, Any, List
import torch
import torch.fx as fx


# Marker to identify our visualizer server process
_VISUALIZER_MARKER = "ONNXPARSER_VISUALIZER_SERVER"


def _is_port_in_use(port: int) -> bool:
    """Check if a port is currently in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def _kill_existing_visualizer(port: int) -> bool:
    """Kill existing visualizer process on the given port."""
    if sys.platform == "darwin" or sys.platform.startswith("linux"):
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0 or not result.stdout.strip():
                return False

            pids = result.stdout.strip().split("\n")
            killed = False

            for pid in pids:
                pid = pid.strip()
                if not pid:
                    continue

                try:
                    is_visualizer = False

                    if sys.platform == "darwin":
                        ps_result = subprocess.run(
                            ["ps", "-p", pid, "-o", "command="],
                            capture_output=True,
                            text=True
                        )
                        cmdline = ps_result.stdout
                        is_visualizer = "python" in cmdline.lower() and (
                            "visualize_graph" in cmdline or
                            "graph_visualizer" in cmdline or
                            "onnxparser" in cmdline
                        )
                    else:
                        with open(f"/proc/{pid}/cmdline", "r") as f:
                            cmdline = f.read()
                            is_visualizer = "visualize_graph" in cmdline or "graph_visualizer" in cmdline

                    if is_visualizer:
                        os.kill(int(pid), signal.SIGTERM)
                        killed = True
                        print(f"Killed visualizer process (PID: {pid})")

                except (ProcessLookupError, PermissionError, FileNotFoundError):
                    continue

            return killed

        except FileNotFoundError:
            return False

    elif sys.platform == "win32":
        try:
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True
            )
            for line in result.stdout.split("\n"):
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        tasklist = subprocess.run(
                            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV"],
                            capture_output=True,
                            text=True
                        )
                        if "python" in tasklist.stdout.lower():
                            subprocess.run(["taskkill", "/F", "/PID", pid])
                            print(f"Killed visualizer process (PID: {pid})")
                            return True
        except FileNotFoundError:
            return False

    return False


class GraphVisualizer:
    """Visualize FX GraphModule with integrated memory analysis"""

    def __init__(self, gm: fx.GraphModule, input_data: Optional[Dict[str, torch.Tensor]] = None):
        self.gm = gm
        self.input_data = input_data
        self.execution_trace = []

        if input_data:
            self._trace_execution()

        self.graph_data = self._extract_graph_data()
        self.memory_data = self._analyze_memory()

    def _trace_execution(self):
        """Trace execution to capture intermediate values"""
        env = {}

        for node in self.gm.graph.nodes:
            if node.op == "placeholder":
                if node.name in self.input_data:
                    env[node.name] = self.input_data[node.name]
                else:
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
                kwargs = {k: env.get(v.name) if isinstance(v, fx.Node) else v for k, v in node.kwargs.items()}
                try:
                    result = node.target(*args, **kwargs)
                    env[node.name] = result
                except Exception:
                    env[node.name] = None
            elif node.op == "call_method":
                obj = env.get(node.args[0].name) if node.args else None
                if obj is not None:
                    args = [env.get(a.name) if isinstance(a, fx.Node) else a for a in node.args[1:]]
                    kwargs = {k: env.get(v.name) if isinstance(v, fx.Node) else v for k, v in node.kwargs.items()}
                    try:
                        result = getattr(obj, node.target)(*args, **kwargs)
                        env[node.name] = result
                    except Exception:
                        env[node.name] = None

            value = env.get(node.name)
            trace_info = {"name": node.name, "op": node.op}

            if isinstance(value, torch.Tensor):
                trace_info["shape"] = list(value.shape)
                trace_info["dtype"] = str(value.dtype)
                trace_info["min"] = float(value.min().item())
                trace_info["max"] = float(value.max().item())
                trace_info["mean"] = float(value.mean().item())
                trace_info["std"] = float(value.std().item()) if value.numel() > 1 else 0.0

                flat = value.flatten().detach().cpu().numpy()
                trace_info["histogram"] = flat[:min(10000, len(flat))].tolist()

                if value.dim() >= 2:
                    if value.dim() == 2:
                        hm = value
                    elif value.dim() == 3:
                        hm = value[0]
                    else:
                        hm = value.reshape(-1, value.shape[-1])

                    h, w = hm.shape
                    if h > 64:
                        hm = hm[::h // 64]
                    if w > 64:
                        hm = hm[:, ::w // 64]
                    trace_info["heatmap"] = hm.detach().cpu().numpy().tolist()

                trace_info["full_data"] = value.detach().cpu().numpy().tolist()

            self.execution_trace.append(trace_info)

    def _extract_graph_data(self) -> Dict[str, Any]:
        """Extract graph structure"""
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
            }

            if node.name in trace_lookup:
                trace = trace_lookup[node.name]
                for key in ["shape", "dtype", "min", "max", "mean", "std", "histogram", "heatmap", "full_data"]:
                    if key in trace:
                        node_info[key] = trace[key]

            if node.op == "placeholder":
                node_info["type"] = "input"
            elif node.op == "output":
                node_info["type"] = "output"
            elif node.op == "call_function":
                target_str = str(node.target)
                if "matmul" in target_str or "linear" in target_str:
                    node_info["type"] = "matmul"
                elif "softmax" in target_str or "relu" in target_str or "gelu" in target_str:
                    node_info["type"] = "activation"
                elif "add" in target_str or "mul" in target_str:
                    node_info["type"] = "elementwise"
                elif "layer_norm" in target_str or "norm" in target_str:
                    node_info["type"] = "norm"
                elif "transpose" in target_str or "reshape" in target_str:
                    node_info["type"] = "reshape"
                else:
                    node_info["type"] = "function"
            elif node.op == "get_attr":
                node_info["type"] = "weight"
            else:
                node_info["type"] = "other"

            nodes.append(node_info)

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

        return {"nodes": nodes, "edges": edges, "has_data": len(self.execution_trace) > 0}

    def _analyze_memory(self) -> Dict[str, Any]:
        """Run memory analysis with multiple strategies"""
        try:
            from ..analysis.memory_analyzer import MemoryAnalyzer
            from ..analysis.strategies import StrategyRegistry

            strategies_data = {}
            available_strategies = StrategyRegistry.list_strategies()

            for strategy in available_strategies:
                try:
                    analyzer = MemoryAnalyzer(self.gm, strategy=strategy)
                    result = analyzer.analyze()

                    # Build tensor lifetime data
                    tensors = []
                    for name, tensor in result.tensors.items():
                        if tensor.is_weight:
                            continue
                        tensors.append({
                            "name": name,
                            "shape": tensor.shape,
                            "dtype": str(tensor.dtype),
                            "size_bytes": tensor.size_bytes,
                            "size_kb": tensor.size_bytes / 1024,
                            "birth": tensor.birth_step,
                            "death": tensor.death_step,
                            "is_input": tensor.is_input,
                            "is_output": tensor.is_output,
                            "reused_from": tensor.reused_from,
                            "memory_offset": tensor.memory_offset,
                        })

                    # Build memory timeline
                    timeline = []
                    for step in result.steps:
                        timeline.append({
                            "step": step.step,
                            "node": step.node_name,
                            "max_memory": step.max_memory,
                            "min_memory": step.min_memory,
                            "live_tensors": step.live_tensors,
                        })

                    strategies_data[strategy] = {
                        "tensors": tensors,
                        "timeline": timeline,
                        "summary": {
                            "peak_max_kb": result.peak_max_memory / 1024,
                            "peak_min_kb": result.peak_min_memory / 1024,
                            "savings_pct": result.savings_ratio * 100,
                        }
                    }
                except Exception as e:
                    strategies_data[strategy] = {"error": str(e)}

            return {
                "available": True,
                "strategies": strategies_data,
                "default_strategy": "greedy" if "greedy" in strategies_data else available_strategies[0] if available_strategies else None,
            }

        except ImportError:
            return {"available": False, "error": "Memory analyzer not available"}
        except Exception as e:
            return {"available": False, "error": str(e)}

    def to_html(self) -> str:
        """Generate HTML with integrated visualization"""
        graph_json = json.dumps(self.graph_data)
        memory_json = json.dumps(self.memory_data)

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ONNX Parser - Graph Visualizer</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e2e8f0;
            overflow: hidden;
            height: 100vh;
        }}
        #app {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 48px 1fr 1fr;
            grid-template-areas:
                "header header"
                "graph sidebar"
                "memory .";
            height: 100vh;
            transition: all 0.3s ease;
        }}
        #app.focus-graph {{
            grid-template-areas:
                "header header"
                "graph graph"
                "graph graph";
        }}
        #app.focus-graph #sidebar,
        #app.focus-graph #memory-panel {{ display: none; }}
        #app.focus-sidebar {{
            grid-template-areas:
                "header header"
                "sidebar sidebar"
                "sidebar sidebar";
        }}
        #app.focus-sidebar #graph-container,
        #app.focus-sidebar #memory-panel {{ display: none; }}
        #app.focus-memory {{
            grid-template-areas:
                "header header"
                "memory memory"
                "memory memory";
        }}
        #app.focus-memory #graph-container,
        #app.focus-memory #sidebar {{ display: none; }}
        header {{
            grid-area: header;
            background: rgba(22, 33, 62, 0.95);
            backdrop-filter: blur(10px);
            display: flex;
            align-items: center;
            padding: 0 20px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            gap: 20px;
        }}
        header h1 {{
            font-size: 16px;
            font-weight: 600;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .nav-tabs {{
            display: flex;
            gap: 4px;
            margin-left: 20px;
        }}
        .nav-tab {{
            padding: 6px 12px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 6px;
            color: #94a3b8;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }}
        .nav-tab:hover {{
            background: rgba(255,255,255,0.1);
            color: #e2e8f0;
        }}
        .nav-tab.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-color: transparent;
            color: white;
        }}
        .controls {{
            display: flex;
            gap: 8px;
            margin-left: auto;
        }}
        .btn {{
            padding: 6px 14px;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 6px;
            color: #e2e8f0;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
            transition: all 0.2s;
        }}
        .btn:hover {{ background: rgba(255,255,255,0.15); }}
        .btn.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-color: transparent;
        }}
        #graph-container {{
            grid-area: graph;
            position: relative;
            overflow: hidden;
            background: rgba(0,0,0,0.2);
        }}
        #graph-svg {{
            width: 100%;
            height: 100%;
        }}
        #sidebar {{
            grid-area: sidebar;
            background: rgba(22, 33, 62, 0.95);
            backdrop-filter: blur(10px);
            border-left: 1px solid rgba(255,255,255,0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        .sidebar-header {{
            padding: 12px 16px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        #search {{
            width: 100%;
            padding: 8px 12px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 6px;
            color: #e2e8f0;
            font-size: 13px;
            outline: none;
        }}
        #search:focus {{
            border-color: #667eea;
        }}
        .sidebar-content {{
            flex: 1;
            overflow-y: auto;
            padding: 16px;
        }}
        .section {{
            margin-bottom: 20px;
        }}
        .section-title {{
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #94a3b8;
            margin-bottom: 10px;
        }}
        .info-grid {{
            display: grid;
            gap: 6px;
        }}
        .info-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 10px;
            background: rgba(255,255,255,0.03);
            border-radius: 4px;
            font-size: 12px;
        }}
        .info-label {{ color: #94a3b8; }}
        .info-value {{ color: #e2e8f0; font-family: 'SF Mono', monospace; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
            border: 1px solid rgba(102,126,234,0.2);
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 16px;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .stat-label {{
            font-size: 10px;
            color: #94a3b8;
            text-transform: uppercase;
            margin-top: 2px;
        }}
        #histogram-plot, #heatmap-plot {{
            width: 100%;
            height: 140px;
            border-radius: 8px;
            overflow: hidden;
        }}
        .placeholder {{
            display: flex;
            align-items: center;
            justify-content: center;
            height: 150px;
            color: #64748b;
            font-size: 13px;
        }}
        /* Memory panel */
        #memory-panel {{
            grid-area: memory;
            background: rgba(22, 33, 62, 0.95);
            border-top: 1px solid rgba(255,255,255,0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        .memory-header {{
            display: flex;
            align-items: center;
            padding: 10px 16px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            gap: 16px;
        }}
        .memory-header h3 {{
            font-size: 13px;
            font-weight: 600;
            color: #f59e0b;
        }}
        .strategy-select {{
            padding: 5px 10px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 4px;
            color: #e2e8f0;
            font-size: 12px;
            cursor: pointer;
        }}
        .memory-stats {{
            display: flex;
            gap: 20px;
            margin-left: auto;
            font-size: 12px;
        }}
        .memory-stats span {{
            color: #94a3b8;
        }}
        .memory-stats strong {{
            color: #10b981;
        }}
        #memory-chart {{
            flex: 1;
            min-height: 0;
        }}
        /* Graph styles */
        .node {{ cursor: pointer; }}
        .node rect {{
            rx: 8; ry: 8;
            stroke-width: 2px;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
            transition: all 0.2s;
        }}
        .node:hover rect {{
            filter: drop-shadow(0 4px 12px rgba(102,126,234,0.4));
        }}
        .node.selected rect {{
            stroke: #667eea !important;
            stroke-width: 3px;
            filter: drop-shadow(0 0 15px rgba(102,126,234,0.6));
        }}
        .node.highlighted rect {{
            stroke: #f59e0b !important;
            stroke-width: 3px;
            filter: drop-shadow(0 0 15px rgba(245,158,11,0.6));
        }}
        .node.dimmed {{
            opacity: 0.3;
        }}
        .node text {{ fill: white; pointer-events: none; }}
        .node-name {{ font-size: 11px; font-weight: 600; }}
        .node-shape {{ font-size: 9px; opacity: 0.7; }}
        .link {{
            fill: none;
            stroke: rgba(255,255,255,0.15);
            stroke-width: 1.5px;
        }}
        .link.highlighted {{
            stroke: #667eea;
            stroke-width: 2px;
        }}
        /* Node colors */
        .node-input rect {{ fill: #10b981; stroke: #059669; }}
        .node-output rect {{ fill: #ef4444; stroke: #dc2626; }}
        .node-matmul rect {{ fill: #3b82f6; stroke: #2563eb; }}
        .node-activation rect {{ fill: #8b5cf6; stroke: #7c3aed; }}
        .node-elementwise rect {{ fill: #f59e0b; stroke: #d97706; }}
        .node-norm rect {{ fill: #14b8a6; stroke: #0d9488; }}
        .node-reshape rect {{ fill: #f97316; stroke: #ea580c; }}
        .node-weight rect {{ fill: #6b7280; stroke: #4b5563; }}
        .node-function rect {{ fill: #374151; stroke: #4b5563; }}
        .node-other rect {{ fill: #374151; stroke: #4b5563; }}
        .node-small rect {{ rx: 5; ry: 5; opacity: 0.85; }}
        .node-small text {{ font-size: 9px; }}
        .order-badge {{
            fill: #fbbf24;
            font-size: 9px;
            font-weight: 700;
        }}
        .legend {{
            padding: 12px 16px;
            border-top: 1px solid rgba(255,255,255,0.1);
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            color: #94a3b8;
        }}
        .legend-dot {{
            width: 10px;
            height: 10px;
            border-radius: 3px;
        }}
        /* Memory chart legend */
        .memory-legend {{
            display: flex;
            gap: 16px;
            padding: 8px 16px;
            font-size: 11px;
            color: #94a3b8;
            border-top: 1px solid rgba(255,255,255,0.05);
        }}
        .memory-legend-item {{
            display: flex;
            align-items: center;
            gap: 4px;
        }}
        .memory-legend-dot {{
            width: 12px;
            height: 4px;
            border-radius: 2px;
        }}
    </style>
</head>
<body>
    <div id="app">
        <header>
            <h1>ONNX Parser Visualizer</h1>
            <div class="nav-tabs">
                <button class="nav-tab active" data-view="all" onclick="setView('all')">All</button>
                <button class="nav-tab" data-view="graph" onclick="setView('graph')">Graph</button>
                <button class="nav-tab" data-view="sidebar" onclick="setView('sidebar')">Node Info</button>
                <button class="nav-tab" data-view="memory" onclick="setView('memory')">Memory</button>
            </div>
            <div class="controls">
                <button class="btn" onclick="resetView()">Reset View</button>
                <button class="btn" id="flowBtn" onclick="toggleFlow()">Data Flow</button>
            </div>
        </header>
        <div id="graph-container">
            <svg id="graph-svg"></svg>
        </div>
        <div id="sidebar">
            <div class="sidebar-header">
                <input type="text" id="search" placeholder="Search nodes...">
            </div>
            <div class="sidebar-content">
                <div id="placeholder" class="placeholder">Click a node to view details</div>
                <div id="details" style="display:none;">
                    <div class="section">
                        <div class="section-title">Node Info</div>
                        <div class="info-grid">
                            <div class="info-row"><span class="info-label">Name</span><span class="info-value" id="info-name">-</span></div>
                            <div class="info-row"><span class="info-label">Op</span><span class="info-value" id="info-op">-</span></div>
                            <div class="info-row"><span class="info-label">Target</span><span class="info-value" id="info-target">-</span></div>
                            <div class="info-row"><span class="info-label">Shape</span><span class="info-value" id="info-shape">-</span></div>
                            <div class="info-row"><span class="info-label">Dtype</span><span class="info-value" id="info-dtype">-</span></div>
                        </div>
                    </div>
                    <div class="section" id="stats-section" style="display:none;">
                        <div class="section-title">Statistics</div>
                        <div class="stats-grid">
                            <div class="stat-card"><div class="stat-value" id="stat-min">-</div><div class="stat-label">Min</div></div>
                            <div class="stat-card"><div class="stat-value" id="stat-max">-</div><div class="stat-label">Max</div></div>
                            <div class="stat-card"><div class="stat-value" id="stat-mean">-</div><div class="stat-label">Mean</div></div>
                            <div class="stat-card"><div class="stat-value" id="stat-std">-</div><div class="stat-label">Std</div></div>
                        </div>
                    </div>
                    <div class="section" id="histogram-section" style="display:none;">
                        <div class="section-title">Distribution</div>
                        <div id="histogram-plot"></div>
                    </div>
                    <div class="section" id="heatmap-section" style="display:none;">
                        <div class="section-title">Tensor Heatmap</div>
                        <div id="heatmap-plot"></div>
                    </div>
                </div>
            </div>
            <div class="legend">
                <div class="legend-item"><div class="legend-dot" style="background:#10b981"></div>Input</div>
                <div class="legend-item"><div class="legend-dot" style="background:#ef4444"></div>Output</div>
                <div class="legend-item"><div class="legend-dot" style="background:#3b82f6"></div>MatMul</div>
                <div class="legend-item"><div class="legend-dot" style="background:#8b5cf6"></div>Activation</div>
                <div class="legend-item"><div class="legend-dot" style="background:#f59e0b"></div>Elementwise</div>
                <div class="legend-item"><div class="legend-dot" style="background:#14b8a6"></div>Norm</div>
            </div>
        </div>
        <div id="memory-panel">
            <div class="memory-header">
                <h3>Memory Lifetime</h3>
                <select class="strategy-select" id="strategy-select" onchange="changeStrategy()">
                </select>
                <div class="memory-stats">
                    <span>Peak: <strong id="mem-peak">-</strong> KB</span>
                    <span>Savings: <strong id="mem-savings">-</strong>%</span>
                </div>
            </div>
            <div id="memory-chart"></div>
            <div class="memory-legend">
                <div class="memory-legend-item"><div class="memory-legend-dot" style="background:#10b981"></div>Input</div>
                <div class="memory-legend-item"><div class="memory-legend-dot" style="background:#3b82f6"></div>Tensor</div>
                <div class="memory-legend-item"><div class="memory-legend-dot" style="background:#8b5cf6"></div>Reused</div>
                <div class="memory-legend-item"><div class="memory-legend-dot" style="background:#f59e0b"></div>Output</div>
                <div class="memory-legend-item" style="margin-left:auto;color:#64748b;">Click block to highlight node</div>
            </div>
        </div>
    </div>

    <script>
        const graphData = {graph_json};
        const memoryData = {memory_json};
        const container = document.getElementById('graph-container');
        const svg = d3.select('#graph-svg');
        const g = svg.append('g');
        let currentStrategy = memoryData.default_strategy || 'greedy';
        let selectedNode = null;
        let highlightedTensor = null;

        // Zoom
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', e => g.attr('transform', e.transform));
        svg.call(zoom);

        // Arrow marker
        svg.append('defs').append('marker')
            .attr('id', 'arrow')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 25)
            .attr('refY', 0)
            .attr('orient', 'auto')
            .attr('markerWidth', 5)
            .attr('markerHeight', 5)
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', 'rgba(255,255,255,0.2)');

        // Layout
        const nodeW = 120, nodeH = 38, layerGap = 160, nodeGap = 14;
        const width = container.clientWidth, height = container.clientHeight;
        const nodeById = {{}};
        graphData.nodes.forEach(n => nodeById[n.id] = n);

        const incoming = {{}}, outgoing = {{}};
        graphData.nodes.forEach(n => {{ incoming[n.id] = []; outgoing[n.id] = []; }});
        graphData.edges.forEach(e => {{
            incoming[e.target].push(e.source);
            outgoing[e.source].push(e.target);
        }});

        // Separate weight nodes from computation nodes
        const weightNodes = graphData.nodes.filter(n => n.type === 'weight');
        const computeNodes = graphData.nodes.filter(n => n.type !== 'weight');

        // Assign layers only to compute nodes
        function assignLayers() {{
            const layers = {{}};
            function getLayer(id) {{
                if (layers[id] !== undefined) return layers[id];
                const node = nodeById[id];
                if (node.type === 'weight') return -1;
                const deps = incoming[id].filter(srcId => nodeById[srcId].type !== 'weight');
                if (deps.length === 0) return layers[id] = 0;
                return layers[id] = Math.max(...deps.map(getLayer)) + 1;
            }}
            computeNodes.forEach(n => getLayer(n.id));
            return layers;
        }}

        const layers = assignLayers();
        const maxLayer = Math.max(...Object.values(layers), 0);

        // Group compute nodes by layer
        const layerNodes = {{}};
        computeNodes.forEach(n => {{
            const l = layers[n.id];
            if (!layerNodes[l]) layerNodes[l] = [];
            layerNodes[l].push(n);
        }});

        // Position compute nodes
        const totalWidth = (maxLayer + 1) * (nodeW + layerGap);
        const startX = Math.max(60, (width - totalWidth) / 2);

        Object.keys(layerNodes).forEach(l => {{
            const nodes = layerNodes[l];
            const total = nodes.length * (nodeH + nodeGap) - nodeGap;
            const startY = (height - total) / 2;
            nodes.forEach((n, idx) => {{
                n.x = startX + parseInt(l) * (nodeW + layerGap);
                n.y = startY + idx * (nodeH + nodeGap);
            }});
        }});

        // Position weight nodes
        const weightW = 80, weightH = 28;
        weightNodes.forEach(n => {{
            n.isWeight = true;
            n.nodeW = weightW;
            n.nodeH = weightH;
            const consumers = outgoing[n.id].map(id => nodeById[id]).filter(c => c.x !== undefined);
            if (consumers.length > 0) {{
                const consumer = consumers[0];
                const siblingWeights = incoming[consumer.id].map(id => nodeById[id]).filter(sib => sib.type === 'weight');
                const sibIdx = siblingWeights.indexOf(n);
                n.x = consumer.x - weightW - 25;
                n.y = consumer.y - 15 + sibIdx * (weightH + 6);
            }} else {{
                n.x = 20;
                n.y = 50 + weightNodes.indexOf(n) * (weightH + 6);
            }}
        }});

        computeNodes.forEach(n => {{
            n.nodeW = nodeW;
            n.nodeH = nodeH;
        }});

        // Draw edges
        const links = g.selectAll('.link')
            .data(graphData.edges)
            .enter()
            .append('path')
            .attr('class', 'link')
            .attr('marker-end', 'url(#arrow)')
            .attr('d', d => {{
                const s = nodeById[d.source], t = nodeById[d.target];
                const sw = s.nodeW || nodeW, sh = s.nodeH || nodeH;
                const th = t.nodeH || nodeH;
                const sx = s.x + sw, sy = s.y + sh/2;
                const tx = t.x, ty = t.y + th/2;
                const mx = (sx + tx) / 2;
                return `M${{sx}},${{sy}} C${{mx}},${{sy}} ${{mx}},${{ty}} ${{tx}},${{ty}}`;
            }});

        // Draw nodes
        const nodes = g.selectAll('.node')
            .data(graphData.nodes)
            .enter()
            .append('g')
            .attr('class', d => `node node-${{d.type}}${{d.isWeight ? ' node-small' : ''}}`)
            .attr('data-name', d => d.name)
            .attr('transform', d => `translate(${{d.x}},${{d.y}})`)
            .on('click', selectNode);

        nodes.append('rect')
            .attr('width', d => d.nodeW || nodeW)
            .attr('height', d => d.nodeH || nodeH);
        nodes.append('text')
            .attr('class', 'node-name')
            .attr('x', d => (d.nodeW || nodeW)/2)
            .attr('y', d => d.isWeight ? 12 : 15)
            .attr('text-anchor', 'middle')
            .text(d => {{
                const maxLen = d.isWeight ? 10 : 14;
                return d.name.length > maxLen ? d.name.slice(0, maxLen-2)+'..' : d.name;
            }});
        nodes.filter(d => !d.isWeight).append('text')
            .attr('class', 'node-shape')
            .attr('x', d => (d.nodeW || nodeW)/2)
            .attr('y', 28)
            .attr('text-anchor', 'middle')
            .text(d => d.shape ? JSON.stringify(d.shape) : '');

        // Order badges
        const badgeNodes = nodes.filter(d => !d.isWeight);
        badgeNodes.append('circle').attr('cx', -5).attr('cy', -5).attr('r', 8).attr('fill', '#fbbf24');
        badgeNodes.append('text')
            .attr('class', 'order-badge')
            .attr('x', -5).attr('y', -2)
            .attr('text-anchor', 'middle')
            .text((d,i) => i+1);

        // Node selection
        function selectNode(e, d) {{
            d3.selectAll('.node').classed('selected', false).classed('dimmed', false);
            d3.selectAll('.link').classed('highlighted', false);
            d3.select(this).classed('selected', true);
            links.classed('highlighted', l => l.source === d.id || l.target === d.id);
            selectedNode = d;

            document.getElementById('placeholder').style.display = 'none';
            document.getElementById('details').style.display = 'block';
            document.getElementById('info-name').textContent = d.name;
            document.getElementById('info-op').textContent = d.op;
            document.getElementById('info-target').textContent = d.target || '-';
            document.getElementById('info-shape').textContent = d.shape ? JSON.stringify(d.shape) : '-';
            document.getElementById('info-dtype').textContent = d.dtype || '-';

            const statsSection = document.getElementById('stats-section');
            const histSection = document.getElementById('histogram-section');
            const heatSection = document.getElementById('heatmap-section');

            if (d.min !== undefined) {{
                statsSection.style.display = 'block';
                document.getElementById('stat-min').textContent = d.min.toFixed(4);
                document.getElementById('stat-max').textContent = d.max.toFixed(4);
                document.getElementById('stat-mean').textContent = d.mean.toFixed(4);
                document.getElementById('stat-std').textContent = d.std.toFixed(4);

                if (d.histogram) {{
                    histSection.style.display = 'block';
                    Plotly.newPlot('histogram-plot', [{{
                        x: d.histogram,
                        type: 'histogram',
                        marker: {{
                            color: 'rgba(102, 126, 234, 0.7)',
                            line: {{ color: 'rgba(102, 126, 234, 1)', width: 1 }}
                        }},
                        nbinsx: 25
                    }}], {{
                        paper_bgcolor: 'transparent',
                        plot_bgcolor: 'transparent',
                        margin: {{ t: 5, r: 5, b: 25, l: 35 }},
                        xaxis: {{ color: '#94a3b8', gridcolor: 'rgba(255,255,255,0.1)' }},
                        yaxis: {{ color: '#94a3b8', gridcolor: 'rgba(255,255,255,0.1)' }},
                        bargap: 0.05
                    }}, {{ responsive: true, displayModeBar: false }});
                }} else {{
                    histSection.style.display = 'none';
                }}

                if (d.heatmap) {{
                    heatSection.style.display = 'block';
                    Plotly.newPlot('heatmap-plot', [{{
                        z: d.heatmap,
                        type: 'heatmap',
                        colorscale: [[0, '#3b0764'], [0.5, '#7c3aed'], [1, '#f5f3ff']],
                        showscale: false,
                    }}], {{
                        paper_bgcolor: 'transparent',
                        plot_bgcolor: 'transparent',
                        margin: {{ t: 5, r: 5, b: 25, l: 35 }},
                        xaxis: {{ color: '#94a3b8', showgrid: false }},
                        yaxis: {{ color: '#94a3b8', showgrid: false, autorange: 'reversed' }}
                    }}, {{ responsive: true, displayModeBar: false }});
                }} else {{
                    heatSection.style.display = 'none';
                }}
            }} else {{
                statsSection.style.display = 'none';
                histSection.style.display = 'none';
                heatSection.style.display = 'none';
            }}

            // Highlight corresponding memory block
            highlightMemoryBlock(d.name);
        }}

        // Search
        document.getElementById('search').addEventListener('input', e => {{
            const q = e.target.value.toLowerCase();
            nodes.style('opacity', d => !q || d.name.toLowerCase().includes(q) ? 1 : 0.15);
        }});

        // Reset view
        function resetView() {{
            const bounds = g.node().getBBox();
            const scale = Math.min(width/(bounds.width+80), height/(bounds.height+80)) * 0.85;
            const tx = (width - bounds.width*scale)/2 - bounds.x*scale;
            const ty = (height - bounds.height*scale)/2 - bounds.y*scale;
            svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
        }}

        // Data flow animation
        let flowActive = false;
        function toggleFlow() {{
            flowActive = !flowActive;
            document.getElementById('flowBtn').classList.toggle('active', flowActive);
            if (flowActive) {{
                links.style('stroke-dasharray', '8,4').style('animation', 'dash 0.5s linear infinite');
            }} else {{
                links.style('stroke-dasharray', null).style('animation', null);
            }}
        }}

        // View switching
        let currentView = 'all';
        function setView(view) {{
            currentView = view;
            const app = document.getElementById('app');
            app.className = '';
            if (view !== 'all') {{
                app.classList.add('focus-' + view);
            }}
            // Update nav tabs
            document.querySelectorAll('.nav-tab').forEach(tab => {{
                tab.classList.toggle('active', tab.dataset.view === view);
            }});
            // Re-render charts after layout change
            setTimeout(() => {{
                if (view === 'all' || view === 'graph') {{
                    resetView();
                }}
                if (view === 'all' || view === 'memory') {{
                    renderMemoryChart();
                }}
            }}, 50);
        }}

        // === Memory visualization ===
        function initMemoryPanel() {{
            if (!memoryData.available) {{
                document.getElementById('memory-chart').innerHTML =
                    '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#64748b;">Memory analysis not available</div>';
                return;
            }}

            const select = document.getElementById('strategy-select');
            const strategies = Object.keys(memoryData.strategies);
            select.innerHTML = strategies.map(s =>
                `<option value="${{s}}" ${{s === currentStrategy ? 'selected' : ''}}>${{formatStrategyName(s)}}</option>`
            ).join('');

            renderMemoryChart();
        }}

        function formatStrategyName(name) {{
            return name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
        }}

        function changeStrategy() {{
            currentStrategy = document.getElementById('strategy-select').value;
            renderMemoryChart();
        }}

        function renderMemoryChart() {{
            if (!memoryData.available || !memoryData.strategies[currentStrategy]) return;

            const stratData = memoryData.strategies[currentStrategy];
            if (stratData.error) {{
                document.getElementById('memory-chart').innerHTML =
                    `<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#ef4444;">Error: ${{stratData.error}}</div>`;
                return;
            }}

            // Update stats
            document.getElementById('mem-peak').textContent = stratData.summary.peak_min_kb.toFixed(1);
            document.getElementById('mem-savings').textContent = stratData.summary.savings_pct.toFixed(1);

            const tensors = stratData.tensors;
            if (!tensors || tensors.length === 0) {{
                document.getElementById('memory-chart').innerHTML =
                    '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#64748b;">No tensor data</div>';
                return;
            }}

            // Sort tensors by birth time for better visualization
            tensors.sort((a, b) => a.birth - b.birth);

            // Create Gantt-style chart
            const maxStep = Math.max(...tensors.map(t => t.death)) + 1;

            const traces = tensors.map((t, idx) => {{
                let color;
                if (t.is_input) color = '#10b981';
                else if (t.is_output) color = '#f59e0b';
                else if (t.reused_from) color = '#8b5cf6';
                else color = '#3b82f6';

                return {{
                    x: [t.death - t.birth + 1],
                    y: [t.name],
                    type: 'bar',
                    orientation: 'h',
                    base: [t.birth],
                    marker: {{ color: color, line: {{ color: 'rgba(255,255,255,0.3)', width: 1 }} }},
                    hovertemplate: `<b>${{t.name}}</b><br>Shape: ${{JSON.stringify(t.shape)}}<br>Size: ${{t.size_kb.toFixed(2)}} KB<br>Lifetime: Step ${{t.birth}} - ${{t.death}}<extra></extra>`,
                    showlegend: false,
                    customdata: [t],
                }};
            }});

            const layout = {{
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                margin: {{ t: 10, r: 20, b: 30, l: 120 }},
                xaxis: {{
                    title: {{ text: 'Execution Step', font: {{ size: 11, color: '#94a3b8' }} }},
                    color: '#94a3b8',
                    gridcolor: 'rgba(255,255,255,0.05)',
                    range: [-0.5, maxStep + 0.5],
                }},
                yaxis: {{
                    color: '#94a3b8',
                    gridcolor: 'rgba(255,255,255,0.03)',
                    tickfont: {{ size: 10 }},
                }},
                barmode: 'overlay',
                hovermode: 'closest',
            }};

            Plotly.newPlot('memory-chart', traces, layout, {{
                responsive: true,
                displayModeBar: false,
            }});

            // Click handler for memory blocks
            document.getElementById('memory-chart').on('plotly_click', function(eventData) {{
                const tensor = eventData.points[0].customdata;
                if (tensor) {{
                    highlightGraphNode(tensor.name);
                }}
            }});
        }}

        function highlightGraphNode(tensorName) {{
            // Clear previous highlights
            d3.selectAll('.node').classed('highlighted', false).classed('dimmed', false);

            // Find matching node
            const matchingNode = graphData.nodes.find(n => n.name === tensorName);
            if (matchingNode) {{
                // Dim all other nodes
                d3.selectAll('.node').classed('dimmed', true);

                // Highlight matching node
                d3.select(`.node[data-name="${{tensorName}}"]`)
                    .classed('highlighted', true)
                    .classed('dimmed', false);

                highlightedTensor = tensorName;

                // Also highlight connected edges
                links.classed('highlighted', l => l.source === matchingNode.id || l.target === matchingNode.id);

                // Pan to node
                const node = nodeById[matchingNode.id];
                if (node && node.x !== undefined) {{
                    const currentTransform = d3.zoomTransform(svg.node());
                    const targetX = width/2 - node.x * currentTransform.k - (node.nodeW || nodeW)/2 * currentTransform.k;
                    const targetY = height/2 - node.y * currentTransform.k - (node.nodeH || nodeH)/2 * currentTransform.k;
                    svg.transition().duration(300).call(
                        zoom.transform,
                        d3.zoomIdentity.translate(targetX, targetY).scale(currentTransform.k)
                    );
                }}
            }}
        }}

        function highlightMemoryBlock(nodeName) {{
            if (!memoryData.available) return;

            // Re-render with highlight
            const stratData = memoryData.strategies[currentStrategy];
            if (!stratData || stratData.error) return;

            const tensors = stratData.tensors;
            const maxStep = Math.max(...tensors.map(t => t.death)) + 1;

            const traces = tensors.map((t, idx) => {{
                let color;
                const isHighlighted = t.name === nodeName;

                if (isHighlighted) {{
                    color = '#fbbf24';  // Highlight color
                }} else if (t.is_input) {{
                    color = 'rgba(16, 185, 129, 0.4)';
                }} else if (t.is_output) {{
                    color = 'rgba(245, 158, 11, 0.4)';
                }} else if (t.reused_from) {{
                    color = 'rgba(139, 92, 246, 0.4)';
                }} else {{
                    color = 'rgba(59, 130, 246, 0.4)';
                }}

                return {{
                    x: [t.death - t.birth + 1],
                    y: [t.name],
                    type: 'bar',
                    orientation: 'h',
                    base: [t.birth],
                    marker: {{
                        color: color,
                        line: {{
                            color: isHighlighted ? '#fbbf24' : 'rgba(255,255,255,0.2)',
                            width: isHighlighted ? 2 : 1
                        }}
                    }},
                    hovertemplate: `<b>${{t.name}}</b><br>Shape: ${{JSON.stringify(t.shape)}}<br>Size: ${{t.size_kb.toFixed(2)}} KB<br>Lifetime: Step ${{t.birth}} - ${{t.death}}<extra></extra>`,
                    showlegend: false,
                    customdata: [t],
                }};
            }});

            const layout = {{
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                margin: {{ t: 10, r: 20, b: 30, l: 120 }},
                xaxis: {{
                    title: {{ text: 'Execution Step', font: {{ size: 11, color: '#94a3b8' }} }},
                    color: '#94a3b8',
                    gridcolor: 'rgba(255,255,255,0.05)',
                    range: [-0.5, maxStep + 0.5],
                }},
                yaxis: {{
                    color: '#94a3b8',
                    gridcolor: 'rgba(255,255,255,0.03)',
                    tickfont: {{ size: 10 }},
                }},
                barmode: 'overlay',
                hovermode: 'closest',
            }};

            Plotly.react('memory-chart', traces, layout, {{
                responsive: true,
                displayModeBar: false,
            }});
        }}

        // Clear highlights when clicking empty space
        svg.on('click', function(e) {{
            if (e.target === svg.node()) {{
                d3.selectAll('.node').classed('selected', false).classed('highlighted', false).classed('dimmed', false);
                d3.selectAll('.link').classed('highlighted', false);
                selectedNode = null;
                highlightedTensor = null;
                document.getElementById('placeholder').style.display = 'flex';
                document.getElementById('details').style.display = 'none';
                renderMemoryChart();  // Reset memory chart colors
            }}
        }});

        // Initialize
        resetView();
        initMemoryPanel();
    </script>
</body>
</html>'''
        return html

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_html())
        print(f"Saved: {path}")

    def serve(self, port: int = 8080, open_browser: bool = True) -> None:
        os.environ[_VISUALIZER_MARKER] = str(port)

        if _is_port_in_use(port):
            print(f"Port {port} is in use, checking for existing visualizer...")
            if _kill_existing_visualizer(port):
                print("Killed existing visualizer, restarting...")
                import time
                time.sleep(0.5)
            else:
                print(f"Port {port} is in use by another process, trying to start anyway...")

        html = self.to_html()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(html.encode())

            def log_message(self, *args):
                pass

        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.TCPServer(("", port), Handler) as httpd:
            url = f"http://localhost:{port}"
            print(f"Serving: {url}")
            if open_browser:
                webbrowser.open(url)
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nStopped")


def visualize(gm: fx.GraphModule, output_path: Optional[str] = None,
              input_data: Optional[Dict[str, torch.Tensor]] = None) -> GraphVisualizer:
    viz = GraphVisualizer(gm, input_data)
    if output_path:
        viz.save(output_path)
    return viz


def serve(gm: fx.GraphModule, port: int = 8080,
          input_data: Optional[Dict[str, torch.Tensor]] = None) -> None:
    viz = GraphVisualizer(gm, input_data)
    viz.serve(port=port)
