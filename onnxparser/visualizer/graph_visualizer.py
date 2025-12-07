# -*- coding: utf-8 -*-
"""Graph visualization with Plotly for beautiful data visualization"""

import json
import http.server
import socketserver
import webbrowser
from typing import Dict, Optional, Any
import torch
import torch.fx as fx


class GraphVisualizer:
    """Visualize FX GraphModule with Plotly"""

    def __init__(self, gm: fx.GraphModule, input_data: Optional[Dict[str, torch.Tensor]] = None):
        self.gm = gm
        self.input_data = input_data
        self.execution_trace = []

        if input_data:
            self._trace_execution()

        self.graph_data = self._extract_graph_data()

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

                # Histogram data
                flat = value.flatten().detach().cpu().numpy()
                trace_info["histogram"] = flat[:min(10000, len(flat))].tolist()

                # Heatmap data (2D slice)
                if value.dim() >= 2:
                    if value.dim() == 2:
                        hm = value
                    elif value.dim() == 3:
                        hm = value[0]  # First batch/channel
                    else:
                        hm = value.reshape(-1, value.shape[-1])

                    # Limit size for performance
                    h, w = hm.shape
                    if h > 64:
                        hm = hm[::h // 64]
                    if w > 64:
                        hm = hm[:, ::w // 64]
                    trace_info["heatmap"] = hm.detach().cpu().numpy().tolist()

                # Full data for dimension browser (store complete tensor)
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
            }

            if node.name in trace_lookup:
                trace = trace_lookup[node.name]
                for key in ["shape", "dtype", "min", "max", "mean", "std", "histogram", "heatmap", "full_data"]:
                    if key in trace:
                        node_info[key] = trace[key]

            # Node type
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

    def to_html(self) -> str:
        """Generate HTML with Plotly visualization"""
        graph_json = json.dumps(self.graph_data)

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
            grid-template-columns: 1fr 400px;
            grid-template-rows: 56px 1fr;
            height: 100vh;
        }}
        header {{
            grid-column: 1 / -1;
            background: rgba(22, 33, 62, 0.95);
            backdrop-filter: blur(10px);
            display: flex;
            align-items: center;
            padding: 0 24px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            gap: 24px;
        }}
        header h1 {{
            font-size: 18px;
            font-weight: 600;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .controls {{
            display: flex;
            gap: 8px;
            margin-left: auto;
        }}
        .btn {{
            padding: 8px 16px;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            color: #e2e8f0;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.2s;
        }}
        .btn:hover {{ background: rgba(255,255,255,0.15); }}
        .btn.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-color: transparent;
        }}
        #graph-container {{
            position: relative;
            overflow: hidden;
        }}
        #graph-svg {{
            width: 100%;
            height: 100%;
        }}
        #sidebar {{
            background: rgba(22, 33, 62, 0.95);
            backdrop-filter: blur(10px);
            border-left: 1px solid rgba(255,255,255,0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        .sidebar-header {{
            padding: 16px 20px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        #search {{
            width: 100%;
            padding: 10px 14px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            color: #e2e8f0;
            font-size: 14px;
            outline: none;
            transition: all 0.2s;
        }}
        #search:focus {{
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102,126,234,0.2);
        }}
        .sidebar-content {{
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }}
        .section {{
            margin-bottom: 24px;
        }}
        .section-title {{
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #94a3b8;
            margin-bottom: 12px;
        }}
        .info-grid {{
            display: grid;
            gap: 8px;
        }}
        .info-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            background: rgba(255,255,255,0.03);
            border-radius: 6px;
        }}
        .info-label {{ color: #94a3b8; font-size: 13px; }}
        .info-value {{ color: #e2e8f0; font-family: 'SF Mono', monospace; font-size: 13px; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
            border: 1px solid rgba(102,126,234,0.2);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 20px;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .stat-label {{
            font-size: 11px;
            color: #94a3b8;
            text-transform: uppercase;
            margin-top: 4px;
        }}
        #histogram-plot, #heatmap-plot {{
            width: 100%;
            height: 180px;
            border-radius: 12px;
            overflow: hidden;
        }}
        #data-table-container {{
            max-height: 300px;
            overflow: auto;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            padding: 8px;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-family: 'SF Mono', monospace;
            font-size: 11px;
        }}
        .data-table td {{
            padding: 4px 6px;
            border: 1px solid rgba(255,255,255,0.1);
            text-align: right;
            color: #a5b4fc;
        }}
        .data-table tr:nth-child(even) {{
            background: rgba(255,255,255,0.03);
        }}
        .data-table .positive {{ color: #4ade80; }}
        .data-table .negative {{ color: #f87171; }}
        .data-table .zero {{ color: #94a3b8; }}
        #dim-selector {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 12px;
        }}
        .dim-control {{
            display: flex;
            align-items: center;
            gap: 6px;
            background: rgba(255,255,255,0.05);
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 12px;
        }}
        .dim-control label {{
            color: #94a3b8;
        }}
        .dim-control select, .dim-control input {{
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 4px;
            color: #e2e8f0;
            padding: 4px 8px;
            font-size: 12px;
            width: 70px;
        }}
        .dim-control input[type="range"] {{
            width: 80px;
        }}
        .dim-value {{
            color: #a5b4fc;
            font-family: monospace;
            min-width: 30px;
        }}
        .placeholder {{
            display: flex;
            align-items: center;
            justify-content: center;
            height: 200px;
            color: #64748b;
            font-size: 14px;
        }}
        /* Graph styles */
        .node {{ cursor: pointer; }}
        .node rect {{
            rx: 10; ry: 10;
            stroke-width: 2px;
            filter: drop-shadow(0 4px 6px rgba(0,0,0,0.3));
            transition: all 0.2s;
        }}
        .node:hover rect {{
            filter: drop-shadow(0 8px 25px rgba(102,126,234,0.4));
            transform: scale(1.02);
        }}
        .node.selected rect {{
            stroke: #667eea !important;
            stroke-width: 3px;
            filter: drop-shadow(0 0 20px rgba(102,126,234,0.6));
        }}
        .node text {{ fill: white; pointer-events: none; }}
        .node-name {{ font-size: 12px; font-weight: 600; }}
        .node-shape {{ font-size: 10px; opacity: 0.7; }}
        .link {{
            fill: none;
            stroke: rgba(255,255,255,0.2);
            stroke-width: 2px;
        }}
        .link.highlighted {{
            stroke: #667eea;
            stroke-width: 3px;
        }}
        .link.flow {{
            stroke: #10b981;
            stroke-width: 3px;
            stroke-dasharray: 10,5;
            animation: dash 0.5s linear infinite;
        }}
        @keyframes dash {{ to {{ stroke-dashoffset: -15; }} }}
        /* Node colors */
        .node-input rect {{ fill: linear-gradient(135deg, #10b981 0%, #059669 100%); fill: #10b981; stroke: #059669; }}
        .node-output rect {{ fill: #ef4444; stroke: #dc2626; }}
        .node-matmul rect {{ fill: #3b82f6; stroke: #2563eb; }}
        .node-activation rect {{ fill: #8b5cf6; stroke: #7c3aed; }}
        .node-elementwise rect {{ fill: #f59e0b; stroke: #d97706; }}
        /* Small weight nodes */
        .node-small rect {{ rx: 6; ry: 6; opacity: 0.85; }}
        .node-small text {{ font-size: 10px; }}
        .node-norm rect {{ fill: #14b8a6; stroke: #0d9488; }}
        .node-reshape rect {{ fill: #f97316; stroke: #ea580c; }}
        .node-weight rect {{ fill: #6b7280; stroke: #4b5563; }}
        .node-function rect {{ fill: #374151; stroke: #4b5563; }}
        .node-other rect {{ fill: #374151; stroke: #4b5563; }}
        .order-badge {{
            fill: #fbbf24;
            font-size: 10px;
            font-weight: 700;
        }}
        .legend {{
            padding: 16px 20px;
            border-top: 1px solid rgba(255,255,255,0.1);
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            color: #94a3b8;
        }}
        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div id="app">
        <header>
            <h1>ONNX Parser Visualizer</h1>
            <div class="controls">
                <button class="btn" onclick="resetView()">Reset View</button>
                <button class="btn" id="playBtn" onclick="togglePlay()">▶ Play</button>
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
                    <div class="section" id="data-section" style="display:none;">
                        <div class="section-title">Tensor Data Browser</div>
                        <div id="dim-selector"></div>
                        <div id="data-table-container"></div>
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
    </div>

    <script>
        const graphData = {graph_json};
        const container = document.getElementById('graph-container');
        const svg = d3.select('#graph-svg');
        const g = svg.append('g');

        // Zoom
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', e => g.attr('transform', e.transform));
        svg.call(zoom);

        // Arrow marker
        svg.append('defs').append('marker')
            .attr('id', 'arrow')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 28)
            .attr('refY', 0)
            .attr('orient', 'auto')
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', 'rgba(255,255,255,0.3)');

        // Layout - improved to handle weights better
        const nodeW = 140, nodeH = 44, layerGap = 180, nodeGap = 16;
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
                if (node.type === 'weight') return -1;  // weights handled separately
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
        const startX = Math.max(80, (width - totalWidth) / 2);

        Object.keys(layerNodes).forEach(l => {{
            const nodes = layerNodes[l];
            const total = nodes.length * (nodeH + nodeGap) - nodeGap;
            const startY = (height - total) / 2;
            nodes.forEach((n, idx) => {{
                n.x = startX + parseInt(l) * (nodeW + layerGap);
                n.y = startY + idx * (nodeH + nodeGap);
            }});
        }});

        // Position weight nodes next to their consumers (smaller, offset to top-left)
        const weightW = 100, weightH = 32;
        weightNodes.forEach(n => {{
            n.isWeight = true;
            n.nodeW = weightW;
            n.nodeH = weightH;

            // Find the first consumer of this weight
            const consumers = outgoing[n.id].map(id => nodeById[id]).filter(c => c.x !== undefined);
            if (consumers.length > 0) {{
                const consumer = consumers[0];
                // Count how many weights feed into this consumer
                const siblingWeights = incoming[consumer.id]
                    .map(id => nodeById[id])
                    .filter(sib => sib.type === 'weight');
                const sibIdx = siblingWeights.indexOf(n);

                // Stack weights vertically above/beside the consumer
                n.x = consumer.x - weightW - 30;
                n.y = consumer.y - 20 + sibIdx * (weightH + 8);
            }} else {{
                // Fallback: place at start
                n.x = 20;
                n.y = 50 + weightNodes.indexOf(n) * (weightH + 8);
            }}
        }});

        // Apply default dimensions to compute nodes
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
            .attr('transform', d => `translate(${{d.x}},${{d.y}})`)
            .on('click', selectNode);

        nodes.append('rect')
            .attr('width', d => d.nodeW || nodeW)
            .attr('height', d => d.nodeH || nodeH);
        nodes.append('text')
            .attr('class', 'node-name')
            .attr('x', d => (d.nodeW || nodeW)/2)
            .attr('y', d => d.isWeight ? 14 : 18)
            .attr('text-anchor', 'middle')
            .style('font-size', d => d.isWeight ? '10px' : '12px')
            .text(d => {{
                const maxLen = d.isWeight ? 12 : 16;
                return d.name.length > maxLen ? d.name.slice(0, maxLen-2)+'..' : d.name;
            }});
        nodes.filter(d => !d.isWeight).append('text')
            .attr('class', 'node-shape')
            .attr('x', d => (d.nodeW || nodeW)/2)
            .attr('y', 34)
            .attr('text-anchor', 'middle')
            .text(d => d.shape ? JSON.stringify(d.shape) : '');
        nodes.filter(d => d.isWeight).append('text')
            .attr('class', 'node-shape')
            .attr('x', d => (d.nodeW || nodeW)/2)
            .attr('y', 26)
            .attr('text-anchor', 'middle')
            .style('font-size', '9px')
            .text(d => d.shape ? JSON.stringify(d.shape) : '');

        // Order badges (only for non-weight nodes)
        const badgeNodes = nodes.filter(d => !d.isWeight);
        badgeNodes.append('circle').attr('cx', -6).attr('cy', -6).attr('r', 10).attr('fill', '#fbbf24');
        badgeNodes.append('text')
            .attr('class', 'order-badge')
            .attr('x', -6).attr('y', -2)
            .attr('text-anchor', 'middle')
            .style('font-size', '10px')
            .text((d,i) => i+1);

        // Node selection
        function selectNode(e, d) {{
            d3.selectAll('.node').classed('selected', false);
            d3.selectAll('.link').classed('highlighted', false);
            d3.select(this).classed('selected', true);
            links.classed('highlighted', l => l.source === d.id || l.target === d.id);

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

                // Histogram with Plotly
                if (d.histogram) {{
                    histSection.style.display = 'block';
                    Plotly.newPlot('histogram-plot', [{{
                        x: d.histogram,
                        type: 'histogram',
                        marker: {{
                            color: 'rgba(102, 126, 234, 0.7)',
                            line: {{ color: 'rgba(102, 126, 234, 1)', width: 1 }}
                        }},
                        nbinsx: 30
                    }}], {{
                        paper_bgcolor: 'transparent',
                        plot_bgcolor: 'transparent',
                        margin: {{ t: 10, r: 10, b: 30, l: 40 }},
                        xaxis: {{ color: '#94a3b8', gridcolor: 'rgba(255,255,255,0.1)' }},
                        yaxis: {{ color: '#94a3b8', gridcolor: 'rgba(255,255,255,0.1)' }},
                        bargap: 0.05
                    }}, {{ responsive: true, displayModeBar: false }});
                }} else {{
                    histSection.style.display = 'none';
                }}

                // Heatmap with Plotly
                if (d.heatmap) {{
                    heatSection.style.display = 'block';
                    Plotly.newPlot('heatmap-plot', [{{
                        z: d.heatmap,
                        type: 'heatmap',
                        colorscale: [
                            [0, '#3b0764'],
                            [0.25, '#7c3aed'],
                            [0.5, '#a78bfa'],
                            [0.75, '#c4b5fd'],
                            [1, '#f5f3ff']
                        ],
                        showscale: true,
                        colorbar: {{
                            thickness: 15,
                            tickfont: {{ color: '#94a3b8' }}
                        }}
                    }}], {{
                        paper_bgcolor: 'transparent',
                        plot_bgcolor: 'transparent',
                        margin: {{ t: 10, r: 60, b: 30, l: 40 }},
                        xaxis: {{ color: '#94a3b8', showgrid: false }},
                        yaxis: {{ color: '#94a3b8', showgrid: false, autorange: 'reversed' }}
                    }}, {{ responsive: true, displayModeBar: false }});
                }} else {{
                    heatSection.style.display = 'none';
                }}

                // Dimension browser
                const dataSection = document.getElementById('data-section');
                if (d.full_data && d.shape) {{
                    dataSection.style.display = 'block';
                    currentTensorData = d.full_data;
                    currentTensorShape = d.shape;
                    currentDimIndices = d.shape.map(() => 0);
                    renderDimSelector(d.shape);
                    updateDataTable();
                }} else {{
                    dataSection.style.display = 'none';
                }}
            }} else {{
                statsSection.style.display = 'none';
                histSection.style.display = 'none';
                heatSection.style.display = 'none';
                document.getElementById('data-section').style.display = 'none';
            }}
        }}

        // Tensor data browser state
        let currentTensorData = null;
        let currentTensorShape = [];
        let currentDimIndices = [];

        function renderDimSelector(shape) {{
            const container = document.getElementById('dim-selector');
            if (shape.length <= 2) {{
                container.innerHTML = '<span style="color:#94a3b8;font-size:12px;">Showing full 2D slice</span>';
                return;
            }}

            let html = '';
            // For dims > 2, we need sliders for all but last 2 dims
            for (let i = 0; i < shape.length - 2; i++) {{
                html += `
                    <div class="dim-control">
                        <label>Dim ${{i}}</label>
                        <input type="range" min="0" max="${{shape[i] - 1}}" value="0"
                               onchange="updateDimIndex(${{i}}, this.value)">
                        <span class="dim-value" id="dim-val-${{i}}">0</span>
                        <span style="color:#64748b;">/ ${{shape[i] - 1}}</span>
                    </div>
                `;
            }}
            container.innerHTML = html;
        }}

        function updateDimIndex(dimIdx, value) {{
            currentDimIndices[dimIdx] = parseInt(value);
            document.getElementById(`dim-val-${{dimIdx}}`).textContent = value;
            updateDataTable();
        }}

        function getSlice(data, indices) {{
            let slice = data;
            for (let i = 0; i < indices.length - 2 && i < indices.length; i++) {{
                if (Array.isArray(slice)) {{
                    slice = slice[indices[i]] || slice[0];
                }}
            }}
            return slice;
        }}

        function updateDataTable() {{
            const container = document.getElementById('data-table-container');
            if (!currentTensorData) return;

            // Get 2D slice based on current indices
            let slice = getSlice(currentTensorData, currentDimIndices);

            // Render as table
            let html = '<table class="data-table">';

            if (Array.isArray(slice) && Array.isArray(slice[0])) {{
                // 2D data
                for (let i = 0; i < slice.length; i++) {{
                    html += '<tr>';
                    const row = slice[i];
                    for (let j = 0; j < row.length; j++) {{
                        const val = parseFloat(row[j]);
                        const cls = val > 0.001 ? 'positive' : val < -0.001 ? 'negative' : 'zero';
                        html += `<td class="${{cls}}">${{val.toFixed(4)}}</td>`;
                    }}
                    html += '</tr>';
                }}
            }} else if (Array.isArray(slice)) {{
                // 1D data
                html += '<tr>';
                for (let i = 0; i < slice.length; i++) {{
                    const val = parseFloat(slice[i]);
                    const cls = val > 0.001 ? 'positive' : val < -0.001 ? 'negative' : 'zero';
                    html += `<td class="${{cls}}">${{val.toFixed(4)}}</td>`;
                }}
                html += '</tr>';
            }} else {{
                // Scalar
                const val = parseFloat(slice);
                const cls = val > 0.001 ? 'positive' : val < -0.001 ? 'negative' : 'zero';
                html += `<tr><td class="${{cls}}">${{val.toFixed(4)}}</td></tr>`;
            }}

            html += '</table>';
            container.innerHTML = html;
        }}

        // Search
        document.getElementById('search').addEventListener('input', e => {{
            const q = e.target.value.toLowerCase();
            nodes.style('opacity', d => !q || d.name.toLowerCase().includes(q) ? 1 : 0.15);
        }});

        // Reset view
        function resetView() {{
            const bounds = g.node().getBBox();
            const scale = Math.min(width/(bounds.width+100), height/(bounds.height+100)) * 0.85;
            const tx = (width - bounds.width*scale)/2 - bounds.x*scale;
            const ty = (height - bounds.height*scale)/2 - bounds.y*scale;
            svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
        }}

        // Animation
        let playing = false, step = 0;
        function togglePlay() {{
            playing = !playing;
            document.getElementById('playBtn').textContent = playing ? '⏸ Pause' : '▶ Play';
            document.getElementById('playBtn').classList.toggle('active', playing);
            if (playing) {{ step = 0; runStep(); }}
        }}
        function runStep() {{
            if (!playing) return;
            nodes.style('opacity', (d,i) => i <= step ? 1 : 0.2);
            links.style('opacity', e => {{
                const ti = graphData.nodes.findIndex(n => n.id === e.target);
                return ti <= step ? 1 : 0.1;
            }});
            if (step < graphData.nodes.length - 1) {{
                step++;
                setTimeout(runStep, 350);
            }} else {{
                playing = false;
                document.getElementById('playBtn').textContent = '▶ Play';
                document.getElementById('playBtn').classList.remove('active');
                setTimeout(() => {{ nodes.style('opacity', 1); links.style('opacity', 1); }}, 400);
            }}
        }}

        // Data flow
        let flowActive = false;
        function toggleFlow() {{
            flowActive = !flowActive;
            document.getElementById('flowBtn').classList.toggle('active', flowActive);
            links.classed('flow', flowActive);
        }}

        resetView();
    </script>
</body>
</html>'''
        return html

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_html())
        print(f"Saved: {path}")

    def serve(self, port: int = 8080, open_browser: bool = True) -> None:
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
