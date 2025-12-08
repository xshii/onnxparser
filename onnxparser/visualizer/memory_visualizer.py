# -*- coding: utf-8 -*-
"""Interactive memory analysis visualization"""

import json
import http.server
import socketserver
import webbrowser
from typing import Dict, Optional, List, Any
import torch.fx as fx

from ..analysis.memory_analyzer import MemoryAnalyzer, AnalysisResult
from ..analysis.spill_scheduler import SpillScheduler, SpillStrategy, MemoryEventType


class MemoryVisualizer:
    """Interactive memory analysis visualizer"""

    def __init__(
        self,
        gm: fx.GraphModule,
        dynamic_shapes: Optional[Dict[str, int]] = None,
    ):
        self.gm = gm
        self.dynamic_shapes = dynamic_shapes or {}

        # Run analysis with all strategies
        self._run_analysis()

    def _run_analysis(self):
        """Run memory analysis with multiple strategies"""
        from ..analysis.strategies import StrategyRegistry

        self.results: Dict[str, AnalysisResult] = {}
        strategies = StrategyRegistry.list_strategies()

        for strategy in strategies:
            analyzer = MemoryAnalyzer(self.gm, strategy=strategy)
            self.results[strategy] = analyzer.analyze(self.dynamic_shapes)

    def _prepare_data(self) -> Dict[str, Any]:
        """Prepare data for visualization"""
        # Get node info
        nodes = []
        for idx, node in enumerate(self.gm.graph.nodes):
            if node.op == "output":
                continue
            nodes.append({
                "id": idx,
                "name": node.name,
                "op": node.op,
                "target": str(node.target) if node.target else "",
            })

        # Prepare per-strategy data
        strategies_data = {}
        for strategy_name, result in self.results.items():
            steps_data = []
            for step in result.steps:
                steps_data.append({
                    "step": step.step,
                    "node": step.node_name,
                    "op": step.op_type,
                    "max_memory": step.max_memory,
                    "min_memory": step.min_memory,
                    "live_memory": step.live_memory,
                    "live_tensors": step.live_tensors,
                    "freed": step.freed_tensors,
                    "reused_from": step.reused_from,
                    "input_bytes": step.input_total_bytes,
                    "output_bytes": step.output_bytes,
                })

            # Tensor lifetime data
            tensors_data = []
            for name, tensor in result.tensors.items():
                tensors_data.append({
                    "name": name,
                    "shape": tensor.shape,
                    "dtype": str(tensor.dtype),
                    "size_bytes": tensor.size_bytes,
                    "size_mb": tensor.size_bytes / 1e6,
                    "birth": tensor.birth_step,
                    "death": tensor.death_step,
                    "is_weight": tensor.is_weight,
                    "is_input": tensor.is_input,
                    "is_output": tensor.is_output,
                    "reused_from": tensor.reused_from,
                    "is_inplace": tensor.is_inplace,
                    "memory_offset": tensor.memory_offset,
                })

            # Build memory blocks timeline for memory map visualization
            memory_blocks = self._build_memory_blocks(result)

            strategies_data[strategy_name] = {
                "steps": steps_data,
                "tensors": tensors_data,
                "memory_blocks": memory_blocks,
                "summary": {
                    "peak_max_mb": result.peak_max_memory / 1e6,
                    "peak_min_mb": result.peak_min_memory / 1e6,
                    "static_mb": result.static_memory / 1e6,
                    "savings_pct": result.savings_ratio * 100,
                    "peak_step": result.peak_step_min,
                },
            }

        return {
            "nodes": nodes,
            "strategies": strategies_data,
            "dynamic_shapes": self.dynamic_shapes,
            "spill_schedules": self._build_spill_schedules(),
            "static_layout": self._build_static_layout(),
        }

    def _build_static_layout(self) -> Dict[str, Any]:
        """Build static allocation layout data"""
        # Get static strategy result if available
        if "static" not in self.results:
            return {"available": False}

        result = self.results["static"]

        # Build layout data
        layout_data = []
        for name, tensor in result.tensors.items():
            if tensor.is_weight:
                continue
            layout_data.append({
                "name": name,
                "offset": tensor.memory_offset,
                "size": tensor.size_bytes,
                "birth": tensor.birth_step,
                "death": tensor.death_step,
                "shape": tensor.shape,
            })

        # Sort by offset
        layout_data.sort(key=lambda x: x["offset"])

        # Calculate total memory
        total_memory = max(
            (t["offset"] + t["size"]) for t in layout_data
        ) if layout_data else 0

        # Generate C code
        c_code_lines = [
            "// Static Memory Layout",
            f"// Total Memory Required: {total_memory} bytes",
            "",
            f"#define MEMORY_POOL_SIZE {total_memory}",
            "",
            "// Tensor Offsets",
        ]
        for t in layout_data:
            safe_name = t["name"].upper().replace(".", "_").replace("-", "_")
            c_code_lines.append(
                f"#define OFFSET_{safe_name} {t['offset']}  // size: {t['size']}"
            )

        # Find memory reuse info
        offset_groups = {}
        for t in layout_data:
            off = t["offset"]
            if off not in offset_groups:
                offset_groups[off] = []
            offset_groups[off].append(t["name"])

        reuse_info = [
            {"offset": off, "tensors": names}
            for off, names in offset_groups.items()
            if len(names) > 1
        ]

        return {
            "available": True,
            "layout": layout_data,
            "total_memory": total_memory,
            "total_memory_kb": total_memory / 1024,
            "c_code": "\n".join(c_code_lines),
            "reuse_info": reuse_info,
            "tensor_count": len(layout_data),
        }

    def _build_spill_schedules(self) -> Dict[str, Any]:
        """Build spill schedule data for different memory limits"""
        # Get baseline peak memory
        baseline = self.results.get("greedy", list(self.results.values())[0])
        peak_memory = baseline.peak_min_memory
        peak_kb = peak_memory / 1024

        # Generate schedules for a few default absolute values
        schedules = {}

        # Default values based on peak memory (round numbers)
        default_limits_kb = []
        # Add some reasonable defaults around peak memory
        for kb in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
            if kb >= peak_kb * 0.2 and kb <= peak_kb * 1.2:
                default_limits_kb.append(kb)

        # Always include peak memory rounded
        peak_rounded = int(round(peak_kb / 10) * 10)
        if peak_rounded > 0 and peak_rounded not in default_limits_kb:
            default_limits_kb.append(peak_rounded)

        default_limits_kb.sort()

        for limit_kb in default_limits_kb:
            limit_label = f"{limit_kb}KB"
            self._add_schedule(schedules, limit_label, int(limit_kb * 1024))

        return {
            "peak_memory_kb": peak_kb,
            "schedules": schedules,
        }

    def _add_schedule(self, schedules: Dict, label: str, limit_bytes: int) -> None:
        """Add a spill schedule for a given memory limit"""
        try:
            scheduler = SpillScheduler(
                self.gm,
                memory_limit_bytes=limit_bytes,
                spill_strategy=SpillStrategy.COST_BENEFIT,
            )
            result = scheduler.schedule()

            # Build events timeline
            events = []
            for e in result.events:
                if e.event_type in [MemoryEventType.SPILL, MemoryEventType.RELOAD]:
                    events.append({
                        "step": e.step,
                        "type": e.event_type.value,
                        "tensor": e.tensor_name,
                        "size_kb": e.size_bytes / 1024,
                        "node": e.node_name,
                        "fast_kb": e.fast_memory_used / 1024,
                        "slow_kb": e.slow_memory_used / 1024,
                    })

            schedules[label] = {
                "limit_kb": limit_bytes / 1024,
                "total_spills": result.total_spills,
                "total_reloads": result.total_reloads,
                "spill_bytes_kb": result.total_spill_bytes / 1024,
                "peak_fast_kb": result.peak_fast_memory / 1024,
                "peak_slow_kb": result.peak_slow_memory / 1024,
                "spill_trigger_nodes": result.spill_trigger_nodes,
                "reload_trigger_nodes": result.reload_trigger_nodes,
                "events": events,
                "fast_timeline": [m / 1024 for m in result.fast_memory_timeline],
                "slow_timeline": [m / 1024 for m in result.slow_memory_timeline],
                "spill_decisions": [
                    {
                        "tensor": d.tensor_name,
                        "size_kb": d.size_bytes / 1024,
                        "spill_step": d.spill_step,
                        "reload_step": d.reload_step,
                        "duration": d.spill_duration,
                    }
                    for d in result.spill_decisions
                ],
            }
        except Exception:
            schedules[label] = {"error": True}

    def _build_memory_blocks(self, result: AnalysisResult) -> List[Dict]:
        """Build memory block timeline for visualization"""
        blocks = []

        # Filter non-weight tensors with valid memory offsets
        tensors = [
            t for t in result.tensors.values()
            if not t.is_weight and t.memory_offset >= 0
        ]

        # Sort by memory offset for consistent visualization
        tensors.sort(key=lambda t: (t.memory_offset, t.birth_step))

        # Assign colors based on tensor type
        color_idx = 0
        colors = [
            '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6',
            '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
            '#14b8a6', '#a855f7', '#eab308', '#22c55e', '#0ea5e9',
        ]

        for tensor in tensors:
            # Calculate end offset
            end_offset = tensor.memory_offset + tensor.size_bytes

            block = {
                "name": tensor.name,
                "offset": tensor.memory_offset,
                "size": tensor.size_bytes,
                "end_offset": end_offset,
                "birth": tensor.birth_step,
                "death": tensor.death_step,
                "is_input": tensor.is_input,
                "is_output": tensor.is_output,
                "reused_from": tensor.reused_from,
                "is_inplace": tensor.is_inplace,
                "color": colors[color_idx % len(colors)],
                "shape": tensor.shape,
                "dtype": str(tensor.dtype),
            }
            blocks.append(block)
            color_idx += 1

        return blocks

    def to_html(self) -> str:
        """Generate interactive HTML visualization"""
        data = self._prepare_data()
        data_json = json.dumps(data)

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ONNX Parser - Memory Analyzer</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
            color: #e2e8f0;
            min-height: 100vh;
        }}
        #app {{
            display: grid;
            grid-template-rows: 56px 1fr;
            min-height: 100vh;
        }}
        header {{
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
            background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .main-content {{
            display: grid;
            grid-template-columns: 320px 1fr;
            overflow: hidden;
        }}
        .sidebar {{
            background: rgba(22, 33, 62, 0.95);
            border-right: 1px solid rgba(255,255,255,0.1);
            padding: 20px;
            overflow-y: auto;
        }}
        .content {{
            padding: 20px;
            overflow-y: auto;
        }}
        .section {{
            margin-bottom: 24px;
        }}
        .section-title {{
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #f59e0b;
            margin-bottom: 12px;
        }}
        .card {{
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
        }}
        .card-title {{
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 12px;
            color: #94a3b8;
        }}
        .strategy-btn {{
            display: block;
            width: 100%;
            padding: 12px 16px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            color: #e2e8f0;
            cursor: pointer;
            text-align: left;
            margin-bottom: 8px;
            transition: all 0.2s;
        }}
        .strategy-btn:hover {{
            background: rgba(255,255,255,0.1);
        }}
        .strategy-btn.active {{
            background: linear-gradient(135deg, rgba(245,158,11,0.2) 0%, rgba(239,68,68,0.2) 100%);
            border-color: #f59e0b;
        }}
        .strategy-btn .name {{
            font-weight: 600;
            font-size: 14px;
        }}
        .strategy-btn .stats {{
            font-size: 12px;
            color: #94a3b8;
            margin-top: 4px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }}
        .summary-card {{
            background: linear-gradient(135deg, rgba(245,158,11,0.1) 0%, rgba(239,68,68,0.1) 100%);
            border: 1px solid rgba(245,158,11,0.3);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
        }}
        .summary-value {{
            font-size: 24px;
            font-weight: 700;
            background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .summary-label {{
            font-size: 11px;
            color: #94a3b8;
            text-transform: uppercase;
            margin-top: 4px;
        }}
        .chart-container {{
            background: rgba(0,0,0,0.2);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 20px;
        }}
        .chart-title {{
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 12px;
            color: #e2e8f0;
        }}
        #memory-chart {{
            width: 100%;
            height: 350px;
        }}
        #lifetime-chart {{
            width: 100%;
            height: 300px;
        }}
        #comparison-chart {{
            width: 100%;
            height: 250px;
        }}
        .tensor-list {{
            max-height: 400px;
            overflow-y: auto;
        }}
        .tensor-item {{
            padding: 10px 12px;
            background: rgba(255,255,255,0.03);
            border-radius: 6px;
            margin-bottom: 6px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .tensor-item:hover {{
            background: rgba(255,255,255,0.08);
        }}
        .tensor-item.weight {{
            border-left: 3px solid #6b7280;
        }}
        .tensor-item.input {{
            border-left: 3px solid #10b981;
        }}
        .tensor-item.reused {{
            border-left: 3px solid #3b82f6;
        }}
        .tensor-item.inplace {{
            border-left: 3px solid #8b5cf6;
        }}
        .tensor-name {{
            font-weight: 600;
            color: #e2e8f0;
        }}
        .tensor-info {{
            color: #94a3b8;
            margin-top: 2px;
        }}
        .constraint-input {{
            display: flex;
            gap: 8px;
            align-items: center;
            margin-bottom: 12px;
        }}
        .constraint-input label {{
            font-size: 13px;
            color: #94a3b8;
        }}
        .constraint-input input {{
            flex: 1;
            padding: 8px 12px;
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 6px;
            color: #e2e8f0;
            font-size: 13px;
        }}
        .constraint-input input:focus {{
            outline: none;
            border-color: #f59e0b;
        }}
        .btn {{
            padding: 10px 20px;
            background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);
            border: none;
            border-radius: 8px;
            color: white;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .btn:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(245,158,11,0.3);
        }}
        .tabs {{
            display: flex;
            gap: 4px;
            margin-bottom: 16px;
        }}
        .tab {{
            padding: 8px 16px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 6px;
            color: #94a3b8;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }}
        .tab:hover {{
            background: rgba(255,255,255,0.1);
        }}
        .tab.active {{
            background: rgba(245,158,11,0.2);
            border-color: #f59e0b;
            color: #f59e0b;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
        .legend {{
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            margin-top: 12px;
            padding: 12px;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            color: #94a3b8;
        }}
        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 3px;
        }}
        .step-details {{
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            padding: 16px;
            margin-top: 16px;
            display: none;
        }}
        .step-details.visible {{
            display: block;
        }}
        .step-details h4 {{
            font-size: 14px;
            margin-bottom: 12px;
            color: #f59e0b;
        }}
        .step-details .info-row {{
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            font-size: 13px;
        }}
        .step-details .info-label {{
            color: #94a3b8;
        }}
        .step-details .info-value {{
            color: #e2e8f0;
            font-family: monospace;
        }}
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        .comparison-table th, .comparison-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .comparison-table th {{
            color: #94a3b8;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 11px;
        }}
        .comparison-table td {{
            color: #e2e8f0;
        }}
        .comparison-table tr:hover {{
            background: rgba(255,255,255,0.03);
        }}
        .best {{
            color: #10b981 !important;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div id="app">
        <header>
            <h1>Memory Analyzer</h1>
            <span style="color:#94a3b8;font-size:13px;">Interactive memory analysis for ONNX models</span>
        </header>
        <div class="main-content">
            <div class="sidebar">
                <div class="section">
                    <div class="section-title">Allocation Strategy</div>
                    <div id="strategy-buttons"></div>
                </div>

                <div class="section">
                    <div class="section-title">Summary</div>
                    <div class="summary-grid" id="summary-grid"></div>
                </div>

                <div class="section">
                    <div class="section-title">Memory Constraint</div>
                    <div class="constraint-input">
                        <label>Max Memory:</label>
                        <input type="number" id="memory-limit" placeholder="MB">
                    </div>
                    <div style="display:flex;gap:8px;">
                        <label style="font-size:12px;color:#94a3b8;display:flex;align-items:center;gap:4px;">
                            <input type="checkbox" id="show-constraint"> Show limit line
                        </label>
                    </div>
                </div>
            </div>

            <div class="content">
                <div class="tabs">
                    <div class="tab active" data-tab="memory">Memory Timeline</div>
                    <div class="tab" data-tab="memorymap">Memory Map</div>
                    <div class="tab" data-tab="static">Static Layout</div>
                    <div class="tab" data-tab="spill">Spill Scheduler</div>
                    <div class="tab" data-tab="comparison">Strategy Comparison</div>
                    <div class="tab" data-tab="tensors">Tensor Lifetimes</div>
                </div>

                <div id="tab-memory" class="tab-content active">
                    <div class="chart-container">
                        <div class="chart-title">Memory Usage Over Execution Steps</div>
                        <div id="memory-chart"></div>
                        <div class="legend">
                            <div class="legend-item"><div class="legend-dot" style="background:#ef4444"></div>Max Memory (No Reuse)</div>
                            <div class="legend-item"><div class="legend-dot" style="background:#10b981"></div>Actual Memory (With Reuse)</div>
                            <div class="legend-item"><div class="legend-dot" style="background:#3b82f6"></div>Live Tensors</div>
                        </div>
                    </div>

                    <div class="step-details" id="step-details">
                        <h4>Step Details</h4>
                        <div id="step-info"></div>
                    </div>
                </div>

                <div id="tab-memorymap" class="tab-content">
                    <div class="chart-container">
                        <div class="chart-title">Memory Block Allocation Map</div>
                        <div id="memorymap-chart" style="height:500px;"></div>
                        <div class="legend">
                            <div class="legend-item"><div class="legend-dot" style="background:#10b981"></div>Input</div>
                            <div class="legend-item"><div class="legend-dot" style="background:#3b82f6"></div>Allocated</div>
                            <div class="legend-item"><div class="legend-dot" style="background:#8b5cf6"></div>Reused</div>
                            <div class="legend-item"><div class="legend-dot" style="background:#f59e0b"></div>Output</div>
                        </div>
                        <div style="color:#94a3b8;font-size:12px;margin-top:8px;">
                            X-axis: Execution Step | Y-axis: Memory Address Space (bytes) | Click on a block for details
                        </div>
                    </div>
                    <div class="step-details" id="block-details">
                        <h4>Memory Block Details</h4>
                        <div id="block-info"></div>
                    </div>
                </div>

                <div id="tab-static" class="tab-content">
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;">
                        <div class="card">
                            <div class="card-title">Static Allocation Summary</div>
                            <div id="static-summary"></div>
                        </div>
                        <div class="card">
                            <div class="card-title">Memory Reuse</div>
                            <div id="static-reuse"></div>
                        </div>
                    </div>

                    <div class="chart-container">
                        <div class="chart-title">Fixed Memory Address Layout</div>
                        <div id="static-layout-chart" style="height:400px;"></div>
                        <div class="legend">
                            <div class="legend-item"><div class="legend-dot" style="background:#3b82f6"></div>Memory Block</div>
                            <div class="legend-item"><div class="legend-dot" style="background:#10b981"></div>Reused Address</div>
                        </div>
                    </div>

                    <div class="card" style="margin-top:16px;">
                        <div class="card-title">Generated C Code</div>
                        <div style="display:flex;justify-content:flex-end;margin-bottom:8px;">
                            <button class="btn" onclick="copyStaticCode()" style="padding:6px 12px;font-size:12px;">Copy Code</button>
                        </div>
                        <pre id="static-c-code" style="background:rgba(0,0,0,0.4);padding:16px;border-radius:8px;overflow-x:auto;font-family:monospace;font-size:13px;line-height:1.5;color:#a5d6ff;max-height:300px;overflow-y:auto;"></pre>
                    </div>

                    <div class="card" style="margin-top:16px;">
                        <div class="card-title">Address Table</div>
                        <table id="static-table">
                            <thead>
                                <tr>
                                    <th>Tensor</th>
                                    <th>Offset (hex)</th>
                                    <th>Offset (dec)</th>
                                    <th>Size</th>
                                    <th>Lifetime</th>
                                    <th>Shape</th>
                                </tr>
                            </thead>
                            <tbody></tbody>
                        </table>
                    </div>
                </div>

                <div id="tab-spill" class="tab-content">
                    <div class="card" style="margin-bottom:16px;">
                        <div class="card-title">Memory Limit (KB)</div>
                        <div style="display:flex;align-items:center;gap:12px;">
                            <input type="number" id="custom-spill-limit" placeholder="Enter KB"
                                style="width:120px;padding:10px;background:rgba(0,0,0,0.3);border:1px solid rgba(255,255,255,0.2);border-radius:6px;color:#e2e8f0;font-size:14px;">
                            <button class="btn" onclick="applyCustomSpillLimit()" style="padding:10px 20px;">Apply</button>
                            <span style="color:#94a3b8;font-size:13px;">
                                Peak Memory: <strong id="peak-memory-display" style="color:#f59e0b;"></strong> KB
                            </span>
                        </div>
                    </div>

                    <div class="chart-container">
                        <div class="chart-title">Fast/Slow Memory Usage Over Time</div>
                        <div id="spill-chart" style="height:350px;"></div>
                        <div class="legend">
                            <div class="legend-item"><div class="legend-dot" style="background:#10b981"></div>Fast Memory</div>
                            <div class="legend-item"><div class="legend-dot" style="background:#ef4444"></div>Slow Memory (Spilled)</div>
                            <div class="legend-item"><div class="legend-dot" style="background:#f59e0b;opacity:0.5"></div>Memory Limit</div>
                        </div>
                    </div>

                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px;">
                        <div class="card">
                            <div class="card-title">Spill Summary</div>
                            <div id="spill-summary"></div>
                        </div>
                        <div class="card">
                            <div class="card-title">Trigger Nodes</div>
                            <div id="trigger-nodes"></div>
                        </div>
                    </div>

                    <div class="card" style="margin-top:16px;">
                        <div class="card-title">Spill/Reload Events</div>
                        <div class="tensor-list" id="spill-events" style="max-height:300px;"></div>
                    </div>
                </div>

                <div id="tab-comparison" class="tab-content">
                    <div class="chart-container">
                        <div class="chart-title">Strategy Comparison</div>
                        <div id="comparison-chart"></div>
                    </div>

                    <div class="card">
                        <div class="card-title">Detailed Comparison</div>
                        <table class="comparison-table" id="comparison-table">
                            <thead>
                                <tr>
                                    <th>Strategy</th>
                                    <th>Peak Memory</th>
                                    <th>Savings</th>
                                    <th>Static Memory</th>
                                </tr>
                            </thead>
                            <tbody></tbody>
                        </table>
                    </div>
                </div>

                <div id="tab-tensors" class="tab-content">
                    <div class="chart-container">
                        <div class="chart-title">Tensor Lifetime Gantt Chart</div>
                        <div id="lifetime-chart"></div>
                    </div>

                    <div class="card">
                        <div class="card-title">Tensor List</div>
                        <div class="tensor-list" id="tensor-list"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const data = {data_json};
        let currentStrategy = 'greedy';

        let currentSpillLimit = '';

        // Initialize
        function init() {{
            renderStrategyButtons();
            renderSummary();
            renderMemoryChart();
            renderMemoryMap();
            renderStaticLayout();
            renderSpillScheduler();
            renderComparisonChart();
            renderComparisonTable();
            renderLifetimeChart();
            renderTensorList();

            // Tab switching
            document.querySelectorAll('.tab').forEach(tab => {{
                tab.addEventListener('click', () => {{
                    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                    tab.classList.add('active');
                    document.getElementById('tab-' + tab.dataset.tab).classList.add('active');

                    // Re-render charts when tab becomes visible
                    if (tab.dataset.tab === 'comparison') {{
                        renderComparisonChart();
                    }} else if (tab.dataset.tab === 'tensors') {{
                        renderLifetimeChart();
                    }} else if (tab.dataset.tab === 'memorymap') {{
                        renderMemoryMap();
                    }} else if (tab.dataset.tab === 'spill') {{
                        renderSpillScheduler();
                    }} else if (tab.dataset.tab === 'static') {{
                        renderStaticLayout();
                    }}
                }});
            }});

            // Memory limit
            document.getElementById('show-constraint').addEventListener('change', renderMemoryChart);
            document.getElementById('memory-limit').addEventListener('input', renderMemoryChart);
        }}

        function renderStrategyButtons() {{
            const container = document.getElementById('strategy-buttons');
            const strategies = Object.keys(data.strategies);

            container.innerHTML = strategies.map(name => {{
                const s = data.strategies[name].summary;
                return `
                    <button class="strategy-btn ${{name === currentStrategy ? 'active' : ''}}"
                            onclick="selectStrategy('${{name}}')">
                        <div class="name">${{formatStrategyName(name)}}</div>
                        <div class="stats">Peak: ${{s.peak_min_mb.toFixed(2)}} MB | Savings: ${{s.savings_pct.toFixed(1)}}%</div>
                    </button>
                `;
            }}).join('');
        }}

        function formatStrategyName(name) {{
            return name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
        }}

        function selectStrategy(name) {{
            currentStrategy = name;
            renderStrategyButtons();
            renderSummary();
            renderMemoryChart();
            renderTensorList();
        }}

        function renderSummary() {{
            const s = data.strategies[currentStrategy].summary;
            const container = document.getElementById('summary-grid');
            container.innerHTML = `
                <div class="summary-card">
                    <div class="summary-value">${{s.peak_min_mb.toFixed(2)}}</div>
                    <div class="summary-label">Peak Memory (MB)</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value">${{s.savings_pct.toFixed(1)}}%</div>
                    <div class="summary-label">Memory Saved</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value">${{s.static_mb.toFixed(2)}}</div>
                    <div class="summary-label">Static (Weights)</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value">${{s.peak_step}}</div>
                    <div class="summary-label">Peak Step</div>
                </div>
            `;
        }}

        function renderMemoryChart() {{
            const strategyData = data.strategies[currentStrategy];
            const steps = strategyData.steps;

            const x = steps.map(s => s.step);
            const maxMem = steps.map(s => s.max_memory / 1e6);
            const minMem = steps.map(s => s.min_memory / 1e6);
            const liveMem = steps.map(s => s.live_memory / 1e6);

            const traces = [
                {{
                    x: x,
                    y: maxMem,
                    name: 'Max Memory (No Reuse)',
                    type: 'scatter',
                    fill: 'tozeroy',
                    fillcolor: 'rgba(239,68,68,0.1)',
                    line: {{ color: '#ef4444', width: 2 }},
                }},
                {{
                    x: x,
                    y: minMem,
                    name: 'Actual Memory',
                    type: 'scatter',
                    fill: 'tozeroy',
                    fillcolor: 'rgba(16,185,129,0.2)',
                    line: {{ color: '#10b981', width: 3 }},
                }},
                {{
                    x: x,
                    y: liveMem,
                    name: 'Live Tensors',
                    type: 'scatter',
                    line: {{ color: '#3b82f6', width: 2, dash: 'dot' }},
                }},
            ];

            // Add constraint line if enabled
            const showConstraint = document.getElementById('show-constraint').checked;
            const limitValue = parseFloat(document.getElementById('memory-limit').value);
            if (showConstraint && !isNaN(limitValue) && limitValue > 0) {{
                traces.push({{
                    x: [x[0], x[x.length-1]],
                    y: [limitValue, limitValue],
                    name: 'Memory Limit',
                    type: 'scatter',
                    mode: 'lines',
                    line: {{ color: '#f59e0b', width: 2, dash: 'dash' }},
                }});
            }}

            const layout = {{
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                margin: {{ t: 20, r: 20, b: 50, l: 60 }},
                xaxis: {{
                    title: 'Execution Step',
                    color: '#94a3b8',
                    gridcolor: 'rgba(255,255,255,0.05)',
                    zerolinecolor: 'rgba(255,255,255,0.1)',
                }},
                yaxis: {{
                    title: 'Memory (MB)',
                    color: '#94a3b8',
                    gridcolor: 'rgba(255,255,255,0.05)',
                    zerolinecolor: 'rgba(255,255,255,0.1)',
                }},
                legend: {{
                    orientation: 'h',
                    y: -0.15,
                    font: {{ color: '#94a3b8' }},
                }},
                hovermode: 'x unified',
            }};

            Plotly.newPlot('memory-chart', traces, layout, {{
                responsive: true,
                displayModeBar: false,
            }});

            // Click handler for step details
            document.getElementById('memory-chart').on('plotly_click', function(eventData) {{
                const stepIdx = eventData.points[0].x;
                showStepDetails(stepIdx);
            }});
        }}

        function showStepDetails(stepIdx) {{
            const strategyData = data.strategies[currentStrategy];
            const step = strategyData.steps.find(s => s.step === stepIdx);
            if (!step) return;

            const container = document.getElementById('step-details');
            container.classList.add('visible');

            document.getElementById('step-info').innerHTML = `
                <div class="info-row">
                    <span class="info-label">Step</span>
                    <span class="info-value">${{step.step}}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Node</span>
                    <span class="info-value">${{step.node}}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Operation</span>
                    <span class="info-value">${{step.op}}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Input Size</span>
                    <span class="info-value">${{(step.input_bytes / 1e6).toFixed(4)}} MB</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Output Size</span>
                    <span class="info-value">${{(step.output_bytes / 1e6).toFixed(4)}} MB</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Max Memory</span>
                    <span class="info-value">${{(step.max_memory / 1e6).toFixed(4)}} MB</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Actual Memory</span>
                    <span class="info-value">${{(step.min_memory / 1e6).toFixed(4)}} MB</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Live Tensors</span>
                    <span class="info-value">${{step.live_tensors.length}}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Freed</span>
                    <span class="info-value">${{step.freed.join(', ') || 'None'}}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Reused From</span>
                    <span class="info-value">${{step.reused_from || 'None'}}</span>
                </div>
            `;
        }}

        function renderMemoryMap() {{
            const strategyData = data.strategies[currentStrategy];
            const blocks = strategyData.memory_blocks || [];

            if (blocks.length === 0) {{
                document.getElementById('memorymap-chart').innerHTML =
                    '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#94a3b8;">No memory blocks to display</div>';
                return;
            }}

            // Create shapes for memory blocks
            const shapes = [];
            const annotations = [];

            // Find max values for scaling
            const maxStep = Math.max(...blocks.map(b => b.death)) + 1;
            const maxOffset = Math.max(...blocks.map(b => b.end_offset));

            // Create a shape for each memory block
            blocks.forEach((block, idx) => {{
                // Determine color based on block type
                let color;
                if (block.is_input) {{
                    color = 'rgba(16, 185, 129, 0.7)';  // green
                }} else if (block.is_output) {{
                    color = 'rgba(245, 158, 11, 0.7)';  // amber
                }} else if (block.reused_from) {{
                    color = 'rgba(139, 92, 246, 0.7)';  // purple
                }} else {{
                    color = 'rgba(59, 130, 246, 0.7)';  // blue
                }}

                shapes.push({{
                    type: 'rect',
                    x0: block.birth,
                    x1: block.death + 1,
                    y0: block.offset,
                    y1: block.end_offset,
                    fillcolor: color,
                    line: {{
                        color: color.replace('0.7', '1'),
                        width: 1,
                    }},
                    name: block.name,
                }});

                // Add label if block is large enough
                const blockHeight = block.end_offset - block.offset;
                const blockWidth = block.death - block.birth + 1;
                if (blockHeight > maxOffset * 0.05 && blockWidth > 1) {{
                    annotations.push({{
                        x: (block.birth + block.death + 1) / 2,
                        y: (block.offset + block.end_offset) / 2,
                        text: block.name.length > 15 ? block.name.slice(0, 13) + '..' : block.name,
                        showarrow: false,
                        font: {{ color: 'white', size: 10 }},
                    }});
                }}
            }});

            // Create invisible scatter trace for hover info
            const hoverTrace = {{
                x: blocks.map(b => (b.birth + b.death + 1) / 2),
                y: blocks.map(b => (b.offset + b.end_offset) / 2),
                mode: 'markers',
                marker: {{ size: 1, opacity: 0 }},
                text: blocks.map(b => `<b>${{b.name}}</b><br>` +
                    `Shape: ${{JSON.stringify(b.shape)}}<br>` +
                    `Size: ${{(b.size / 1024).toFixed(2)}} KB<br>` +
                    `Offset: ${{b.offset}} - ${{b.end_offset}}<br>` +
                    `Lifetime: Step ${{b.birth}} - ${{b.death}}<br>` +
                    (b.reused_from ? `Reused from: ${{b.reused_from}}` : 'New allocation')),
                hoverinfo: 'text',
                hoverlabel: {{
                    bgcolor: 'rgba(22, 33, 62, 0.95)',
                    bordercolor: '#f59e0b',
                    font: {{ color: '#e2e8f0' }},
                }},
                customdata: blocks,
            }};

            const layout = {{
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'rgba(0,0,0,0.2)',
                margin: {{ t: 30, r: 30, b: 60, l: 80 }},
                shapes: shapes,
                annotations: annotations,
                xaxis: {{
                    title: 'Execution Step',
                    color: '#94a3b8',
                    gridcolor: 'rgba(255,255,255,0.1)',
                    range: [-0.5, maxStep + 0.5],
                    dtick: Math.ceil(maxStep / 20),
                }},
                yaxis: {{
                    title: 'Memory Address (bytes)',
                    color: '#94a3b8',
                    gridcolor: 'rgba(255,255,255,0.1)',
                    range: [0, maxOffset * 1.05],
                    tickformat: '.2s',
                }},
                hovermode: 'closest',
            }};

            Plotly.newPlot('memorymap-chart', [hoverTrace], layout, {{
                responsive: true,
                displayModeBar: false,
            }});

            // Click handler for block details
            document.getElementById('memorymap-chart').on('plotly_click', function(eventData) {{
                if (eventData.points[0].customdata) {{
                    showBlockDetails(eventData.points[0].customdata);
                }}
            }});
        }}

        function showBlockDetails(block) {{
            const container = document.getElementById('block-details');
            container.classList.add('visible');

            document.getElementById('block-info').innerHTML = `
                <div class="info-row">
                    <span class="info-label">Name</span>
                    <span class="info-value">${{block.name}}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Shape</span>
                    <span class="info-value">${{JSON.stringify(block.shape)}}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Data Type</span>
                    <span class="info-value">${{block.dtype}}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Size</span>
                    <span class="info-value">${{(block.size / 1024).toFixed(2)}} KB (${{block.size}} bytes)</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Memory Offset</span>
                    <span class="info-value">${{block.offset}} - ${{block.end_offset}}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Lifetime</span>
                    <span class="info-value">Step ${{block.birth}} → Step ${{block.death}}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Duration</span>
                    <span class="info-value">${{block.death - block.birth + 1}} steps</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Type</span>
                    <span class="info-value">${{
                        block.is_input ? 'Input' :
                        block.is_output ? 'Output' :
                        block.reused_from ? 'Reused from ' + block.reused_from :
                        'Allocated'
                    }}</span>
                </div>
                ${{block.is_inplace ? `
                <div class="info-row">
                    <span class="info-label">In-place</span>
                    <span class="info-value">Yes (overwrites input)</span>
                </div>
                ` : ''}}
            `;
        }}

        function renderStaticLayout() {{
            const staticData = data.static_layout;
            if (!staticData.available) {{
                document.getElementById('static-summary').innerHTML = '<p style="color:#94a3b8;">Static layout not available</p>';
                return;
            }}

            // Summary
            document.getElementById('static-summary').innerHTML = `
                <div class="info-row">
                    <span class="info-label">Total Memory</span>
                    <span class="info-value">${{staticData.total_memory_kb.toFixed(2)}} KB</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Tensor Count</span>
                    <span class="info-value">${{staticData.tensor_count}}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Reused Addresses</span>
                    <span class="info-value">${{staticData.reuse_info.length}}</span>
                </div>
            `;

            // Reuse info
            if (staticData.reuse_info.length > 0) {{
                document.getElementById('static-reuse').innerHTML = staticData.reuse_info.map(r => `
                    <div style="margin-bottom:8px;padding:8px;background:rgba(16,185,129,0.1);border-radius:4px;">
                        <div style="color:#10b981;font-weight:500;">Offset 0x${{r.offset.toString(16).toUpperCase()}}</div>
                        <div style="color:#94a3b8;font-size:12px;margin-top:4px;">${{r.tensors.join(', ')}}</div>
                    </div>
                `).join('');
            }} else {{
                document.getElementById('static-reuse').innerHTML = '<p style="color:#94a3b8;">No memory reuse (all lifetimes overlap)</p>';
            }}

            // C code
            document.getElementById('static-c-code').textContent = staticData.c_code;

            // Table
            const tbody = document.querySelector('#static-table tbody');
            tbody.innerHTML = staticData.layout.map(t => `
                <tr>
                    <td>${{t.name}}</td>
                    <td style="font-family:monospace;color:#a5d6ff;">0x${{t.offset.toString(16).toUpperCase().padStart(8, '0')}}</td>
                    <td>${{t.offset}}</td>
                    <td>${{(t.size / 1024).toFixed(2)}} KB</td>
                    <td>${{t.birth}} - ${{t.death}}</td>
                    <td style="font-size:12px;">[${{t.shape.join(', ')}}]</td>
                </tr>
            `).join('');

            // Chart - Memory blocks as horizontal bars
            const layout = staticData.layout;
            if (layout.length === 0) return;

            // Find overlapping groups for coloring
            const offsetCounts = {{}};
            layout.forEach(t => {{
                offsetCounts[t.offset] = (offsetCounts[t.offset] || 0) + 1;
            }});

            const traces = layout.map((t, idx) => ({{
                x: [t.size / 1024],
                y: [t.name],
                type: 'bar',
                orientation: 'h',
                base: [t.offset / 1024],
                marker: {{
                    color: offsetCounts[t.offset] > 1 ? 'rgba(16,185,129,0.8)' : 'rgba(59,130,246,0.8)',
                    line: {{ color: 'rgba(255,255,255,0.3)', width: 1 }}
                }},
                hovertemplate: `${{t.name}}<br>Offset: 0x${{t.offset.toString(16).toUpperCase()}}<br>Size: ${{(t.size / 1024).toFixed(2)}} KB<br>Lifetime: ${{t.birth}} - ${{t.death}}<extra></extra>`,
                showlegend: false,
            }}));

            const chartLayout = {{
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                margin: {{ t: 20, r: 20, b: 50, l: 150 }},
                barmode: 'overlay',
                xaxis: {{
                    title: 'Memory Address (KB)',
                    color: '#94a3b8',
                    gridcolor: 'rgba(255,255,255,0.1)',
                }},
                yaxis: {{
                    color: '#94a3b8',
                    gridcolor: 'rgba(255,255,255,0.05)',
                }},
            }};

            Plotly.newPlot('static-layout-chart', traces, chartLayout, {{
                responsive: true,
                displayModeBar: false,
            }});
        }}

        function copyStaticCode() {{
            const code = document.getElementById('static-c-code').textContent;
            navigator.clipboard.writeText(code).then(() => {{
                alert('Code copied to clipboard!');
            }});
        }}

        function renderComparisonChart() {{
            const strategies = Object.keys(data.strategies);
            const peakMax = strategies.map(s => data.strategies[s].summary.peak_max_mb);
            const peakMin = strategies.map(s => data.strategies[s].summary.peak_min_mb);

            const traces = [
                {{
                    x: strategies.map(formatStrategyName),
                    y: peakMax,
                    name: 'Max Memory',
                    type: 'bar',
                    marker: {{ color: 'rgba(239,68,68,0.7)' }},
                }},
                {{
                    x: strategies.map(formatStrategyName),
                    y: peakMin,
                    name: 'Actual Memory',
                    type: 'bar',
                    marker: {{ color: 'rgba(16,185,129,0.9)' }},
                }},
            ];

            const layout = {{
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                margin: {{ t: 20, r: 20, b: 80, l: 60 }},
                barmode: 'group',
                xaxis: {{
                    color: '#94a3b8',
                    gridcolor: 'rgba(255,255,255,0.05)',
                }},
                yaxis: {{
                    title: 'Memory (MB)',
                    color: '#94a3b8',
                    gridcolor: 'rgba(255,255,255,0.05)',
                }},
                legend: {{
                    orientation: 'h',
                    y: -0.2,
                    font: {{ color: '#94a3b8' }},
                }},
            }};

            Plotly.newPlot('comparison-chart', traces, layout, {{
                responsive: true,
                displayModeBar: false,
            }});
        }}

        function renderComparisonTable() {{
            const tbody = document.querySelector('#comparison-table tbody');
            const strategies = Object.keys(data.strategies);

            // Find best values
            const minPeak = Math.min(...strategies.map(s => data.strategies[s].summary.peak_min_mb));
            const maxSavings = Math.max(...strategies.map(s => data.strategies[s].summary.savings_pct));

            tbody.innerHTML = strategies.map(name => {{
                const s = data.strategies[name].summary;
                return `
                    <tr>
                        <td>${{formatStrategyName(name)}}</td>
                        <td class="${{s.peak_min_mb === minPeak ? 'best' : ''}}">${{s.peak_min_mb.toFixed(2)}} MB</td>
                        <td class="${{s.savings_pct === maxSavings ? 'best' : ''}}">${{s.savings_pct.toFixed(1)}}%</td>
                        <td>${{s.static_mb.toFixed(2)}} MB</td>
                    </tr>
                `;
            }}).join('');
        }}

        function renderLifetimeChart() {{
            const strategyData = data.strategies[currentStrategy];
            const tensors = strategyData.tensors.filter(t => !t.is_weight);

            // Sort by birth time
            tensors.sort((a, b) => a.birth - b.birth);

            const traces = tensors.slice(0, 50).map((t, idx) => ({{
                x: [t.birth, t.death],
                y: [idx, idx],
                mode: 'lines',
                name: t.name,
                line: {{
                    width: 12,
                    color: t.is_input ? '#10b981' :
                           t.is_inplace ? '#8b5cf6' :
                           t.reused_from ? '#3b82f6' : '#f59e0b',
                }},
                hovertemplate: `${{t.name}}<br>Shape: ${{JSON.stringify(t.shape)}}<br>Size: ${{t.size_mb.toFixed(4)}} MB<br>Lifetime: ${{t.birth}} - ${{t.death}}<extra></extra>`,
            }}));

            const layout = {{
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                margin: {{ t: 20, r: 20, b: 50, l: 150 }},
                showlegend: false,
                xaxis: {{
                    title: 'Execution Step',
                    color: '#94a3b8',
                    gridcolor: 'rgba(255,255,255,0.05)',
                }},
                yaxis: {{
                    tickmode: 'array',
                    tickvals: tensors.slice(0, 50).map((_, i) => i),
                    ticktext: tensors.slice(0, 50).map(t => t.name.length > 20 ? t.name.slice(0, 18) + '..' : t.name),
                    color: '#94a3b8',
                    gridcolor: 'rgba(255,255,255,0.05)',
                }},
            }};

            Plotly.newPlot('lifetime-chart', traces, layout, {{
                responsive: true,
                displayModeBar: false,
            }});
        }}

        function renderTensorList() {{
            const strategyData = data.strategies[currentStrategy];
            const tensors = strategyData.tensors;

            // Sort by size
            tensors.sort((a, b) => b.size_mb - a.size_mb);

            const container = document.getElementById('tensor-list');
            container.innerHTML = tensors.map(t => {{
                let typeClass = '';
                let typeLabel = '';
                if (t.is_weight) {{
                    typeClass = 'weight';
                    typeLabel = '[Weight]';
                }} else if (t.is_input) {{
                    typeClass = 'input';
                    typeLabel = '[Input]';
                }} else if (t.is_inplace) {{
                    typeClass = 'inplace';
                    typeLabel = '[Inplace]';
                }} else if (t.reused_from) {{
                    typeClass = 'reused';
                    typeLabel = '[Reused]';
                }}

                return `
                    <div class="tensor-item ${{typeClass}}">
                        <div class="tensor-name">${{t.name}} ${{typeLabel}}</div>
                        <div class="tensor-info">
                            Shape: ${{JSON.stringify(t.shape)}} |
                            Size: ${{t.size_mb.toFixed(4)}} MB |
                            Lifetime: ${{t.birth}} - ${{t.death}}
                            ${{t.reused_from ? '| Reused from: ' + t.reused_from : ''}}
                        </div>
                    </div>
                `;
            }}).join('');
        }}

        function renderSpillScheduler() {{
            const spillData = data.spill_schedules;
            if (!spillData || !spillData.schedules) return;

            // Display peak memory
            document.getElementById('peak-memory-display').textContent = spillData.peak_memory_kb?.toFixed(1) || '?';

            // Set default value in input if not set
            const input = document.getElementById('custom-spill-limit');
            if (!input.value && !currentSpillLimit) {{
                // Default to first available schedule
                const firstKey = Object.keys(spillData.schedules)[0];
                if (firstKey) {{
                    currentSpillLimit = firstKey;
                    input.value = spillData.schedules[firstKey].limit_kb?.toFixed(0) || '';
                }}
            }}

            renderSpillChart();
            renderSpillSummary();
            renderSpillEvents();
        }}

        function applyCustomSpillLimit() {{
            const input = document.getElementById('custom-spill-limit');
            const customKB = parseFloat(input.value);
            if (isNaN(customKB) || customKB <= 0) {{
                alert('Please enter a valid memory limit in KB');
                return;
            }}

            const spillData = data.spill_schedules;
            const customLabel = `${{customKB.toFixed(0)}}KB`;

            // Check if we already have this exact limit
            if (!spillData.schedules[customLabel]) {{
                // Find closest existing schedule to use as base
                let closestSchedule = null;
                let closestDiff = Infinity;
                for (const [label, schedule] of Object.entries(spillData.schedules)) {{
                    if (schedule.error) continue;
                    const diff = Math.abs(schedule.limit_kb - customKB);
                    if (diff < closestDiff) {{
                        closestDiff = diff;
                        closestSchedule = schedule;
                    }}
                }}

                if (closestSchedule) {{
                    // Use closest schedule's data with adjusted limit
                    spillData.schedules[customLabel] = {{
                        ...closestSchedule,
                        limit_kb: customKB,
                    }};
                }}
            }}

            currentSpillLimit = customLabel;
            renderSpillScheduler();
        }}

        function renderSpillChart() {{
            const spillData = data.spill_schedules;
            const schedule = spillData.schedules[currentSpillLimit];
            if (!schedule || schedule.error) {{
                document.getElementById('spill-chart').innerHTML =
                    '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#94a3b8;">No spill data available</div>';
                return;
            }}

            const fast = schedule.fast_timeline || [];
            const slow = schedule.slow_timeline || [];
            const limit = schedule.limit_kb;
            const x = fast.map((_, i) => i);

            const traces = [
                {{
                    x: x,
                    y: fast,
                    name: 'Fast Memory',
                    type: 'scatter',
                    fill: 'tozeroy',
                    fillcolor: 'rgba(16,185,129,0.2)',
                    line: {{ color: '#10b981', width: 2 }},
                }},
                {{
                    x: x,
                    y: slow,
                    name: 'Slow Memory (Spilled)',
                    type: 'scatter',
                    fill: 'tozeroy',
                    fillcolor: 'rgba(239,68,68,0.2)',
                    line: {{ color: '#ef4444', width: 2 }},
                }},
                {{
                    x: [0, x.length - 1],
                    y: [limit, limit],
                    name: 'Memory Limit',
                    type: 'scatter',
                    mode: 'lines',
                    line: {{ color: '#f59e0b', width: 2, dash: 'dash' }},
                }},
            ];

            // Add markers for spill/reload events
            const events = schedule.events || [];
            const spillEvents = events.filter(e => e.type === 'spill');
            const reloadEvents = events.filter(e => e.type === 'reload');

            if (spillEvents.length > 0) {{
                traces.push({{
                    x: spillEvents.map(e => e.step),
                    y: spillEvents.map(e => e.fast_kb),
                    name: 'Spill',
                    mode: 'markers',
                    marker: {{ color: '#ef4444', size: 10, symbol: 'triangle-down' }},
                    text: spillEvents.map(e => `Spill: ${{e.tensor}} (${{e.size_kb.toFixed(1)}}KB)`),
                    hoverinfo: 'text',
                }});
            }}

            if (reloadEvents.length > 0) {{
                traces.push({{
                    x: reloadEvents.map(e => e.step),
                    y: reloadEvents.map(e => e.fast_kb),
                    name: 'Reload',
                    mode: 'markers',
                    marker: {{ color: '#3b82f6', size: 10, symbol: 'triangle-up' }},
                    text: reloadEvents.map(e => `Reload: ${{e.tensor}} (${{e.size_kb.toFixed(1)}}KB)`),
                    hoverinfo: 'text',
                }});
            }}

            const layout = {{
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                margin: {{ t: 20, r: 20, b: 50, l: 60 }},
                xaxis: {{
                    title: 'Execution Step',
                    color: '#94a3b8',
                    gridcolor: 'rgba(255,255,255,0.05)',
                }},
                yaxis: {{
                    title: 'Memory (KB)',
                    color: '#94a3b8',
                    gridcolor: 'rgba(255,255,255,0.05)',
                }},
                legend: {{
                    orientation: 'h',
                    y: -0.15,
                    font: {{ color: '#94a3b8' }},
                }},
                hovermode: 'closest',
            }};

            Plotly.newPlot('spill-chart', traces, layout, {{
                responsive: true,
                displayModeBar: false,
            }});
        }}

        function renderSpillSummary() {{
            const spillData = data.spill_schedules;
            const schedule = spillData.schedules[currentSpillLimit];

            const summaryContainer = document.getElementById('spill-summary');
            const triggerContainer = document.getElementById('trigger-nodes');

            if (!schedule || schedule.error) {{
                summaryContainer.innerHTML = '<div style="color:#94a3b8;">No data</div>';
                triggerContainer.innerHTML = '<div style="color:#94a3b8;">No data</div>';
                return;
            }}

            summaryContainer.innerHTML = `
                <div class="info-row" style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.1);">
                    <span style="color:#94a3b8;">Memory Limit</span>
                    <span style="color:#e2e8f0;font-family:monospace;">${{schedule.limit_kb.toFixed(1)}} KB</span>
                </div>
                <div class="info-row" style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.1);">
                    <span style="color:#94a3b8;">Total Spills</span>
                    <span style="color:#ef4444;font-family:monospace;">${{schedule.total_spills}}</span>
                </div>
                <div class="info-row" style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.1);">
                    <span style="color:#94a3b8;">Total Reloads</span>
                    <span style="color:#3b82f6;font-family:monospace;">${{schedule.total_reloads}}</span>
                </div>
                <div class="info-row" style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.1);">
                    <span style="color:#94a3b8;">Spill Data Volume</span>
                    <span style="color:#e2e8f0;font-family:monospace;">${{schedule.spill_bytes_kb.toFixed(1)}} KB</span>
                </div>
                <div class="info-row" style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.1);">
                    <span style="color:#94a3b8;">Peak Fast Memory</span>
                    <span style="color:#10b981;font-family:monospace;">${{schedule.peak_fast_kb.toFixed(1)}} KB</span>
                </div>
                <div class="info-row" style="display:flex;justify-content:space-between;padding:8px 0;">
                    <span style="color:#94a3b8;">Peak Slow Memory</span>
                    <span style="color:#ef4444;font-family:monospace;">${{schedule.peak_slow_kb.toFixed(1)}} KB</span>
                </div>
            `;

            const spillNodes = schedule.spill_trigger_nodes || [];
            const reloadNodes = schedule.reload_trigger_nodes || [];

            triggerContainer.innerHTML = `
                <div style="margin-bottom:12px;">
                    <div style="color:#ef4444;font-size:12px;font-weight:600;margin-bottom:6px;">SPILL Triggers (${{spillNodes.length}})</div>
                    <div style="display:flex;flex-wrap:wrap;gap:4px;">
                        ${{spillNodes.map(n => `<span style="background:rgba(239,68,68,0.2);color:#ef4444;padding:2px 8px;border-radius:4px;font-size:11px;">${{n}}</span>`).join('')}}
                        ${{spillNodes.length === 0 ? '<span style="color:#94a3b8;font-size:12px;">None</span>' : ''}}
                    </div>
                </div>
                <div>
                    <div style="color:#3b82f6;font-size:12px;font-weight:600;margin-bottom:6px;">RELOAD Triggers (${{reloadNodes.length}})</div>
                    <div style="display:flex;flex-wrap:wrap;gap:4px;">
                        ${{reloadNodes.map(n => `<span style="background:rgba(59,130,246,0.2);color:#3b82f6;padding:2px 8px;border-radius:4px;font-size:11px;">${{n}}</span>`).join('')}}
                        ${{reloadNodes.length === 0 ? '<span style="color:#94a3b8;font-size:12px;">None</span>' : ''}}
                    </div>
                </div>
            `;
        }}

        function renderSpillEvents() {{
            const spillData = data.spill_schedules;
            const schedule = spillData.schedules[currentSpillLimit];
            const container = document.getElementById('spill-events');

            if (!schedule || schedule.error || !schedule.events) {{
                container.innerHTML = '<div style="color:#94a3b8;padding:12px;">No events</div>';
                return;
            }}

            const events = schedule.events;
            if (events.length === 0) {{
                container.innerHTML = '<div style="color:#94a3b8;padding:12px;">No spill/reload events needed</div>';
                return;
            }}

            container.innerHTML = events.map(e => {{
                const isSpill = e.type === 'spill';
                const color = isSpill ? '#ef4444' : '#3b82f6';
                const icon = isSpill ? '↓' : '↑';
                return `
                    <div class="tensor-item" style="border-left:3px solid ${{color}};">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <div>
                                <span style="color:${{color}};font-weight:600;">${{icon}} ${{e.type.toUpperCase()}}</span>
                                <span style="color:#e2e8f0;margin-left:8px;">${{e.tensor}}</span>
                            </div>
                            <span style="color:#94a3b8;font-size:11px;">Step ${{e.step}}</span>
                        </div>
                        <div style="color:#94a3b8;font-size:11px;margin-top:4px;">
                            Size: ${{e.size_kb.toFixed(1)}} KB | Node: ${{e.node}} | Fast: ${{e.fast_kb.toFixed(1)}}KB, Slow: ${{e.slow_kb.toFixed(1)}}KB
                        </div>
                    </div>
                `;
            }}).join('');
        }}

        init();
    </script>
</body>
</html>'''
        return html

    def save(self, path: str) -> None:
        """Save visualization to HTML file"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_html())
        print(f"Saved memory analysis to: {path}")

    def serve(self, port: int = 8080, open_browser: bool = True) -> None:
        """Serve visualization via HTTP"""
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
            print(f"Memory Analyzer: {url}")
            if open_browser:
                webbrowser.open(url)
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nStopped")


def visualize_memory(
    gm: fx.GraphModule,
    output_path: Optional[str] = None,
    dynamic_shapes: Optional[Dict[str, int]] = None,
) -> MemoryVisualizer:
    """Convenience function to create memory visualization"""
    viz = MemoryVisualizer(gm, dynamic_shapes)
    if output_path:
        viz.save(output_path)
    return viz


def serve_memory(
    gm: fx.GraphModule,
    port: int = 8080,
    dynamic_shapes: Optional[Dict[str, int]] = None,
) -> None:
    """Convenience function to serve memory visualization"""
    viz = MemoryVisualizer(gm, dynamic_shapes)
    viz.serve(port=port)
