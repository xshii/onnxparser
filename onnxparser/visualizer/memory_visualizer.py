# -*- coding: utf-8 -*-
"""Interactive memory analysis visualization"""

import json
import http.server
import socketserver
import webbrowser
from typing import Dict, Optional, List, Any
import torch.fx as fx

from ..analysis.memory_analyzer import MemoryAnalyzer, MemoryConstraint, AnalysisResult


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
                    "size_mb": tensor.size_bytes / 1e6,
                    "birth": tensor.birth_step,
                    "death": tensor.death_step,
                    "is_weight": tensor.is_weight,
                    "is_input": tensor.is_input,
                    "is_output": tensor.is_output,
                    "reused_from": tensor.reused_from,
                    "is_inplace": tensor.is_inplace,
                })

            strategies_data[strategy_name] = {
                "steps": steps_data,
                "tensors": tensors_data,
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
        }

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

        // Initialize
        function init() {{
            renderStrategyButtons();
            renderSummary();
            renderMemoryChart();
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
