/**
 * ONNX Parser Visualizer - Frontend Application
 *
 * Pure rendering and interaction logic.
 * All data processing and layout computation happens on the backend.
 */

class VisualizerApp {
    constructor() {
        // State
        this.currentModel = null;
        this.graphData = null;
        this.memoryData = null;
        this.currentStrategy = 'greedy';
        this.selectedNode = null;
        this.flowActive = false;
        this.currentView = 'all';
        this.memoryViewMode = 'lifetime';  // 'lifetime' or 'layout'

        // D3 elements
        this.svg = null;
        this.g = null;
        this.zoom = null;
        this.nodes = null;
        this.links = null;
        this.nodeById = {};

        // DOM elements
        this.container = document.getElementById('graph-container');

        this.init();
    }

    async init() {
        // Initialize graph first (synchronous) before async operations
        this.initGraph();
        this.setupEventListeners();
        // Then load data from server
        await this.refreshModels();
        await this.loadStrategies();
    }

    setupEventListeners() {
        document.getElementById('search').addEventListener('input', (e) => {
            const q = e.target.value.toLowerCase();
            if (this.nodes) {
                this.nodes.style('opacity', d => !q || d.name.toLowerCase().includes(q) ? 1 : 0.15);
            }
        });
    }

    // === API Methods ===

    async refreshModels() {
        try {
            const res = await fetch('/api/models');
            const data = await res.json();
            const select = document.getElementById('model-select');
            const current = select.value;
            select.innerHTML = '<option value="">Select Model...</option>' +
                data.models.map(m => `<option value="${m}" ${m === current ? 'selected' : ''}>${m}</option>`).join('');
        } catch (e) {
            console.error('Failed to load models:', e);
        }
    }

    async loadStrategies() {
        try {
            const res = await fetch('/api/strategies');
            const data = await res.json();
            const select = document.getElementById('strategy-select');
            select.innerHTML = data.strategies.map(s =>
                `<option value="${s}" ${s === this.currentStrategy ? 'selected' : ''}>${this.formatStrategyName(s)}</option>`
            ).join('');
        } catch (e) {
            console.error('Failed to load strategies:', e);
        }
    }

    formatStrategyName(name) {
        return name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
    }

    async loadModel() {
        const select = document.getElementById('model-select');
        const name = select.value;
        console.log('loadModel called, name:', name, 'type:', typeof name);

        if (!name) {
            this.currentModel = null;
            this.graphData = null;
            this.memoryData = null;
            this.showNoModel();
            return;
        }

        this.showLoading();

        try {
            // Get graph data with layout computed by backend
            let width = 800, height = 600;
            if (this.container) {
                width = this.container.clientWidth || 800;
                height = this.container.clientHeight || 600;
            }
            const url = `/api/model?name=${encodeURIComponent(name)}&width=${width}&height=${height}`;
            console.log('Fetching model:', url);
            const res = await fetch(url);
            console.log('Fetch response status:', res.status);

            // Parse JSON with better error handling for Safari
            const text = await res.text();
            console.log('Response length:', text.length);
            let data;
            try {
                data = JSON.parse(text);
            } catch (jsonErr) {
                console.error('JSON parse error:', jsonErr);
                console.error('Response preview:', text.substring(0, 500));
                throw new Error('Invalid JSON response from server');
            }

            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }

            this.currentModel = name;
            this.graphData = data.graph;

            // Load memory data
            await this.loadMemoryData();

            try {
                this.renderGraph();
            } catch (graphErr) {
                console.error('renderGraph failed:', graphErr);
                throw new Error('renderGraph: ' + graphErr.message);
            }

            try {
                this.renderMemoryChart();
            } catch (memErr) {
                console.error('renderMemoryChart failed:', memErr);
                // Don't throw - memory chart failure shouldn't block graph
            }

            this.enableControls();
        } catch (e) {
            console.error('Failed to load model:', e);
            alert('Failed to load model: ' + e.message);
        }
    }

    async loadMemoryData() {
        if (!this.currentModel) return;

        try {
            let url = `/api/memory?name=${encodeURIComponent(this.currentModel)}&strategy=${this.currentStrategy}`;

            // Include memory limit if specified
            const limitInput = document.getElementById('memory-limit');
            if (limitInput && limitInput.value) {
                const limitKB = parseFloat(limitInput.value);
                if (limitKB > 0) {
                    url += `&limit=${limitKB}`;
                }
            }

            const res = await fetch(url);
            this.memoryData = await res.json();
        } catch (e) {
            console.error('Failed to load memory data:', e);
            this.memoryData = { error: e.message };
        }
    }

    async applyMemoryConstraint() {
        // Reload memory data with the new constraint
        await this.loadMemoryData();
        this.renderMemoryChart();
    }

    async changeStrategy() {
        this.currentStrategy = document.getElementById('strategy-select').value;
        await this.loadMemoryData();
        this.renderMemoryChart();
    }

    async reloadModel() {
        if (!this.currentModel) return;

        // Clear backend cache
        try {
            await fetch(`/api/reload?name=${encodeURIComponent(this.currentModel)}`, {
                method: 'POST'
            });
        } catch (e) {
            console.error('Failed to clear cache:', e);
        }

        // Reload current model
        await this.loadModel();
    }

    // === UI State Methods ===

    showNoModel() {
        document.getElementById('no-model').style.display = 'flex';
        document.getElementById('no-model').innerHTML = `
            <div class="no-model-icon">📊</div>
            <div>Select a model to visualize</div>
        `;
        document.getElementById('graph-svg').style.display = 'none';
        document.getElementById('reloadBtn').disabled = true;
        document.getElementById('resetBtn').disabled = true;
        document.getElementById('flowBtn').disabled = true;
        document.getElementById('memory-chart').innerHTML = '<div class="no-model"><div>Select a model</div></div>';
    }

    showLoading() {
        document.getElementById('no-model').innerHTML = '<div class="loading">Loading model</div>';
        document.getElementById('no-model').style.display = 'flex';
        document.getElementById('memory-chart').innerHTML = '<div class="loading">Loading</div>';
    }

    enableControls() {
        document.getElementById('reloadBtn').disabled = false;
        document.getElementById('resetBtn').disabled = false;
        document.getElementById('flowBtn').disabled = false;
    }

    setView(view) {
        this.currentView = view;
        const app = document.getElementById('app');
        app.className = '';
        if (view !== 'all') {
            app.classList.add('focus-' + view);
        }
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.view === view);
        });
        setTimeout(() => {
            if (view === 'all' || view === 'graph') this.resetGraphView();
            if (view === 'all' || view === 'memory') this.renderMemoryChart();
        }, 50);
    }

    // === Graph Rendering ===

    initGraph() {
        this.svg = d3.select('#graph-svg');
        this.g = this.svg.append('g');

        this.zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', e => this.g.attr('transform', e.transform));
        this.svg.call(this.zoom);

        // Arrow marker
        this.svg.append('defs').append('marker')
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

        this.svg.on('click', (e) => {
            if (e.target === this.svg.node()) {
                this.clearSelection();
            }
        });
    }

    renderGraph() {
        if (!this.graphData) {
            console.log('renderGraph: no graph data');
            return;
        }

        console.log('renderGraph: starting with', this.graphData.nodes.length, 'nodes');

        document.getElementById('no-model').style.display = 'none';
        document.getElementById('graph-svg').style.display = 'block';

        if (!this.g) {
            console.error('renderGraph: this.g is not initialized');
            throw new Error('Graph group not initialized - call initGraph first');
        }

        // Clear previous
        this.g.selectAll('*').remove();
        this.nodeById = {};

        // Build lookup
        this.graphData.nodes.forEach(n => this.nodeById[n.id] = n);

        // Draw edges (positions come from backend)
        this.links = this.g.selectAll('.link')
            .data(this.graphData.edges.filter(d => {
                // Filter out edges with missing nodes
                return this.nodeById[d.source] && this.nodeById[d.target];
            }))
            .enter()
            .append('path')
            .attr('class', 'link')
            .attr('marker-end', function() { return 'url(#arrow)'; })
            .attr('d', d => {
                const s = this.nodeById[d.source];
                const t = this.nodeById[d.target];
                const sx = (s.x || 0) + (s.width || 0);
                const sy = (s.y || 0) + (s.height || 0) / 2;
                const tx = t.x || 0;
                const ty = (t.y || 0) + (t.height || 0) / 2;
                const mx = (sx + tx) / 2;
                return `M${sx},${sy} C${mx},${sy} ${mx},${ty} ${tx},${ty}`;
            });

        // Draw nodes
        this.nodes = this.g.selectAll('.node')
            .data(this.graphData.nodes)
            .enter()
            .append('g')
            .attr('class', d => `node node-${d.type}${d.is_weight ? ' node-small' : ''}`)
            .attr('data-name', d => d.name)
            .attr('transform', d => `translate(${d.x},${d.y})`)
            .on('click', (e, d) => this.selectNode(e, d));

        this.nodes.append('rect')
            .attr('width', d => d.width)
            .attr('height', d => d.height);

        this.nodes.append('text')
            .attr('class', 'node-name')
            .attr('x', d => d.width / 2)
            .attr('y', d => d.is_weight ? 12 : 15)
            .attr('text-anchor', 'middle')
            .text(d => {
                const maxLen = d.is_weight ? 10 : 14;
                return d.name.length > maxLen ? d.name.slice(0, maxLen - 2) + '..' : d.name;
            });

        this.nodes.filter(d => !d.is_weight).append('text')
            .attr('class', 'node-shape')
            .attr('x', d => d.width / 2)
            .attr('y', 28)
            .attr('text-anchor', 'middle')
            .text(d => d.shape ? JSON.stringify(d.shape) : '');

        // Order badges
        const badgeNodes = this.nodes.filter(d => !d.is_weight);
        badgeNodes.append('circle').attr('cx', -5).attr('cy', -5).attr('r', 8).attr('fill', '#fbbf24');
        badgeNodes.append('text')
            .attr('class', 'order-badge')
            .attr('x', -5).attr('y', -2)
            .attr('text-anchor', 'middle')
            .text((d, i) => i + 1);

        this.resetGraphView();
    }

    selectNode(e, d) {
        e.stopPropagation();

        d3.selectAll('.node').classed('selected', false).classed('dimmed', false);
        d3.selectAll('.link').classed('highlighted', false);
        d3.select(e.currentTarget).classed('selected', true);
        this.links.classed('highlighted', l => l.source === d.id || l.target === d.id);
        this.selectedNode = d;

        // Update sidebar
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

        const insightsSection = document.getElementById('insights-section');

        if (d.min !== undefined && d.min !== null) {
            statsSection.style.display = 'block';
            document.getElementById('stat-min').textContent = (d.min ?? 0).toFixed(4);
            document.getElementById('stat-max').textContent = (d.max ?? 0).toFixed(4);
            document.getElementById('stat-mean').textContent = (d.mean ?? 0).toFixed(4);
            document.getElementById('stat-std').textContent = (d.std ?? 0).toFixed(4);

            // Render insights
            if (d.insights && d.insights.length > 0) {
                insightsSection.style.display = 'block';
                document.getElementById('insights-list').innerHTML = this.renderInsights(d.insights);
            } else {
                insightsSection.style.display = 'none';
            }

            if (d.histogram && d.histogram.length > 0) {
                histSection.style.display = 'block';
                Plotly.newPlot('histogram-plot', [{
                    x: d.histogram,
                    type: 'histogram',
                    marker: { color: 'rgba(102, 126, 234, 0.7)', line: { color: 'rgba(102, 126, 234, 1)', width: 1 } },
                    nbinsx: 25
                }], {
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    margin: { t: 5, r: 5, b: 25, l: 35 },
                    xaxis: { color: '#94a3b8', gridcolor: 'rgba(255,255,255,0.1)' },
                    yaxis: { color: '#94a3b8', gridcolor: 'rgba(255,255,255,0.1)' },
                    bargap: 0.05
                }, { responsive: true, displayModeBar: false });
            } else {
                histSection.style.display = 'none';
            }

            if (d.heatmap && d.heatmap.slices && d.heatmap.slices.length > 0) {
                heatSection.style.display = 'block';
                this.currentHeatmapData = d.heatmap;
                this.currentHeatmapIndex = 0;
                this.renderHeatmapControls(d.heatmap);
                const infMask = d.heatmap.inf_masks ? d.heatmap.inf_masks[0] : null;
                this.renderHeatmap(d.heatmap.slices[0], infMask);
            } else {
                heatSection.style.display = 'none';
            }

            // Show raw values preview with slice controls
            const valuesSection = document.getElementById('values-section');
            if (d.values && d.values.slices && d.values.slices.length > 0) {
                valuesSection.style.display = 'block';
                this.currentValuesData = d.values;
                this.currentValuesIndex = 0;
                this.renderValuesControls(d.values);
                this.renderValuesPreview(d.values.slices[0], d.values.shape);
            } else {
                valuesSection.style.display = 'none';
            }
        } else {
            statsSection.style.display = 'none';
            histSection.style.display = 'none';
            heatSection.style.display = 'none';
            insightsSection.style.display = 'none';
            document.getElementById('values-section').style.display = 'none';
        }

        // Highlight memory block
        this.renderMemoryChart();
    }

    renderHeatmapControls(heatmapData) {
        const controlsEl = document.getElementById('heatmap-controls');
        if (!heatmapData.dims || heatmapData.dims.length === 0) {
            controlsEl.innerHTML = '';
            return;
        }

        let html = '<div class="slice-controls">';
        if (heatmapData.dims.length === 1) {
            // Single dimension slider
            const dim = heatmapData.dims[0];
            html += `<label>${dim.name}: <input type="range" id="slice-slider" min="0" max="${dim.shown - 1}" value="0"
                     oninput="app.updateHeatmapSlice(this.value)">
                     <span id="slice-value">0</span>/${dim.shown}${dim.size > dim.shown ? ` (of ${dim.size})` : ''}</label>`;
        } else if (heatmapData.dims.length === 2) {
            // Two dimension controls
            const dim0 = heatmapData.dims[0];
            const dim1 = heatmapData.dims[1];
            html += `<label>${dim0.name}: <select id="slice-dim0" onchange="app.updateHeatmapSlice2D()">
                     ${Array.from({length: dim0.shown}, (_, i) => `<option value="${i}">${i}</option>`).join('')}
                     </select>${dim0.size > dim0.shown ? ` (of ${dim0.size})` : ''}</label>`;
            html += `<label>${dim1.name}: <select id="slice-dim1" onchange="app.updateHeatmapSlice2D()">
                     ${Array.from({length: dim1.shown}, (_, i) => `<option value="${i}">${i}</option>`).join('')}
                     </select>${dim1.size > dim1.shown ? ` (of ${dim1.size})` : ''}</label>`;
        }
        html += '</div>';
        controlsEl.innerHTML = html;
    }

    updateHeatmapSlice(index) {
        this.currentHeatmapIndex = parseInt(index);
        document.getElementById('slice-value').textContent = index;
        const infMask = this.currentHeatmapData.inf_masks ? this.currentHeatmapData.inf_masks[this.currentHeatmapIndex] : null;
        this.renderHeatmap(this.currentHeatmapData.slices[this.currentHeatmapIndex], infMask);
    }

    updateHeatmapSlice2D() {
        const dim0 = parseInt(document.getElementById('slice-dim0').value);
        const dim1 = parseInt(document.getElementById('slice-dim1').value);
        const dim1Size = this.currentHeatmapData.dims[1].shown;
        const index = dim0 * dim1Size + dim1;
        this.currentHeatmapIndex = index;
        const infMask = this.currentHeatmapData.inf_masks ? this.currentHeatmapData.inf_masks[index] : null;
        this.renderHeatmap(this.currentHeatmapData.slices[index], infMask);
    }

    renderHeatmap(data, infMask = null) {
        // Find data range for symmetric colorscale
        const flat = data.flat();
        const absMax = Math.max(...flat.map(v => Math.abs(v)));
        const hasNegative = flat.some(v => v < 0);

        // Diverging colorscale: blue (negative) -> white (zero) -> red (positive)
        let colorscale, zmin, zmax, zmid;
        if (hasNegative) {
            // Diverging: symmetric around zero
            colorscale = [
                [0, '#2563eb'],      // Blue (negative max)
                [0.3, '#60a5fa'],    // Light blue
                [0.5, '#ffffff'],    // White (zero)
                [0.7, '#fca5a5'],    // Light red
                [1, '#dc2626']       // Red (positive max)
            ];
            zmin = -absMax;
            zmax = absMax;
            zmid = 0;
        } else {
            // Sequential warm scale for all-positive data
            colorscale = [
                [0, '#ffffff'],      // White (zero/min)
                [0.25, '#fef3c7'],   // Light yellow
                [0.5, '#fbbf24'],    // Yellow
                [0.75, '#f97316'],   // Orange
                [1, '#dc2626']       // Red (max)
            ];
            zmin = 0;
            zmax = absMax || 1;
            zmid = undefined;
        }

        const traces = [{
            z: data,
            type: 'heatmap',
            colorscale: colorscale,
            zmin: zmin,
            zmax: zmax,
            zmid: zmid,
            showscale: true,
            colorbar: {
                thickness: 10,
                len: 0.8,
                tickfont: { size: 9, color: '#94a3b8' },
                outlinewidth: 0
            },
            hovertemplate: 'x: %{x}<br>y: %{y}<br>value: %{z:.4f}<extra></extra>'
        }];

        // Overlay black cells for inf/nan values
        if (infMask) {
            // Create a second heatmap trace for inf values
            const infData = data.map((row, i) =>
                row.map((val, j) => infMask[i] && infMask[i][j] ? 1 : null)
            );
            traces.push({
                z: infData,
                type: 'heatmap',
                colorscale: [[0, '#000000'], [1, '#000000']],
                showscale: false,
                hovertemplate: 'x: %{x}<br>y: %{y}<br>value: Inf/NaN<extra></extra>'
            });
        }

        Plotly.newPlot('heatmap-plot', traces, {
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            margin: { t: 5, r: 40, b: 25, l: 35 },
            xaxis: { color: '#94a3b8', showgrid: false },
            yaxis: { color: '#94a3b8', showgrid: false, autorange: 'reversed' }
        }, { responsive: true, displayModeBar: false });
    }

    formatValuesTable(values, shape) {
        // Format tensor values as a table
        const formatNum = (n) => {
            if (typeof n !== 'number') return String(n);
            if (Math.abs(n) < 0.0001 && n !== 0) return n.toExponential(2);
            return n.toFixed(4);
        };

        if (!Array.isArray(values)) {
            return `<span class="value-cell">${formatNum(values)}</span>`;
        }

        // 1D array
        if (!Array.isArray(values[0])) {
            return `<div class="values-row">${values.map(v =>
                `<span class="value-cell">${formatNum(v)}</span>`
            ).join('')}</div>`;
        }

        // 2D array - render as table
        let html = '<table class="values-table">';
        for (const row of values) {
            html += '<tr>';
            for (const val of row) {
                html += `<td>${formatNum(val)}</td>`;
            }
            html += '</tr>';
        }
        html += '</table>';

        // Add shape info
        html += `<div class="values-info">Showing [${values.length}×${values[0]?.length || 0}] of ${JSON.stringify(shape)}</div>`;
        return html;
    }

    renderInsights(insights) {
        const icons = {
            'warning': '⚠️',
            'error': '🔴',
            'success': '✅',
            'info': 'ℹ️'
        };

        return insights.map(insight => `
            <div class="insight-card ${insight.type}">
                <div class="insight-title">
                    <span class="insight-icon">${icons[insight.type] || 'ℹ️'}</span>
                    ${insight.title}
                </div>
                <div class="insight-detail">${insight.detail}</div>
            </div>
        `).join('');
    }

    renderValuesControls(valuesData) {
        const controlsEl = document.getElementById('values-controls');
        if (!valuesData.dims || valuesData.dims.length === 0) {
            controlsEl.innerHTML = '';
            return;
        }

        let html = '<div class="slice-controls">';
        if (valuesData.dims.length === 1) {
            const dim = valuesData.dims[0];
            html += `<label>${dim.name}: <input type="range" id="values-slider" min="0" max="${dim.shown - 1}" value="0"
                     oninput="app.updateValuesSlice(this.value)">
                     <span id="values-slice-value">0</span>/${dim.shown}${dim.size > dim.shown ? ` (of ${dim.size})` : ''}</label>`;
        } else if (valuesData.dims.length === 2) {
            const dim0 = valuesData.dims[0];
            const dim1 = valuesData.dims[1];
            html += `<label>${dim0.name}: <select id="values-dim0" onchange="app.updateValuesSlice2D()">
                     ${Array.from({length: dim0.shown}, (_, i) => `<option value="${i}">${i}</option>`).join('')}
                     </select>${dim0.size > dim0.shown ? ` (of ${dim0.size})` : ''}</label>`;
            html += `<label>${dim1.name}: <select id="values-dim1" onchange="app.updateValuesSlice2D()">
                     ${Array.from({length: dim1.shown}, (_, i) => `<option value="${i}">${i}</option>`).join('')}
                     </select>${dim1.size > dim1.shown ? ` (of ${dim1.size})` : ''}</label>`;
        }
        html += '</div>';
        controlsEl.innerHTML = html;
    }

    updateValuesSlice(index) {
        this.currentValuesIndex = parseInt(index);
        document.getElementById('values-slice-value').textContent = index;
        this.renderValuesPreview(this.currentValuesData.slices[this.currentValuesIndex], this.currentValuesData.shape);
    }

    updateValuesSlice2D() {
        const dim0 = parseInt(document.getElementById('values-dim0').value);
        const dim1 = parseInt(document.getElementById('values-dim1').value);
        const dim1Size = this.currentValuesData.dims[1].shown;
        const index = dim0 * dim1Size + dim1;
        this.currentValuesIndex = index;
        this.renderValuesPreview(this.currentValuesData.slices[index], this.currentValuesData.shape);
    }

    renderValuesPreview(values, shape) {
        const valuesEl = document.getElementById('values-preview');
        valuesEl.innerHTML = this.formatValuesTable(values, shape);
    }

    clearSelection() {
        d3.selectAll('.node').classed('selected', false).classed('highlighted', false).classed('dimmed', false);
        d3.selectAll('.link').classed('highlighted', false);
        this.selectedNode = null;
        document.getElementById('placeholder').style.display = 'flex';
        document.getElementById('details').style.display = 'none';
        this.renderMemoryChart();
    }

    resetGraphView() {
        if (!this.g || !this.g.node()) return;
        const bounds = this.g.node().getBBox();
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        const scale = Math.min(width / (bounds.width + 80), height / (bounds.height + 80)) * 0.85;
        const tx = (width - bounds.width * scale) / 2 - bounds.x * scale;
        const ty = (height - bounds.height * scale) / 2 - bounds.y * scale;
        this.svg.transition().duration(500).call(this.zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
    }

    toggleFlow() {
        this.flowActive = !this.flowActive;
        document.getElementById('flowBtn').classList.toggle('active', this.flowActive);
        if (this.links) {
            if (this.flowActive) {
                this.links.style('stroke-dasharray', '8,4').style('animation', 'dash 0.5s linear infinite');
            } else {
                this.links.style('stroke-dasharray', null).style('animation', null);
            }
        }
    }

    // === Memory Visualization ===

    setMemoryView(mode) {
        this.memoryViewMode = mode;

        // Update tabs
        document.querySelectorAll('.mem-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.view === mode);
        });

        // Update legends
        document.getElementById('memory-legend-lifetime').style.display = mode === 'lifetime' ? 'flex' : 'none';
        document.getElementById('memory-legend-layout').style.display = mode === 'layout' ? 'flex' : 'none';

        this.renderMemoryChart();
    }

    renderMemoryChart() {
        if (this.memoryViewMode === 'layout') {
            this.renderMemoryLayoutChart();
        } else {
            this.renderMemoryLifetimeChart();
        }
    }

    renderMemoryLifetimeChart() {
        if (!this.memoryData || this.memoryData.error) {
            document.getElementById('memory-chart').innerHTML =
                `<div class="no-model"><div>${this.memoryData?.error || 'Memory analysis not available'}</div></div>`;
            return;
        }

        const peakEl = document.getElementById('mem-peak');
        peakEl.textContent = this.memoryData.summary.peak_min_kb.toFixed(1);
        document.getElementById('mem-savings').textContent = this.memoryData.summary.savings_pct.toFixed(1);

        // Update constraint status display
        const statusEl = document.getElementById('constraint-status');
        const constraint = this.memoryData.constraint;
        if (constraint && constraint.limit_kb) {
            if (constraint.fits_in_limit) {
                statusEl.textContent = 'Fits';
                statusEl.className = 'constraint-status fits';
                peakEl.style.color = '#10b981';
            } else {
                statusEl.textContent = `+${constraint.overflow_kb.toFixed(1)}KB`;
                statusEl.className = 'constraint-status exceeded';
                peakEl.style.color = '#ef4444';
            }
        } else {
            statusEl.textContent = '';
            statusEl.className = 'constraint-status';
            peakEl.style.color = '';
        }

        const tensors = this.memoryData.tensors;
        if (!tensors || tensors.length === 0) {
            document.getElementById('memory-chart').innerHTML = '<div class="no-model"><div>No tensor data</div></div>';
            return;
        }

        tensors.sort((a, b) => a.birth - b.birth);
        const maxStep = Math.max(...tensors.map(t => t.death)) + 1;

        // Build memory usage timeline for stacked area chart
        const timeline = this.memoryData.timeline || [];
        let memoryTimeline = [];
        if (timeline.length > 0) {
            memoryTimeline = timeline.map(t => ({
                step: t.step,
                memory_kb: (t.max_memory || 0) / 1024
            }));
        }

        const traces = tensors.map((t) => {
            let color;
            const isHighlighted = this.selectedNode && t.name === this.selectedNode.name;

            if (isHighlighted) {
                color = '#fbbf24';
            } else if (t.is_input) {
                color = this.selectedNode ? 'rgba(16, 185, 129, 0.4)' : '#10b981';
            } else if (t.is_output) {
                color = this.selectedNode ? 'rgba(245, 158, 11, 0.4)' : '#f59e0b';
            } else if (t.reused_from) {
                color = this.selectedNode ? 'rgba(139, 92, 246, 0.4)' : '#8b5cf6';
            } else {
                color = this.selectedNode ? 'rgba(59, 130, 246, 0.4)' : '#3b82f6';
            }

            return {
                x: [t.death - t.birth + 1],
                y: [t.name],
                type: 'bar',
                orientation: 'h',
                base: [t.birth],
                marker: {
                    color: color,
                    line: {
                        color: isHighlighted ? '#fbbf24' : 'rgba(255,255,255,0.2)',
                        width: isHighlighted ? 2 : 1
                    }
                },
                hovertemplate: `<b>${t.name}</b><br>Shape: ${JSON.stringify(t.shape)}<br>Size: ${t.size_kb.toFixed(2)} KB<br>Lifetime: Step ${t.birth} - ${t.death}<extra></extra>`,
                showlegend: false,
                customdata: [t],
            };
        });

        const layout = {
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            margin: { t: 10, r: 20, b: 30, l: 120 },
            xaxis: {
                title: { text: 'Execution Step', font: { size: 11, color: '#94a3b8' } },
                color: '#94a3b8',
                gridcolor: 'rgba(255,255,255,0.05)',
                range: [-0.5, maxStep + 0.5],
            },
            yaxis: {
                color: '#94a3b8',
                gridcolor: 'rgba(255,255,255,0.03)',
                tickfont: { size: 10 },
            },
            barmode: 'overlay',
            hovermode: 'closest',
            shapes: [],
        };

        // Add memory limit line if enabled
        const limitInput = document.getElementById('memory-limit');
        const showLimit = document.getElementById('show-limit');
        if (showLimit && showLimit.checked && limitInput && limitInput.value) {
            const limitKB = parseFloat(limitInput.value);
            if (limitKB > 0) {
                // For lifetime chart, we show peak memory vs limit
                const peakKB = this.memoryData.summary.peak_min_kb;
                const exceedsLimit = peakKB > limitKB;

                // Update the peak display color based on limit
                const peakEl = document.getElementById('mem-peak');
                if (peakEl) {
                    peakEl.style.color = exceedsLimit ? '#ef4444' : '#10b981';
                }
            }
        }

        Plotly.newPlot('memory-chart', traces, layout, {
            responsive: true,
            displayModeBar: false,
        }).then(() => {
            // Click handler for memory blocks (must be after Plotly.newPlot completes)
            const chartEl = document.getElementById('memory-chart');
            chartEl.on('plotly_click', (eventData) => {
                const tensor = eventData.points[0].customdata;
                if (tensor) {
                    this.highlightGraphNode(tensor.name);
                }
            });
        });
    }

    renderMemoryLayoutChart() {
        // Memory layout view: Y-axis = address, X-axis = TID (execution step)
        if (!this.memoryData || this.memoryData.error) {
            document.getElementById('memory-chart').innerHTML =
                `<div class="no-model"><div>${this.memoryData?.error || 'Memory analysis not available'}</div></div>`;
            return;
        }

        const tensors = this.memoryData.tensors;
        if (!tensors || tensors.length === 0) {
            document.getElementById('memory-chart').innerHTML = '<div class="no-model"><div>No tensor data</div></div>';
            return;
        }

        // Use real memory_offset from backend if available, otherwise compute
        const allocations = tensors.map(t => {
            const sizeBytes = t.size_bytes || (t.size_kb * 1024);
            // Use backend-computed offset if available (>= 0), otherwise use birth order
            const hasRealOffset = t.memory_offset !== undefined && t.memory_offset >= 0;
            return {
                ...t,
                address: hasRealOffset ? t.memory_offset : null,
                size: sizeBytes,
                tid: t.birth,
            };
        });

        // If no real offsets, compute simulated layout for display
        if (allocations.some(a => a.address === null)) {
            // Sort by birth time, then use simple stacking
            allocations.sort((a, b) => a.birth - b.birth);
            let currentAddress = 0;
            allocations.forEach(a => {
                if (a.address === null) {
                    a.address = currentAddress;
                    currentAddress += a.size;
                }
            });
        }

        // Create traces for the layout view
        const traces = allocations.map((t) => {
            let color;
            let lineColor;
            const isHighlighted = this.selectedNode && t.name === this.selectedNode.name;

            if (isHighlighted) {
                color = '#fbbf24';
                lineColor = '#fbbf24';
            } else if (t.reused_from) {
                // Memory reuse - purple
                color = this.selectedNode ? 'rgba(139, 92, 246, 0.4)' : '#8b5cf6';
                lineColor = 'rgba(139, 92, 246, 0.8)';
            } else if (t.is_input) {
                color = this.selectedNode ? 'rgba(16, 185, 129, 0.4)' : '#10b981';
                lineColor = 'rgba(255,255,255,0.3)';
            } else if (t.is_output) {
                color = this.selectedNode ? 'rgba(245, 158, 11, 0.4)' : '#f59e0b';
                lineColor = 'rgba(255,255,255,0.3)';
            } else {
                color = this.selectedNode ? 'rgba(59, 130, 246, 0.4)' : '#3b82f6';
                lineColor = 'rgba(255,255,255,0.3)';
            }

            const addrStart = t.address;
            const addrEnd = t.address + t.size;
            const reusedInfo = t.reused_from ? `<br>Reused from: ${t.reused_from}` : '';

            return {
                x: [t.tid, t.tid + 1, t.tid + 1, t.tid, t.tid],  // Rectangle vertices
                y: [addrStart, addrStart, addrEnd, addrEnd, addrStart],
                type: 'scatter',
                mode: 'lines',
                fill: 'toself',
                fillcolor: color,
                line: {
                    color: isHighlighted ? '#fbbf24' : lineColor,
                    width: isHighlighted ? 2 : 1
                },
                hovertemplate: `<b>${t.name}</b><br>Address: 0x${addrStart.toString(16)} - 0x${addrEnd.toString(16)}<br>Size: ${t.size_kb.toFixed(2)} KB<br>TID: ${t.tid}${reusedInfo}<extra></extra>`,
                showlegend: false,
                customdata: t,
            };
        });

        const maxAddress = Math.max(...allocations.map(t => t.address + t.size));
        const maxTid = Math.max(...allocations.map(t => t.tid)) + 2;

        // Check for memory limit
        const limitInput = document.getElementById('memory-limit');
        const showLimit = document.getElementById('show-limit');
        let limitBytes = null;
        if (showLimit && showLimit.checked && limitInput && limitInput.value) {
            limitBytes = parseFloat(limitInput.value) * 1024;  // KB to bytes
        }

        const yMax = limitBytes ? Math.max(maxAddress * 1.05, limitBytes * 1.1) : maxAddress * 1.05;

        const layout = {
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            margin: { t: 10, r: 20, b: 30, l: 80 },
            xaxis: {
                title: { text: 'TID (Execution Order)', font: { size: 11, color: '#94a3b8' } },
                color: '#94a3b8',
                gridcolor: 'rgba(255,255,255,0.05)',
                range: [-0.5, maxTid],
                dtick: 1,
            },
            yaxis: {
                title: { text: 'Address (bytes)', font: { size: 11, color: '#94a3b8' } },
                color: '#94a3b8',
                gridcolor: 'rgba(255,255,255,0.05)',
                range: [0, yMax],
                tickformat: '.2s',
            },
            hovermode: 'closest',
            shapes: [],
            annotations: [],
        };

        // Add memory limit line if enabled
        if (limitBytes) {
            layout.shapes.push({
                type: 'line',
                x0: -0.5,
                x1: maxTid,
                y0: limitBytes,
                y1: limitBytes,
                line: {
                    color: '#ef4444',
                    width: 2,
                    dash: 'dash'
                }
            });
            layout.annotations.push({
                x: maxTid - 0.5,
                y: limitBytes,
                xanchor: 'right',
                yanchor: 'bottom',
                text: `Limit: ${(limitBytes/1024).toFixed(1)} KB`,
                showarrow: false,
                font: { size: 10, color: '#ef4444' }
            });

            // Check if any allocation exceeds limit
            const exceedsLimit = maxAddress > limitBytes;
            const peakEl = document.getElementById('mem-peak');
            if (peakEl) {
                peakEl.style.color = exceedsLimit ? '#ef4444' : '#10b981';
            }
        }

        Plotly.newPlot('memory-chart', traces, layout, {
            responsive: true,
            displayModeBar: false,
        }).then(() => {
            const chartEl = document.getElementById('memory-chart');
            chartEl.on('plotly_click', (eventData) => {
                const tensor = eventData.points[0].customdata;
                if (tensor && tensor.name) {
                    this.highlightGraphNode(tensor.name);
                }
            });
        });
    }

    highlightGraphNode(tensorName) {
        d3.selectAll('.node').classed('highlighted', false).classed('dimmed', false);

        const matchingNode = this.graphData.nodes.find(n => n.name === tensorName);
        if (matchingNode) {
            d3.selectAll('.node').classed('dimmed', true);
            d3.select(`.node[data-name="${tensorName}"]`)
                .classed('highlighted', true)
                .classed('dimmed', false);

            this.links.classed('highlighted', l => l.source === matchingNode.id || l.target === matchingNode.id);

            // Pan to node
            const node = this.nodeById[matchingNode.id];
            if (node && node.x !== undefined) {
                const width = this.container.clientWidth;
                const height = this.container.clientHeight;
                const currentTransform = d3.zoomTransform(this.svg.node());
                const targetX = width / 2 - node.x * currentTransform.k - node.width / 2 * currentTransform.k;
                const targetY = height / 2 - node.y * currentTransform.k - node.height / 2 * currentTransform.k;
                this.svg.transition().duration(300).call(
                    this.zoom.transform,
                    d3.zoomIdentity.translate(targetX, targetY).scale(currentTransform.k)
                );
            }
        }
    }
}

// Initialize app
const app = new VisualizerApp();
