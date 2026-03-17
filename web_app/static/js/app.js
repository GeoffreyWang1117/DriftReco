/* ========== SmartNet Playground — Main Application ========== */

const LAYER_COLORS = {
    linear:     '#3b82f6',
    relu:       '#22c55e',
    sigmoid:    '#f59e0b',
    tanh:       '#a855f7',
    leaky_relu: '#14b8a6',
    dropout:    '#64748b',
    batchnorm:  '#06b6d4'
};

const LAYER_LABELS = {
    linear:     'Linear',
    relu:       'ReLU',
    sigmoid:    'Sigmoid',
    tanh:       'Tanh',
    leaky_relu: 'LeakyReLU',
    dropout:    'Dropout',
    batchnorm:  'BatchNorm'
};

const PRESETS = {
    xor: {
        dataset: 'xor',
        epochs: 150,
        lr: -1.5,
        layers: [
            { type: 'linear', out_features: 16 },
            { type: 'relu' },
            { type: 'linear', out_features: 16 },
            { type: 'relu' },
            { type: 'linear', out_features: 2 }
        ]
    },
    circles: {
        dataset: 'circles',
        epochs: 200,
        lr: -2,
        layers: [
            { type: 'linear', out_features: 32 },
            { type: 'relu' },
            { type: 'linear', out_features: 16 },
            { type: 'relu' },
            { type: 'linear', out_features: 2 }
        ]
    },
    spiral: {
        dataset: 'spiral',
        epochs: 300,
        lr: -2,
        layers: [
            { type: 'linear', out_features: 64 },
            { type: 'relu' },
            { type: 'linear', out_features: 64 },
            { type: 'relu' },
            { type: 'linear', out_features: 32 },
            { type: 'relu' },
            { type: 'linear', out_features: 3 }
        ]
    },
    moons: {
        dataset: 'moons',
        epochs: 150,
        lr: -2,
        layers: [
            { type: 'linear', out_features: 32 },
            { type: 'relu' },
            { type: 'linear', out_features: 16 },
            { type: 'relu' },
            { type: 'linear', out_features: 2 }
        ]
    },
    gauss: {
        dataset: 'gauss',
        epochs: 200,
        lr: -2.5,
        layers: [
            { type: 'linear', out_features: 32 },
            { type: 'relu' },
            { type: 'batchnorm' },
            { type: 'linear', out_features: 16 },
            { type: 'relu' },
            { type: 'linear', out_features: 3 }
        ]
    }
};

const DATASET_META = {
    xor:     { features: 2, classes: 2, is_2d: true },
    circles: { features: 2, classes: 2, is_2d: true },
    spiral:  { features: 2, classes: 3, is_2d: true },
    moons:   { features: 2, classes: 2, is_2d: true },
    gauss:   { features: 4, classes: 3, is_2d: false }
};

/* ========== Playground Class ========== */

class SmartNetPlayground {
    constructor() {
        this.layers = [];
        this.selectedIndex = -1;
        this.chart = null;
        this.eventSource = null;
        this.isTraining = false;
        this.dataPoints = null;
        this.flowAnimId = null;
        this.idCounter = 0;
    }

    /* ---------- Lifecycle ---------- */

    init() {
        this.initChart();
        this.initListeners();
        this.updateDatasetInfo();
        this.render();
        this.setStatus('Ready');
    }

    /* ---------- Listeners ---------- */

    initListeners() {
        // Palette buttons — add layer
        document.querySelectorAll('.pal-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const type = btn.getAttribute('data-type');
                if (type) this.addLayer(type);
            });
        });

        // Preset buttons
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const name = btn.getAttribute('data-preset');
                if (name) this.loadPreset(name);
            });
        });

        // Reset
        const resetBtn = document.getElementById('resetBtn');
        if (resetBtn) resetBtn.addEventListener('click', () => this.reset());

        // Train / Stop
        const trainBtn = document.getElementById('trainBtn');
        if (trainBtn) trainBtn.addEventListener('click', () => this.startTraining());

        const stopBtn = document.getElementById('stopBtn');
        if (stopBtn) stopBtn.addEventListener('click', () => this.stopTraining());

        // Dataset select
        const ds = document.getElementById('datasetSelect');
        if (ds) {
            ds.addEventListener('change', () => {
                this.updateDatasetInfo();
                this.inferShapes();
                this.render();
            });
        }

        // Epochs slider
        const epochsSlider = document.getElementById('epochsSlider');
        const epochsVal = document.getElementById('epochsVal');
        if (epochsSlider && epochsVal) {
            epochsSlider.addEventListener('input', () => {
                epochsVal.textContent = epochsSlider.value;
            });
        }

        // LR slider
        const lrSlider = document.getElementById('lrSlider');
        const lrVal = document.getElementById('lrVal');
        if (lrSlider && lrVal) {
            lrSlider.addEventListener('input', () => {
                lrVal.textContent = Math.pow(10, parseFloat(lrSlider.value)).toFixed(4);
            });
        }

        // Click on SVG to deselect
        const svgContainer = document.getElementById('svgContainer');
        if (svgContainer) {
            svgContainer.addEventListener('click', (e) => {
                if (e.target === svgContainer || e.target.id === 'networkSvg') {
                    this.selectLayer(-1);
                }
            });
        }
    }

    /* ---------- Layer Management ---------- */

    addLayer(type) {
        if (this.layers.length >= 8) {
            this.setStatus('Max 8 layers allowed', true);
            return;
        }

        const layer = { id: ++this.idCounter, type: type };

        if (type === 'linear') {
            layer.out_features = 16;
        } else if (type === 'dropout') {
            layer.rate = 0.2;
        }

        this.layers.push(layer);
        this.inferShapes();
        this.render();
        this.setStatus('Added ' + LAYER_LABELS[type]);
    }

    removeLayer(index) {
        if (index < 0 || index >= this.layers.length) return;
        this.layers.splice(index, 1);
        if (this.selectedIndex === index) {
            this.selectedIndex = -1;
        } else if (this.selectedIndex > index) {
            this.selectedIndex--;
        }
        this.inferShapes();
        this.render();
    }

    selectLayer(index) {
        this.selectedIndex = index;
        this.renderLayerList();
        this.renderProperties();
        this.renderSVG();
    }

    updateLayerProp(index, key, value) {
        if (index < 0 || index >= this.layers.length) return;
        this.layers[index][key] = value;
        this.inferShapes();
        this.render();
    }

    /* ---------- Shape Inference ---------- */

    inferShapes() {
        let dim = this.getInputDim();
        const outputClasses = this.getOutputClasses();

        for (let i = 0; i < this.layers.length; i++) {
            const l = this.layers[i];
            if (l.type === 'linear') {
                l.in_features = dim;
                dim = l.out_features;
            } else if (l.type === 'batchnorm') {
                l.num_features = dim;
            }
            // activations and dropout pass dim through unchanged
        }

        // Update param badge
        const badge = document.getElementById('paramBadge');
        if (badge) {
            let params = 0;
            for (const l of this.layers) {
                if (l.type === 'linear') {
                    params += l.in_features * l.out_features + l.out_features;
                } else if (l.type === 'batchnorm') {
                    params += l.num_features * 2;
                }
            }
            badge.textContent = params.toLocaleString() + ' params';
        }

        // Enable / disable train button
        const trainBtn = document.getElementById('trainBtn');
        if (trainBtn) {
            const lastLinear = [...this.layers].reverse().find(l => l.type === 'linear');
            const valid = this.layers.length > 0 &&
                          lastLinear &&
                          lastLinear.out_features === outputClasses;
            trainBtn.disabled = !valid || this.isTraining;
        }
    }

    getInputDim() {
        const ds = document.getElementById('datasetSelect');
        const name = ds ? ds.value : 'xor';
        return DATASET_META[name] ? DATASET_META[name].features : 2;
    }

    getOutputClasses() {
        const ds = document.getElementById('datasetSelect');
        const name = ds ? ds.value : 'xor';
        return DATASET_META[name] ? DATASET_META[name].classes : 2;
    }

    /* ---------- Render Orchestration ---------- */

    render() {
        this.renderLayerList();
        this.renderProperties();
        this.renderSVG();
    }

    /* ---------- Layer List ---------- */

    renderLayerList() {
        const list = document.getElementById('layerList');
        const hint = document.getElementById('emptyHint');
        if (!list) return;

        if (this.layers.length === 0) {
            list.innerHTML = '';
            if (hint) {
                hint.style.display = '';
                list.appendChild(hint);
            }
            return;
        }

        if (hint) hint.style.display = 'none';

        let html = '';
        for (let i = 0; i < this.layers.length; i++) {
            const l = this.layers[i];
            const color = LAYER_COLORS[l.type] || '#64748b';
            const label = LAYER_LABELS[l.type] || l.type;
            const sel = i === this.selectedIndex ? ' selected' : '';

            let dimStr = '';
            if (l.type === 'linear') {
                dimStr = l.in_features + '\u2192' + l.out_features;
            } else if (l.type === 'dropout') {
                dimStr = 'p=' + l.rate;
            } else if (l.type === 'batchnorm') {
                dimStr = l.num_features;
            }

            html += '<div class="layer-item' + sel + '" data-index="' + i + '">' +
                '<span class="layer-dot" style="background:' + color + '"></span>' +
                '<span class="layer-name">' + label + '</span>' +
                '<span class="layer-dim">' + dimStr + '</span>' +
                '<button class="layer-del" data-delidx="' + i + '" title="Remove">' +
                '<i class="fas fa-times"></i></button>' +
                '</div>';
        }
        list.innerHTML = html;

        // Event delegation
        list.onclick = (e) => {
            const delBtn = e.target.closest('[data-delidx]');
            if (delBtn) {
                e.stopPropagation();
                this.removeLayer(parseInt(delBtn.getAttribute('data-delidx')));
                return;
            }
            const item = e.target.closest('.layer-item');
            if (item) {
                this.selectLayer(parseInt(item.getAttribute('data-index')));
            }
        };
    }

    /* ---------- Properties Panel ---------- */

    renderProperties() {
        const sec = document.getElementById('propsSection');
        const content = document.getElementById('propsContent');
        if (!sec || !content) return;

        if (this.selectedIndex < 0 || this.selectedIndex >= this.layers.length) {
            sec.style.display = 'none';
            return;
        }

        sec.style.display = '';
        const l = this.layers[this.selectedIndex];
        let html = '';

        if (l.type === 'linear') {
            html += '<div class="prop-row"><label>Input</label>' +
                '<input type="number" value="' + l.in_features + '" readonly></div>';
            html += '<div class="prop-row"><label>Output</label>' +
                '<input type="number" id="propOut" value="' + l.out_features + '" min="1" max="128"></div>';
        } else if (l.type === 'dropout') {
            html += '<div class="prop-row"><label>Rate</label>' +
                '<input type="number" id="propRate" value="' + l.rate + '" min="0" max="0.9" step="0.05"></div>';
        } else {
            html += '<div style="font-size:0.78rem;color:var(--text3);">No configurable parameters</div>';
        }

        content.innerHTML = html;

        // Bind change
        const propOut = document.getElementById('propOut');
        if (propOut) {
            propOut.addEventListener('change', () => {
                let v = parseInt(propOut.value);
                if (isNaN(v) || v < 1) v = 1;
                if (v > 128) v = 128;
                propOut.value = v;
                this.updateLayerProp(this.selectedIndex, 'out_features', v);
            });
        }

        const propRate = document.getElementById('propRate');
        if (propRate) {
            propRate.addEventListener('change', () => {
                let v = parseFloat(propRate.value);
                if (isNaN(v) || v < 0) v = 0;
                if (v > 0.9) v = 0.9;
                propRate.value = v;
                this.updateLayerProp(this.selectedIndex, 'rate', v);
            });
        }
    }

    /* ---------- SVG Rendering ---------- */

    renderSVG() {
        const svg = document.getElementById('networkSvg');
        const emptyEl = document.getElementById('svgEmpty');
        if (!svg) return;

        // Clear everything except <defs>
        const defs = svg.querySelector('defs');
        svg.innerHTML = '';
        if (defs) svg.appendChild(defs);

        if (this.layers.length === 0) {
            if (emptyEl) emptyEl.style.display = '';
            svg.setAttribute('height', '100%');
            return;
        }
        if (emptyEl) emptyEl.style.display = 'none';

        const inputDim = this.getInputDim();
        const outputClasses = this.getOutputClasses();

        const nodeW = 130;
        const nodeH = 52;
        const gapY = 28;
        const cx = 200;

        const totalNodes = this.layers.length + 2; // input + layers + output
        const totalH = totalNodes * (nodeH + gapY) + 20;
        svg.setAttribute('height', totalH);
        svg.setAttribute('viewBox', '0 0 400 ' + totalH);

        let y = 16;

        // Input node
        const inputColor = '#38bdf8';
        this._svgNode(svg, cx, y, nodeW, nodeH, 'Input', inputDim + ' features', inputColor, -1, false);
        let prevY = y + nodeH;
        y += nodeH + gapY;

        // Layer nodes
        let dim = inputDim;
        for (let i = 0; i < this.layers.length; i++) {
            const l = this.layers[i];
            const color = LAYER_COLORS[l.type] || '#64748b';
            const label = LAYER_LABELS[l.type] || l.type;
            const sel = i === this.selectedIndex;

            let sub = '';
            if (l.type === 'linear') {
                sub = l.in_features + ' \u2192 ' + l.out_features;
                dim = l.out_features;
            } else if (l.type === 'dropout') {
                sub = 'rate = ' + l.rate;
            } else if (l.type === 'batchnorm') {
                sub = 'features = ' + l.num_features;
            }

            // Connection from previous
            let connDim = '';
            if (l.type === 'linear') {
                connDim = l.in_features.toString();
            } else {
                connDim = dim.toString();
            }
            this._svgConn(svg, cx, prevY, y, connDim);

            // Flow particles during training
            if (this.isTraining) {
                this._svgFlowParticles(svg, cx, prevY, y, this.layers.length);
            }

            this._svgNode(svg, cx, y, nodeW, nodeH, label, sub, color, i, sel);
            prevY = y + nodeH;
            y += nodeH + gapY;
        }

        // Output node
        const lastLinear = [...this.layers].reverse().find(l => l.type === 'linear');
        const outDim = lastLinear ? lastLinear.out_features : dim;
        const mismatch = outDim !== outputClasses;
        const outColor = mismatch ? '#fb7185' : '#34d399';
        this._svgConn(svg, cx, prevY, y, outDim.toString());

        if (this.isTraining) {
            this._svgFlowParticles(svg, cx, prevY, y, this.layers.length);
        }

        const outSub = outputClasses + ' classes' + (mismatch ? ' (MISMATCH!)' : '');
        this._svgNode(svg, cx, y, nodeW, nodeH, 'Output', outSub, outColor, -2, false);
    }

    _svgNode(svg, cx, y, w, h, label, sub, color, idx, selected) {
        const ns = 'http://www.w3.org/2000/svg';
        const g = document.createElementNS(ns, 'g');
        g.setAttribute('class', 'svg-layer-group' + (selected ? ' selected' : ''));

        const rx = 10;
        const x = cx - w / 2;

        const rect = document.createElementNS(ns, 'rect');
        rect.setAttribute('x', x);
        rect.setAttribute('y', y);
        rect.setAttribute('width', w);
        rect.setAttribute('height', h);
        rect.setAttribute('rx', rx);
        rect.setAttribute('fill', color + '22');
        rect.setAttribute('stroke', color);
        rect.setAttribute('stroke-width', selected ? '2.5' : '1.5');
        rect.setAttribute('class', 'svg-layer-rect');
        if (selected) {
            rect.setAttribute('filter', 'url(#glow-sm)');
        }
        g.appendChild(rect);

        const txt = document.createElementNS(ns, 'text');
        txt.setAttribute('x', cx);
        txt.setAttribute('y', y + 22);
        txt.setAttribute('text-anchor', 'middle');
        txt.setAttribute('class', 'svg-layer-text');
        txt.textContent = label;
        g.appendChild(txt);

        if (sub) {
            const stxt = document.createElementNS(ns, 'text');
            stxt.setAttribute('x', cx);
            stxt.setAttribute('y', y + 38);
            stxt.setAttribute('text-anchor', 'middle');
            stxt.setAttribute('class', 'svg-layer-sub');
            stxt.textContent = sub;
            g.appendChild(stxt);
        }

        // Click handler for layer nodes
        if (idx >= 0) {
            g.style.cursor = 'pointer';
            g.addEventListener('click', (e) => {
                e.stopPropagation();
                this.selectLayer(idx);
            });
        }

        svg.appendChild(g);
    }

    _svgConn(svg, cx, y1, y2, dim) {
        const ns = 'http://www.w3.org/2000/svg';

        const line = document.createElementNS(ns, 'line');
        line.setAttribute('x1', cx);
        line.setAttribute('y1', y1);
        line.setAttribute('x2', cx);
        line.setAttribute('y2', y2);
        line.setAttribute('stroke', 'url(#connGrad)');
        line.setAttribute('stroke-width', '1.5');
        line.setAttribute('stroke-dasharray', '4 3');
        svg.appendChild(line);

        if (dim) {
            const txt = document.createElementNS(ns, 'text');
            txt.setAttribute('x', cx + 12);
            txt.setAttribute('y', (y1 + y2) / 2 + 3);
            txt.setAttribute('class', 'svg-dim-text');
            txt.textContent = dim;
            svg.appendChild(txt);
        }
    }

    _svgFlowParticles(svg, cx, topY, bottomY, nSegments) {
        const ns = 'http://www.w3.org/2000/svg';
        const count = 5;
        const travel = bottomY - topY;

        for (let p = 0; p < count; p++) {
            const circle = document.createElementNS(ns, 'circle');
            circle.setAttribute('cx', cx);
            circle.setAttribute('r', '2.5');
            circle.setAttribute('fill', '#38bdf8');
            circle.setAttribute('opacity', '0.7');

            const anim = document.createElementNS(ns, 'animate');
            anim.setAttribute('attributeName', 'cy');
            anim.setAttribute('from', topY);
            anim.setAttribute('to', bottomY);
            anim.setAttribute('dur', (1.2 + nSegments * 0.1) + 's');
            anim.setAttribute('begin', (p * 0.25) + 's');
            anim.setAttribute('repeatCount', 'indefinite');
            circle.appendChild(anim);

            const animOp = document.createElementNS(ns, 'animate');
            animOp.setAttribute('attributeName', 'opacity');
            animOp.setAttribute('values', '0;0.8;0.8;0');
            animOp.setAttribute('dur', (1.2 + nSegments * 0.1) + 's');
            animOp.setAttribute('begin', (p * 0.25) + 's');
            animOp.setAttribute('repeatCount', 'indefinite');
            circle.appendChild(animOp);

            svg.appendChild(circle);
        }
    }

    /* ---------- Chart.js ---------- */

    initChart() {
        const canvas = document.getElementById('chartCanvas');
        if (!canvas) return;

        this.chart = new Chart(canvas, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Loss',
                        data: [],
                        borderColor: '#fb7185',
                        backgroundColor: 'rgba(251,113,133,0.1)',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        tension: 0.3,
                        yAxisID: 'yLoss'
                    },
                    {
                        label: 'Accuracy',
                        data: [],
                        borderColor: '#34d399',
                        backgroundColor: 'rgba(52,211,153,0.1)',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        tension: 0.3,
                        yAxisID: 'yAcc'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: '#94a3b8',
                            font: { size: 10 },
                            boxWidth: 12,
                            padding: 8
                        }
                    }
                },
                scales: {
                    x: {
                        display: true,
                        ticks: { color: '#64748b', font: { size: 9 }, maxTicksLimit: 8 },
                        grid: { color: 'rgba(148,163,184,0.06)' }
                    },
                    yLoss: {
                        type: 'linear',
                        position: 'left',
                        ticks: { color: '#fb7185', font: { size: 9 }, maxTicksLimit: 5 },
                        grid: { color: 'rgba(148,163,184,0.06)' },
                        title: { display: false }
                    },
                    yAcc: {
                        type: 'linear',
                        position: 'right',
                        min: 0,
                        max: 1,
                        ticks: {
                            color: '#34d399',
                            font: { size: 9 },
                            maxTicksLimit: 5,
                            callback: function(v) { return (v * 100).toFixed(0) + '%'; }
                        },
                        grid: { drawOnChartArea: false }
                    }
                }
            }
        });
    }

    resetChart() {
        if (!this.chart) return;
        this.chart.data.labels = [];
        this.chart.data.datasets[0].data = [];
        this.chart.data.datasets[1].data = [];
        this.chart.update();

        const lossVal = document.getElementById('lossVal');
        const accVal = document.getElementById('accVal');
        const epochVal = document.getElementById('epochVal');
        if (lossVal) lossVal.innerHTML = '&mdash;';
        if (accVal) accVal.innerHTML = '&mdash;';
        if (epochVal) epochVal.innerHTML = '&mdash;';
    }

    pushChartData(epoch, loss, accuracy) {
        if (!this.chart) return;

        this.chart.data.labels.push(epoch);
        this.chart.data.datasets[0].data.push(loss);
        this.chart.data.datasets[1].data.push(accuracy);

        // Downsample if too many points
        if (this.chart.data.labels.length > 200) {
            const labels = this.chart.data.labels;
            const d0 = this.chart.data.datasets[0].data;
            const d1 = this.chart.data.datasets[1].data;
            const newLabels = [];
            const newD0 = [];
            const newD1 = [];
            for (let i = 0; i < labels.length; i += 2) {
                newLabels.push(labels[i]);
                newD0.push(d0[i]);
                newD1.push(d1[i]);
            }
            this.chart.data.labels = newLabels;
            this.chart.data.datasets[0].data = newD0;
            this.chart.data.datasets[1].data = newD1;
        }

        this.chart.update();

        // Update stat boxes
        const lossVal = document.getElementById('lossVal');
        const accVal = document.getElementById('accVal');
        const epochVal = document.getElementById('epochVal');
        if (lossVal) lossVal.textContent = loss.toFixed(4);
        if (accVal) accVal.textContent = (accuracy * 100).toFixed(1) + '%';
        if (epochVal) epochVal.textContent = epoch;
    }

    /* ---------- Decision Boundary ---------- */

    renderBoundary(boundary) {
        const canvas = document.getElementById('boundaryCanvas');
        const section = document.getElementById('boundarySection');
        if (!canvas || !section) return;

        section.style.display = '';

        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;
        const gs = boundary.grid_size;
        const cellW = w / gs;
        const cellH = h / gs;

        const classColors = [
            [59, 130, 246],   // blue
            [239, 68, 68],    // red
            [34, 197, 94],    // green
            [168, 85, 247],   // purple
            [245, 158, 11]    // amber
        ];

        // Draw grid cells
        for (let i = 0; i < gs; i++) {
            for (let j = 0; j < gs; j++) {
                const cls = boundary.predictions[i][j];
                const conf = boundary.confidence[i][j];
                const rgb = classColors[cls % classColors.length];
                const alpha = 0.25 + conf * 0.45;
                ctx.fillStyle = 'rgba(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ',' + alpha + ')';
                ctx.fillRect(j * cellW, i * cellH, cellW + 0.5, cellH + 0.5);
            }
        }

        // Overlay data points
        if (this.dataPoints) {
            const xr = boundary.x_range;
            const yr = boundary.y_range;
            const scaleX = w / (xr[1] - xr[0]);
            const scaleY = h / (yr[1] - yr[0]);

            const X = this.dataPoints.X;
            const yLabels = this.dataPoints.y;

            for (let k = 0; k < X.length; k++) {
                const px = (X[k][0] - xr[0]) * scaleX;
                const py = (X[k][1] - yr[0]) * scaleY;
                const cls = yLabels[k];
                const rgb = classColors[cls % classColors.length];

                ctx.beginPath();
                ctx.arc(px, py, 2.5, 0, 2 * Math.PI);
                ctx.fillStyle = 'rgb(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ')';
                ctx.fill();
                ctx.strokeStyle = 'rgba(255,255,255,0.6)';
                ctx.lineWidth = 0.5;
                ctx.stroke();
            }
        }
    }

    /* ---------- Training ---------- */

    startTraining() {
        if (this.isTraining) return;

        // Build layer config for the API
        const layers = [];
        let dim = this.getInputDim();

        for (const l of this.layers) {
            const cfg = { type: l.type };
            if (l.type === 'linear') {
                cfg.in_features = l.in_features;
                cfg.out_features = l.out_features;
                dim = l.out_features;
            } else if (l.type === 'dropout') {
                cfg.rate = l.rate;
            } else if (l.type === 'batchnorm') {
                cfg.num_features = l.num_features;
            }
            layers.push(cfg);
        }

        const ds = document.getElementById('datasetSelect');
        const epochsSlider = document.getElementById('epochsSlider');
        const lrSlider = document.getElementById('lrSlider');

        const body = {
            layers: layers,
            dataset: ds ? ds.value : 'xor',
            epochs: epochsSlider ? parseInt(epochsSlider.value) : 100,
            learning_rate: lrSlider ? Math.pow(10, parseFloat(lrSlider.value)) : 0.01
        };

        this.setStatus('Starting training...');
        this.resetChart();
        this.dataPoints = null;

        // Hide boundary for non-2D datasets
        const meta = DATASET_META[body.dataset];
        const bSec = document.getElementById('boundarySection');
        if (bSec) bSec.style.display = (meta && meta.is_2d) ? '' : 'none';

        fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        })
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                this.listenTraining(data.training_id);
            } else {
                this.setStatus(data.error || 'Training failed', true);
            }
        })
        .catch(err => {
            this.setStatus('Network error: ' + err.message, true);
        });
    }

    listenTraining(tid) {
        this.isTraining = true;
        this.updateTrainUI(true);
        this.renderSVG();

        this.eventSource = new EventSource('/api/train/stream/' + tid);
        this._trainingId = tid;

        this.eventSource.onmessage = (event) => {
            let msg;
            try {
                msg = JSON.parse(event.data);
            } catch (e) {
                return;
            }

            if (msg.type === 'data') {
                this.dataPoints = msg.data_points;
            } else if (msg.type === 'epoch') {
                this.pushChartData(msg.epoch, msg.loss, msg.accuracy);
                if (msg.boundary) {
                    this.renderBoundary(msg.boundary);
                }
                this.setStatus('Epoch ' + msg.epoch + '/' + msg.total_epochs +
                    ' — Loss: ' + msg.loss.toFixed(4) +
                    ' — Acc: ' + (msg.accuracy * 100).toFixed(1) + '%');
            } else if (msg.type === 'done') {
                this.onTrainingEnd();
                this.setStatus('Training complete');
            } else if (msg.type === 'stopped') {
                this.onTrainingEnd();
                this.setStatus('Training stopped at epoch ' + msg.epoch);
            } else if (msg.type === 'error') {
                this.onTrainingEnd();
                this.setStatus('Error: ' + msg.message, true);
            }
            // heartbeat — ignore
        };

        this.eventSource.onerror = () => {
            this.onTrainingEnd();
            this.setStatus('Stream disconnected', true);
        };
    }

    stopTraining() {
        if (!this.isTraining || !this._trainingId) return;

        fetch('/api/train/stop/' + this._trainingId, { method: 'POST' })
            .catch(() => {});
    }

    onTrainingEnd() {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
        this.isTraining = false;
        this._trainingId = null;
        this.updateTrainUI(false);
        this.renderSVG();
    }

    updateTrainUI(training) {
        const trainBtn = document.getElementById('trainBtn');
        const stopBtn = document.getElementById('stopBtn');

        if (trainBtn) {
            if (training) {
                trainBtn.classList.add('training');
                trainBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Training...';
                trainBtn.disabled = true;
            } else {
                trainBtn.classList.remove('training');
                trainBtn.innerHTML = '<i class="fas fa-play"></i> Start Training';
                this.inferShapes(); // re-evaluates disabled state
            }
        }

        if (stopBtn) {
            stopBtn.style.display = training ? '' : 'none';
        }
    }

    /* ---------- Presets ---------- */

    loadPreset(name) {
        const preset = PRESETS[name];
        if (!preset) return;

        this.layers = [];
        this.selectedIndex = -1;
        this.idCounter = 0;

        for (const def of preset.layers) {
            const layer = { id: ++this.idCounter, type: def.type };
            if (def.type === 'linear') layer.out_features = def.out_features;
            if (def.type === 'dropout') layer.rate = def.rate || 0.2;
            this.layers.push(layer);
        }

        // Set dataset
        const ds = document.getElementById('datasetSelect');
        if (ds) ds.value = preset.dataset;

        // Set sliders
        const epochsSlider = document.getElementById('epochsSlider');
        const epochsVal = document.getElementById('epochsVal');
        if (epochsSlider) {
            epochsSlider.value = preset.epochs;
            if (epochsVal) epochsVal.textContent = preset.epochs;
        }

        const lrSlider = document.getElementById('lrSlider');
        const lrVal = document.getElementById('lrVal');
        if (lrSlider) {
            lrSlider.value = preset.lr;
            if (lrVal) lrVal.textContent = Math.pow(10, preset.lr).toFixed(4);
        }

        // Highlight active preset button
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.classList.toggle('active', btn.getAttribute('data-preset') === name);
        });

        this.inferShapes();
        this.updateDatasetInfo();
        this.render();
        this.setStatus('Loaded preset: ' + name);
    }

    /* ---------- Reset ---------- */

    reset() {
        if (this.isTraining) this.stopTraining();

        this.layers = [];
        this.selectedIndex = -1;
        this.dataPoints = null;
        this.idCounter = 0;

        this.resetChart();

        // Clear boundary
        const bSec = document.getElementById('boundarySection');
        if (bSec) bSec.style.display = 'none';

        // Deselect presets
        document.querySelectorAll('.preset-btn').forEach(btn => btn.classList.remove('active'));

        this.inferShapes();
        this.updateDatasetInfo();
        this.render();
        this.setStatus('Ready');
    }

    /* ---------- Dataset Info ---------- */

    updateDatasetInfo() {
        const ds = document.getElementById('datasetSelect');
        const info = document.getElementById('datasetInfo');
        if (!ds || !info) return;

        const name = ds.value;
        const meta = DATASET_META[name];
        if (!meta) { info.textContent = ''; return; }

        info.textContent = meta.features + ' features, ' + meta.classes + ' classes' +
            (meta.is_2d ? ' (2D — boundary plot enabled)' : ' (high-D — no boundary plot)');
    }

    /* ---------- Status ---------- */

    setStatus(msg, isError) {
        const el = document.getElementById('statusText');
        if (!el) return;
        el.textContent = msg;
        el.style.color = isError ? '#fb7185' : '#34d399';
    }
}

/* ========== Bootstrap ========== */

const app = new SmartNetPlayground();

document.addEventListener('DOMContentLoaded', () => {
    app.init();
});
