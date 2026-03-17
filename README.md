# SmartNet Playground

Interactive neural network playground — build, train & visualize in your browser.

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

---

## What is this?

A lightweight web app where you can **drag-and-drop neural network layers**, **train on toy datasets**, and **watch the model learn in real-time** — all running on CPU.

**Features at a glance:**

- **Visual network builder** — click to add Linear, ReLU, Tanh, Dropout, BatchNorm layers; SVG graph auto-updates
- **Real training** — actual PyTorch backprop on CPU, streamed to your browser via Server-Sent Events
- **Live charts** — loss & accuracy curves update every epoch (Chart.js)
- **Decision boundary** — for 2D datasets, watch the classification boundary evolve during training
- **5 toy datasets** — XOR, Circles, Spiral, Moons, Gaussian 4D
- **5 one-click presets** — pre-configured networks that just work
- **Dark modern UI** — glass-morphism panels, SVG data-flow animation

Designed to run on tiny machines (t3a.micro, 1 GB RAM).

## Quick Start

### Docker (recommended)

```bash
docker build -t smartnet .
docker run -p 5000:5000 smartnet
```

Open [http://localhost:5000](http://localhost:5000).

### Local

```bash
# CPU-only PyTorch (much smaller than CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install flask

python web_app/app.py
```

## How to Use

1. **Pick a preset** (top bar) or build your own network
2. **Add layers** from the left panel — shapes are inferred automatically
3. **Choose a dataset** and tweak epochs / learning rate on the right
4. **Click Start Training** — watch loss drop, accuracy rise, and decision boundary form

## Architecture

```
web_app/
├── app.py              Flask backend — datasets, model builder, training loop, SSE
├── templates/
│   └── index.html      Single-page UI
└── static/
    ├── css/style.css    Dark theme
    └── js/app.js        Network SVG, Chart.js, layer management
```

| Component | Detail |
|-----------|--------|
| Backend | Flask + PyTorch CPU, threaded training with SSE streaming |
| Datasets | XOR, Circles, Spiral, Moons, Gaussian — all generated with `torch` |
| Models | `nn.Sequential` built from user config, max 8 layers / 128 units / 100K params |
| Visualization | SVG network graph, Chart.js dual-axis chart, Canvas decision boundary |

## Resource Usage

| Metric | Value |
|--------|-------|
| Runtime memory | ~220 MB (CPU-only torch) |
| Docker image | ~1.2 GB |
| Training speed | 2-10 seconds per preset |
| Concurrent sessions | 2 max (server-enforced) |

Fits comfortably on **t3a.micro** (1 GB RAM, 2 vCPU).

## License

[Apache 2.0](LICENSE)
