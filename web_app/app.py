"""
SmartNet Playground - Lightweight Neural Network Demo
CPU-only, designed for small EC2 instances (t3a.micro: 1GB RAM, 2 vCPU)

Memory budget (CPU-only torch):
  torch import:  ~140MB
  Flask + app:   ~10MB
  1 training:    ~20MB
  Total idle:    ~160MB  |  Training peak: ~200MB
"""

from flask import Flask, render_template, request, jsonify, Response
import torch
import torch.nn as nn
import json
import threading
import time
import uuid
import queue
import logging
import gc
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Limit PyTorch threads for small machines
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

# ==================== Concurrency Guard ====================

MAX_CONCURRENT_TRAINING = 2
training_sessions = {}       # { tid: { queue, stop, thread } }
training_lock = threading.Lock()


# ==================== Toy Datasets ====================

def make_xor(n=200):
    base = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    labels = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    X = base.repeat(n // 4, 1) + torch.randn(n, 2) * 0.15
    y = labels.repeat(n // 4)
    perm = torch.randperm(n)
    return X[perm], y[perm]


def make_circles(n=300):
    half = n // 2
    theta1 = torch.linspace(0, 2 * math.pi, half)
    X1 = torch.stack([theta1.cos(), theta1.sin()], dim=1) + torch.randn(half, 2) * 0.08
    theta2 = torch.linspace(0, 2 * math.pi, half)
    X2 = 0.5 * torch.stack([theta2.cos(), theta2.sin()], dim=1) + torch.randn(half, 2) * 0.08
    X = torch.cat([X1, X2])
    y = torch.cat([torch.zeros(half, dtype=torch.long), torch.ones(half, dtype=torch.long)])
    perm = torch.randperm(n)
    return X[perm], y[perm]


def make_spiral(n_per_class=150, n_classes=3):
    X_all, y_all = [], []
    for i in range(n_classes):
        r = torch.linspace(0.2, 1.0, n_per_class)
        t = torch.linspace(i * 4.0, (i + 1) * 4.0, n_per_class) + torch.randn(n_per_class) * 0.25
        X_all.append(torch.stack([r * t.sin(), r * t.cos()], dim=1))
        y_all.append(torch.full((n_per_class,), i, dtype=torch.long))
    X = torch.cat(X_all)
    y = torch.cat(y_all)
    perm = torch.randperm(len(X))
    return X[perm], y[perm]


def make_moons(n=300):
    half = n // 2
    t1 = torch.linspace(0, math.pi, half)
    X1 = torch.stack([t1.cos(), t1.sin()], dim=1) + torch.randn(half, 2) * 0.1
    t2 = torch.linspace(0, math.pi, half)
    X2 = torch.stack([1 - t2.cos(), 0.5 - t2.sin()], dim=1) + torch.randn(half, 2) * 0.1
    X = torch.cat([X1, X2])
    y = torch.cat([torch.zeros(half, dtype=torch.long), torch.ones(half, dtype=torch.long)])
    perm = torch.randperm(n)
    return X[perm], y[perm]


def make_gaussian(n_per_class=50, n_classes=3):
    centers = [
        [0.0, 0.0, 0.0, 0.0],
        [2.0, 2.0, 1.0, 1.0],
        [1.0, 3.0, 2.5, 3.0],
    ]
    X_all, y_all = [], []
    for i in range(n_classes):
        c = torch.tensor(centers[i], dtype=torch.float32)
        X_all.append(torch.randn(n_per_class, 4) * 0.5 + c)
        y_all.append(torch.full((n_per_class,), i, dtype=torch.long))
    X = torch.cat(X_all)
    y = torch.cat(y_all)
    X = (X - X.mean(0)) / (X.std(0) + 1e-7)
    perm = torch.randperm(len(X))
    return X[perm], y[perm]


DATASET_INFO = {
    'xor':     {'name': 'XOR Gate',    'features': 2, 'classes': 2, 'samples': 200, 'is_2d': True,  'desc': 'Classic XOR — linearly inseparable'},
    'circles': {'name': 'Circles',     'features': 2, 'classes': 2, 'samples': 300, 'is_2d': True,  'desc': 'Two concentric circles'},
    'spiral':  {'name': 'Spiral',      'features': 2, 'classes': 3, 'samples': 450, 'is_2d': True,  'desc': 'Three interleaving spirals'},
    'moons':   {'name': 'Two Moons',   'features': 2, 'classes': 2, 'samples': 300, 'is_2d': True,  'desc': 'Two crescent moons'},
    'gauss':   {'name': 'Gaussian 4D', 'features': 4, 'classes': 3, 'samples': 150, 'is_2d': False, 'desc': '3-class Gaussian clusters in 4D'},
}

DATASET_MAKERS = {
    'xor': make_xor, 'circles': make_circles, 'spiral': make_spiral,
    'moons': make_moons, 'gauss': make_gaussian,
}


# ==================== Model Builder ====================

def build_model(layers_config):
    modules = []
    for layer in layers_config:
        t = layer['type']
        if t == 'linear':
            modules.append(nn.Linear(int(layer['in_features']), int(layer['out_features'])))
        elif t == 'relu':
            modules.append(nn.ReLU())
        elif t == 'sigmoid':
            modules.append(nn.Sigmoid())
        elif t == 'tanh':
            modules.append(nn.Tanh())
        elif t == 'leaky_relu':
            modules.append(nn.LeakyReLU(0.1))
        elif t == 'dropout':
            modules.append(nn.Dropout(float(layer.get('rate', 0.2))))
        elif t == 'batchnorm':
            modules.append(nn.BatchNorm1d(int(layer['num_features'])))
    return nn.Sequential(*modules)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


_boundary_cache = {}


def compute_boundary(model, X, grid_size=30):
    x_min, x_max = X[:, 0].min().item() - 0.5, X[:, 0].max().item() + 0.5
    y_min, y_max = X[:, 1].min().item() - 0.5, X[:, 1].max().item() + 0.5

    cache_key = (grid_size, round(x_min, 2), round(x_max, 2), round(y_min, 2), round(y_max, 2))
    if cache_key in _boundary_cache:
        grid = _boundary_cache[cache_key]
    else:
        xx = torch.linspace(x_min, x_max, grid_size)
        yy = torch.linspace(y_min, y_max, grid_size)
        gx, gy = torch.meshgrid(xx, yy, indexing='ij')
        grid = torch.stack([gx.flatten(), gy.flatten()], dim=1)
        if len(_boundary_cache) > 10:
            _boundary_cache.clear()
        _boundary_cache[cache_key] = grid

    model.eval()
    with torch.no_grad():
        out = model(grid)
        probs = torch.softmax(out, dim=1)
        classes = probs.argmax(dim=1).reshape(grid_size, grid_size)
        confidence = probs.max(dim=1).values.reshape(grid_size, grid_size)

    return {
        'x_range': [round(x_min, 3), round(x_max, 3)],
        'y_range': [round(y_min, 3), round(y_max, 3)],
        'grid_size': grid_size,
        'predictions': classes.tolist(),
        'confidence': [[round(v, 2) for v in row] for row in confidence.tolist()],
    }


def _active_training_count():
    with training_lock:
        dead = [tid for tid, s in training_sessions.items()
                if 'thread' in s and not s['thread'].is_alive()]
        for tid in dead:
            training_sessions.pop(tid, None)
        return sum(1 for s in training_sessions.values()
                   if not s.get('finished', False))


# ==================== Flask App ====================

def create_app():
    app = Flask(__name__)
    app.secret_key = 'smartnet-playground'

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/api/datasets')
    def datasets():
        return jsonify(DATASET_INFO)

    @app.route('/api/train', methods=['POST'])
    def train():
        if _active_training_count() >= MAX_CONCURRENT_TRAINING:
            return jsonify({
                'success': False,
                'error': f'Server busy — max {MAX_CONCURRENT_TRAINING} concurrent trainings. Try again shortly.'
            })

        config = request.json or {}
        layers = config.get('layers', [])
        dataset_name = config.get('dataset', 'xor')
        epochs = min(int(config.get('epochs', 50)), 500)
        lr = float(config.get('learning_rate', 0.01))

        if not layers:
            return jsonify({'success': False, 'error': 'No layers defined'})
        if len(layers) > 8:
            return jsonify({'success': False, 'error': 'Max 8 layers allowed'})
        if dataset_name not in DATASET_MAKERS:
            return jsonify({'success': False, 'error': f'Unknown dataset: {dataset_name}'})

        for l in layers:
            if l.get('type') == 'linear':
                if int(l.get('out_features', 0)) > 128:
                    return jsonify({'success': False, 'error': 'Max 128 units per layer'})

        try:
            model = build_model(layers)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Build error: {e}'})

        params = count_parameters(model)
        if params > 100_000:
            return jsonify({'success': False, 'error': f'Too many parameters ({params:,}). Max 100K for this demo.'})

        try:
            X, y = DATASET_MAKERS[dataset_name]()
        except Exception as e:
            return jsonify({'success': False, 'error': f'Dataset error: {e}'})

        info = DATASET_INFO[dataset_name]
        n_classes = info['classes']
        is_2d = info['is_2d']

        try:
            test_out = model(X[:1])
            out_dim = test_out.shape[-1]
            if out_dim != n_classes:
                return jsonify({
                    'success': False,
                    'error': f'Output dim {out_dim} ≠ {n_classes} classes'
                })
        except Exception as e:
            return jsonify({'success': False, 'error': f'Forward pass error: {e}'})

        tid = uuid.uuid4().hex[:8]
        q = queue.Queue(maxsize=600)
        session = {'queue': q, 'stop': False}

        with training_lock:
            training_sessions[tid] = session

        def train_loop(_model, _X, _y):
            optimizer = None
            try:
                optimizer = torch.optim.Adam(_model.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss()

                if is_2d:
                    compact_X = [[round(v, 4) for v in row] for row in _X.tolist()]
                    q.put(json.dumps({
                        'type': 'data',
                        'data_points': {'X': compact_X, 'y': _y.tolist()}
                    }))

                boundary_interval = max(1, epochs // 40)

                for epoch in range(1, epochs + 1):
                    if session['stop']:
                        q.put(json.dumps({'type': 'stopped', 'epoch': epoch}))
                        return

                    _model.train()
                    optimizer.zero_grad()
                    output = _model(_X)
                    loss = criterion(output, _y)
                    loss.backward()
                    optimizer.step()

                    _model.eval()
                    with torch.no_grad():
                        pred = _model(_X).argmax(dim=1)
                        acc = (pred == _y).float().mean().item()

                    event = {
                        'type': 'epoch',
                        'epoch': epoch,
                        'total_epochs': epochs,
                        'loss': round(loss.item(), 5),
                        'accuracy': round(acc, 4),
                    }

                    if is_2d and (epoch == 1 or epoch % boundary_interval == 0 or epoch == epochs):
                        event['boundary'] = compute_boundary(_model, _X, grid_size=30)

                    q.put(json.dumps(event))
                    time.sleep(0.03)

                q.put(json.dumps({'type': 'done'}))

            except Exception as e:
                logger.error(f'Training error [{tid}]: {e}')
                q.put(json.dumps({'type': 'error', 'message': str(e)}))
            finally:
                session['finished'] = True
                del _model, _X, _y, optimizer
                gc.collect()
                time.sleep(15)
                with training_lock:
                    training_sessions.pop(tid, None)

        t = threading.Thread(target=train_loop, args=(model, X, y), daemon=True)
        session['thread'] = t
        t.start()

        return jsonify({'success': True, 'training_id': tid, 'params': params})

    @app.route('/api/train/stream/<tid>')
    def train_stream(tid):
        def generate():
            sess = training_sessions.get(tid)
            if not sess:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Session expired'})}\n\n"
                return
            q = sess['queue']
            while True:
                try:
                    data = q.get(timeout=30)
                    yield f"data: {data}\n\n"
                    parsed = json.loads(data)
                    if parsed.get('type') in ('done', 'stopped', 'error'):
                        break
                except queue.Empty:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
        )

    @app.route('/api/train/stop/<tid>', methods=['POST'])
    def train_stop(tid):
        sess = training_sessions.get(tid)
        if sess:
            sess['stop'] = True
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'Not found'})

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=False, host='0.0.0.0', port=5000)
