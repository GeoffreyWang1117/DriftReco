"""
SmartNet Web应用主模块
提供可视化拖拽式神经网络构建界面
"""

from flask import Flask, render_template, request, jsonify, session
import os
import sys
import json
import traceback
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入SmartNet组件
try:
    from smartnet import (
        NetworkBuilder, NetworkConfig,
        MLPBlock, MLPConfig,
        TransformerBlock, TransformerConfig,
        CNNBlock, CNNConfig,
        RNNBlock, RNNConfig
    )
    SMARTNET_AVAILABLE = True
except ImportError as e:
    logging.warning(f"SmartNet导入失败: {e}")
    SMARTNET_AVAILABLE = False


def create_app():
    """创建Flask应用"""
    app = Flask(__name__)
    
    # 配置密钥
    app.secret_key = os.urandom(24)
    
    # 配置上传文件夹
    app.config['UPLOAD_FOLDER'] = str(Path(__file__).parent / 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
    
    # 确保上传目录存在
    Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
    
    # 配置日志
    if not app.debug:
        logging.basicConfig(level=logging.INFO)
    
    @app.route('/')
    def index():
        """主页 - 可视化拖拽界面"""
        return render_template('index.html')

    @app.route('/code-editor')
    def code_editor():
        """代码编辑器页面"""
        return render_template('code-editor.html')

    @app.route('/dsl-test')
    def dsl_test():
        """DSL解析器测试页面"""
        return render_template('dsl-test.html')    @app.route('/api/components')
    def get_components():
        """获取可用的网络组件列表"""
        components = [
            {
                'id': 'mlp',
                'name': 'MLP层',
                'description': '多层感知机',
                'category': 'basic',
                'color': '#4CAF50',
                'params': {
                    'input_dim': {'type': 'int', 'default': 128, 'min': 1, 'max': 2048},
                    'output_dim': {'type': 'int', 'default': 64, 'min': 1, 'max': 2048},
                    'hidden_dims': {'type': 'list', 'default': [96], 'item_type': 'int'},
                    'dropout': {'type': 'float', 'default': 0.1, 'min': 0.0, 'max': 0.9}
                }
            },
            {
                'id': 'transformer',
                'name': 'Transformer层',
                'description': '多头注意力机制',
                'category': 'advanced',
                'color': '#2196F3',
                'params': {
                    'input_dim': {'type': 'int', 'default': 256, 'min': 1, 'max': 2048},
                    'output_dim': {'type': 'int', 'default': 256, 'min': 1, 'max': 2048},
                    'num_heads': {'type': 'int', 'default': 8, 'min': 1, 'max': 32},
                    'ff_dim': {'type': 'int', 'default': 512, 'min': 1, 'max': 4096},
                    'dropout': {'type': 'float', 'default': 0.1, 'min': 0.0, 'max': 0.9}
                }
            },
            {
                'id': 'cnn',
                'name': 'CNN层',
                'description': '卷积神经网络',
                'category': 'advanced',
                'color': '#FF9800',
                'params': {
                    'input_channels': {'type': 'int', 'default': 3, 'min': 1, 'max': 512},
                    'output_channels': {'type': 'int', 'default': 32, 'min': 1, 'max': 512},
                    'kernel_size': {'type': 'int', 'default': 3, 'min': 1, 'max': 11},
                    'stride': {'type': 'int', 'default': 1, 'min': 1, 'max': 4},
                    'padding': {'type': 'int', 'default': 1, 'min': 0, 'max': 5}
                }
            },
            {
                'id': 'rnn',
                'name': 'RNN层',
                'description': '循环神经网络',
                'category': 'advanced', 
                'color': '#9C27B0',
                'params': {
                    'input_size': {'type': 'int', 'default': 128, 'min': 1, 'max': 2048},
                    'hidden_size': {'type': 'int', 'default': 64, 'min': 1, 'max': 2048},
                    'num_layers': {'type': 'int', 'default': 1, 'min': 1, 'max': 6},
                    'rnn_type': {'type': 'select', 'default': 'LSTM', 'options': ['LSTM', 'GRU']},
                    'dropout': {'type': 'float', 'default': 0.1, 'min': 0.0, 'max': 0.9}
                }
            }
        ]
        
        return jsonify({
            'success': True,
            'components': components,
            'smartnet_available': SMARTNET_AVAILABLE
        })
    
    @app.route('/api/build', methods=['POST'])
    def build_network():
        """根据拖拽设计构建神经网络"""
        try:
            data = request.json
            components = data.get('components', [])
            connections = data.get('connections', [])
            network_config = data.get('config', {})
            
            if not SMARTNET_AVAILABLE:
                return jsonify({
                    'success': False,
                    'error': 'SmartNet未正确加载'
                })
            
            # 验证网络深度限制（双3090训练限制）
            if len(components) > 10:
                return jsonify({
                    'success': False,
                    'error': f'网络深度过大({len(components)}层)，为确保在双3090上正常训练，请限制在10层以内'
                })
            
            # 估算参数量
            total_params = estimate_parameters(components)
            if total_params > 50_000_000:  # 5000万参数限制
                return jsonify({
                    'success': False,
                    'error': f'预估参数量过大({total_params:,})，为确保在双3090上正常训练，请控制在5000万参数以内'
                })
            
            # 创建网络配置
            config = NetworkConfig(
                name=network_config.get('name', 'custom_network'),
                input_features=network_config.get('input_features', 128),
                output_features=network_config.get('output_features', 1),
                learning_rate=network_config.get('learning_rate', 1e-4)
            )
            
            # 构建网络
            builder = NetworkBuilder(config)
            
            # 添加组件
            for i, comp in enumerate(components):
                block = create_block_from_config(comp)
                if block:
                    builder.add_block(block, f"block_{i}")
            
            # 自动连接（简化版，实际应该根据connections参数）
            model = builder.auto_connect().build()
            
            # 计算实际参数量
            actual_params = sum(p.numel() for p in model.parameters())
            
            # 保存模型配置到session
            session['current_model_config'] = {
                'components': components,
                'connections': connections,
                'config': network_config,
                'parameter_count': actual_params
            }
            
            return jsonify({
                'success': True,
                'message': '网络构建成功',
                'parameter_count': actual_params,
                'memory_estimate': estimate_memory(actual_params),
                'training_feasible': actual_params <= 50_000_000
            })
            
        except Exception as e:
            logging.error(f"网络构建失败: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
    
    @app.route('/api/train', methods=['POST'])
    def start_training():
        """开始训练当前构建的网络"""
        try:
            if 'current_model_config' not in session:
                return jsonify({
                    'success': False,
                    'error': '没有可训练的模型，请先构建网络'
                })
            
            training_config = request.json
            
            # 获取训练参数
            epochs = training_config.get('epochs', 10)
            batch_size = training_config.get('batch_size', 32)
            learning_rate = training_config.get('learning_rate', 1e-4)
            
            # 验证训练参数（双3090限制）
            if epochs > 100:
                return jsonify({
                    'success': False,
                    'error': '训练轮数过大，请限制在100轮以内'
                })
            
            if batch_size > 256:
                return jsonify({
                    'success': False,
                    'error': 'batch_size过大，请限制在256以内'
                })
            
            # 这里应该启动实际的训练进程
            # 暂时返回成功信息
            training_id = f"training_{len(session.get('training_jobs', []))}"
            
            if 'training_jobs' not in session:
                session['training_jobs'] = []
            
            training_job = {
                'id': training_id,
                'status': 'running',
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'current_epoch': 0,
                'loss': None,
                'accuracy': None
            }
            
            session['training_jobs'].append(training_job)
            session.modified = True
            
            return jsonify({
                'success': True,
                'training_id': training_id,
                'message': f'训练任务已启动 (ID: {training_id})',
                'estimated_time': f'{epochs * 2} 分钟'  # 粗略估计
            })
            
        except Exception as e:
            logging.error(f"训练启动失败: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            })
    
    @app.route('/api/training/<training_id>')
    def get_training_status(training_id):
        """获取训练状态"""
        training_jobs = session.get('training_jobs', [])
        
        for job in training_jobs:
            if job['id'] == training_id:
                # 模拟训练进度
                if job['status'] == 'running':
                    job['current_epoch'] = min(job['current_epoch'] + 1, job['epochs'])
                    job['loss'] = max(1.0 - job['current_epoch'] * 0.05, 0.1)
                    job['accuracy'] = min(0.5 + job['current_epoch'] * 0.03, 0.95)
                    
                    if job['current_epoch'] >= job['epochs']:
                        job['status'] = 'completed'
                
                return jsonify({
                    'success': True,
                    'job': job
                })
        
        return jsonify({
            'success': False,
            'error': '训练任务不存在'
        })
    
    @app.route('/api/system_info')
    def get_system_info():
        """获取系统信息"""
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count() if gpu_available else 0
            
            gpu_info = []
            if gpu_available:
                for i in range(gpu_count):
                    gpu_info.append({
                        'id': i,
                        'name': torch.cuda.get_device_name(i),
                        'memory_total': torch.cuda.get_device_properties(i).total_memory // 1024**2,  # MB
                    })
            
            return jsonify({
                'success': True,
                'system': {
                    'gpu_available': gpu_available,
                    'gpu_count': gpu_count,
                    'gpu_info': gpu_info,
                    'pytorch_version': torch.__version__,
                    'cuda_version': torch.version.cuda if gpu_available else None
                }
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })
    
    def create_block_from_config(comp_config):
        """根据配置创建网络块"""
        comp_type = comp_config.get('type')
        params = comp_config.get('params', {})
        
        try:
            if comp_type == 'mlp':
                config = MLPConfig(
                    input_dim=params.get('input_dim', 128),
                    output_dim=params.get('output_dim', 64),
                    hidden_dims=params.get('hidden_dims', [96]),
                    dropout=params.get('dropout', 0.1),
                    name=comp_config.get('id', 'mlp')
                )
                return MLPBlock(config)
                
            elif comp_type == 'transformer':
                config = TransformerConfig(
                    input_dim=params.get('input_dim', 256),
                    output_dim=params.get('output_dim', 256),
                    num_heads=params.get('num_heads', 8),
                    ff_dim=params.get('ff_dim', 512),
                    dropout=params.get('dropout', 0.1),
                    name=comp_config.get('id', 'transformer')
                )
                return TransformerBlock(config)
                
            elif comp_type == 'cnn':
                config = CNNConfig(
                    input_channels=params.get('input_channels', 3),
                    output_channels=params.get('output_channels', 32),
                    kernel_size=params.get('kernel_size', 3),
                    stride=params.get('stride', 1),
                    padding=params.get('padding', 1),
                    name=comp_config.get('id', 'cnn')
                )
                return CNNBlock(config)
                
            elif comp_type == 'rnn':
                config = RNNConfig(
                    input_size=params.get('input_size', 128),
                    hidden_size=params.get('hidden_size', 64),
                    num_layers=params.get('num_layers', 1),
                    rnn_type=params.get('rnn_type', 'LSTM'),
                    dropout=params.get('dropout', 0.1),
                    name=comp_config.get('id', 'rnn')
                )
                return RNNBlock(config)
                
        except Exception as e:
            logging.error(f"创建{comp_type}块失败: {e}")
            return None
        
        return None
    
    def estimate_parameters(components):
        """估算模型参数量"""
        total_params = 0
        
        for comp in components:
            comp_type = comp.get('type')
            params = comp.get('params', {})
            
            if comp_type == 'mlp':
                input_dim = params.get('input_dim', 128)
                output_dim = params.get('output_dim', 64) 
                hidden_dims = params.get('hidden_dims', [96])
                
                # 计算MLP参数
                prev_dim = input_dim
                for hidden_dim in hidden_dims:
                    total_params += prev_dim * hidden_dim + hidden_dim  # 权重+偏置
                    prev_dim = hidden_dim
                total_params += prev_dim * output_dim + output_dim
                
            elif comp_type == 'transformer':
                d_model = params.get('input_dim', 256)
                ff_dim = params.get('ff_dim', 512)
                num_heads = params.get('num_heads', 8)
                
                # 粗略估算Transformer参数
                # 注意力机制: 3 * d_model * d_model (Q, K, V)
                # Feed-forward: 2 * d_model * ff_dim
                # Layer norm: 2 * d_model * 2
                total_params += 3 * d_model * d_model + 2 * d_model * ff_dim + 4 * d_model
                
            elif comp_type == 'cnn':
                in_channels = params.get('input_channels', 3)
                out_channels = params.get('output_channels', 32)
                kernel_size = params.get('kernel_size', 3)
                
                # 卷积层参数
                total_params += in_channels * out_channels * kernel_size * kernel_size + out_channels
                
            elif comp_type == 'rnn':
                input_size = params.get('input_size', 128)
                hidden_size = params.get('hidden_size', 64)
                num_layers = params.get('num_layers', 1)
                
                # LSTM参数估算 (4个门)
                total_params += num_layers * 4 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)
        
        return total_params
    
    def estimate_memory(param_count):
        """估算内存使用量(MB)"""
        # 粗略估算: 参数 + 梯度 + 优化器状态 + 激活值
        return param_count * 4 * 3 // 1024 // 1024  # 假设float32, 3倍参数量
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
