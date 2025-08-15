"""
SmartNet: 智能神经网络积木构建框架

一个独立的、模块化的神经网络构建框架，专门用于基于可解释性信息
(SHAP、Fisher等)智能构建小型高效的transformer网络。

主要特性:
- 🧱 模块化组件设计 (Transformer、CNN、RNN、MLP等)
- 🤖 基于可解释性信息的智能架构搜索
- ⚡ 轻量级、高效推理
- 🔍 内置可解释性支持
- 🎯 专为推荐系统优化

版本: 1.0.0
作者: DriftRec Team
日期: 2025-08-15
"""

__version__ = "1.0.0"
__author__ = "DriftRec Team"

# 导入核心组件
try:
    from .core import (
        BaseBlock, NetworkBuilder, SmartOptimizer, SmartNetwork,
        BlockConfig, NetworkConfig
    )
    from .blocks import (
        TransformerBlock, CNNBlock, RNNBlock, MLPBlock,
        AttentionBlock, FeedForwardBlock, EmbeddingBlock,
        TransformerConfig, CNNConfig, RNNConfig, MLPConfig
    )
    from .architectures import (
        SmallTransformer, HybridCNN, AdaptiveRNN, 
        ExplainableNet, RecommenderNet,
        SmallTransformerConfig, HybridCNNConfig, RecommenderNetConfig
    )
    from .builders import (
        ShapBasedBuilder, FisherBasedBuilder, AutoArchBuilder,
        PerformanceOptimizer, ExplainabilityOptimizer, BuilderConfig
    )
    from .explainable import (
        ExplainableLayer, AttentionVisualizer, FeatureImportanceTracker,
        NetworkExplainer, ExplainabilityConfig
    )
    
    _import_success = True
except ImportError as e:
    print(f"SmartNet导入警告: {e}")
    _import_success = False

__all__ = [
    # 核心组件
    'BaseBlock', 'NetworkBuilder', 'SmartOptimizer', 'SmartNetwork',
    'BlockConfig', 'NetworkConfig',
    
    # 基础积木块
    'TransformerBlock', 'CNNBlock', 'RNNBlock', 'MLPBlock',
    'AttentionBlock', 'FeedForwardBlock', 'EmbeddingBlock',
    'TransformerConfig', 'CNNConfig', 'RNNConfig', 'MLPConfig',
    
    # 预定义架构
    'SmallTransformer', 'HybridCNN', 'AdaptiveRNN', 
    'ExplainableNet', 'RecommenderNet',
    'SmallTransformerConfig', 'HybridCNNConfig', 'RecommenderNetConfig',
    
    # 智能构建器
    'ShapBasedBuilder', 'FisherBasedBuilder', 'AutoArchBuilder',
    'PerformanceOptimizer', 'ExplainabilityOptimizer', 'BuilderConfig',
    
    # 可解释性工具
    'ExplainableLayer', 'AttentionVisualizer', 'FeatureImportanceTracker',
    'NetworkExplainer', 'ExplainabilityConfig'
]
