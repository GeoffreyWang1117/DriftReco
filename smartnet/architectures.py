"""
SmartNet预定义架构

提供常用的网络架构模板，用户可以直接使用或作为基础进行修改。
专门针对推荐系统和可解释性需求进行优化。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass

from .core import BaseBlock, NetworkBuilder, NetworkConfig, BlockConfig
from .blocks import (
    TransformerBlock, CNNBlock, RNNBlock, MLPBlock,
    EmbeddingBlock, AttentionBlock, FeedForwardBlock,
    TransformerConfig, CNNConfig, RNNConfig, MLPConfig
)


# ============================== 架构配置类 ==============================

@dataclass
class SmallTransformerConfig(NetworkConfig):
    """小型Transformer配置"""
    num_layers: int = 6
    num_heads: int = 8
    hidden_dim: int = 512
    ff_dim: int = 2048
    vocab_size: int = 30000
    use_embedding: bool = True
    use_positional: bool = True

@dataclass
class HybridCNNConfig(NetworkConfig):
    """混合CNN配置"""
    cnn_layers: int = 3
    cnn_filters: int = 64
    mlp_layers: int = 2
    mlp_hidden_dim: int = 256
    kernel_sizes: List[int] = None
    
    def __post_init__(self):
        if self.kernel_sizes is None:
            self.kernel_sizes = [3, 5, 7]  # 多尺度卷积核

@dataclass
class RecommenderNetConfig(NetworkConfig):
    """推荐系统网络配置"""
    user_embedding_dim: int = 64
    item_embedding_dim: int = 64
    feature_dim: int = 256
    attention_dim: int = 128
    num_experts: int = 4  # MOE专家数量
    use_dcn: bool = True  # 是否使用Deep&Cross
    use_attention: bool = True


# ============================== 预定义架构 ==============================

class SmallTransformer(nn.Module):
    """
    小型高效Transformer
    
    专门为推荐系统设计的轻量级Transformer，
    在保持性能的同时大幅减少参数量和计算量。
    """
    
    def __init__(self, config: SmallTransformerConfig):
        super().__init__()
        self.config = config
        
        # 构建网络
        builder = NetworkBuilder(config)
        
        # 嵌入层（如果需要）
        if config.use_embedding:
            embedding_config = BlockConfig(
                input_dim=config.vocab_size,
                output_dim=config.hidden_dim,
                name="embedding"
            )
            embedding = EmbeddingBlock(embedding_config, config.vocab_size)
            builder.add_block(embedding, "embedding")
        
        # Transformer层
        for i in range(config.num_layers):
            transformer_config = TransformerConfig(
                input_dim=config.hidden_dim,
                output_dim=config.hidden_dim,
                num_heads=config.num_heads,
                hidden_dim=config.hidden_dim,
                ff_dim=config.ff_dim,
                max_seq_length=config.max_seq_length,
                name=f"transformer_{i}"
            )
            
            transformer_layer = TransformerBlock(transformer_config)
            builder.add_block(transformer_layer, f"transformer_{i}")
        
        # 输出头
        output_config = MLPConfig(
            input_dim=config.hidden_dim,
            output_dim=config.output_features,
            hidden_dims=[config.hidden_dim // 2],
            name="output_head"
        )
        output_head = MLPBlock(output_config)
        builder.add_block(output_head, "output_head")
        
        # 自动连接
        builder.auto_connect()
        
        # 构建模型
        self.model = builder.build()
        
        # 特殊处理：输出层需要全局池化
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播"""
        result = self.model(x, **kwargs)
        output = result['output']
        
        # 如果输出是3D，进行全局平均池化
        if len(output.shape) == 3:
            output = output.transpose(1, 2)  # (batch, dim, seq)
            output = self.global_pooling(output).squeeze(-1)  # (batch, dim)
        
        return output
    
    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """获取所有层的注意力权重"""
        return self.model.get_explainability_info()['attention_weights']


class HybridCNN(nn.Module):
    """
    混合CNN网络
    
    结合多尺度卷积和MLP，适用于序列特征提取和分类任务。
    """
    
    def __init__(self, config: HybridCNNConfig):
        super().__init__()
        self.config = config
        
        builder = NetworkBuilder(config)
        
        # 多尺度CNN分支
        cnn_outputs = []
        for i, kernel_size in enumerate(config.kernel_sizes):
            cnn_config = CNNConfig(
                input_dim=config.input_features,
                output_dim=config.cnn_filters,
                num_filters=config.cnn_filters,
                kernel_size=kernel_size,
                num_layers=config.cnn_layers,
                name=f"cnn_k{kernel_size}"
            )
            
            cnn_block = CNNBlock(cnn_config)
            builder.add_block(cnn_block, f"cnn_{i}")
            cnn_outputs.append(config.cnn_filters)
        
        # 特征融合层
        fusion_input_dim = sum(cnn_outputs)
        fusion_config = MLPConfig(
            input_dim=fusion_input_dim,
            output_dim=config.mlp_hidden_dim,
            hidden_dims=[fusion_input_dim // 2],
            name="fusion"
        )
        fusion_layer = MLPBlock(fusion_config)
        builder.add_block(fusion_layer, "fusion")
        
        # 分类头
        classifier_config = MLPConfig(
            input_dim=config.mlp_hidden_dim,
            output_dim=config.output_features,
            hidden_dims=[config.mlp_hidden_dim // 2],
            num_layers=config.mlp_layers,
            name="classifier"
        )
        classifier = MLPBlock(classifier_config)
        builder.add_block(classifier, "classifier")
        
        # 手动连接（因为是并行结构）
        for i in range(len(config.kernel_sizes)):
            builder.connect(f"cnn_{i}", "fusion", "parallel")
        builder.connect("fusion", "classifier", "sequential")
        
        self.model = builder.build()
        
        # 自定义前向传播来处理并行结构
        self.cnn_blocks = [self.model.blocks[f"cnn_{i}"] for i in range(len(config.kernel_sizes))]
        self.fusion_block = self.model.blocks["fusion"]
        self.classifier_block = self.model.blocks["classifier"]
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播"""
        # 并行CNN特征提取
        cnn_outputs = []
        for cnn_block in self.cnn_blocks:
            cnn_result = cnn_block(x, **kwargs)
            cnn_outputs.append(cnn_result['output'])
        
        # 特征拼接
        fused_features = torch.cat(cnn_outputs, dim=-1)
        
        # 融合和分类
        fusion_result = self.fusion_block(fused_features, **kwargs)
        classifier_result = self.classifier_block(fusion_result['output'], **kwargs)
        
        return classifier_result['output']


class AdaptiveRNN(nn.Module):
    """
    自适应RNN网络
    
    根据序列长度和复杂度自动调整RNN结构。
    """
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        
        builder = NetworkBuilder(config)
        
        # 双向LSTM层
        lstm_config = RNNConfig(
            input_dim=config.input_features,
            output_dim=config.input_features,
            hidden_size=256,
            num_layers=2,
            rnn_type="LSTM",
            bidirectional=True,
            return_sequences=True,
            name="lstm"
        )
        lstm_block = RNNBlock(lstm_config)
        builder.add_block(lstm_block, "lstm")
        
        # 注意力层
        attention_config = BlockConfig(
            input_dim=config.input_features,
            output_dim=config.input_features,
            name="attention"
        )
        attention_block = AttentionBlock(attention_config, num_heads=8)
        builder.add_block(attention_block, "attention")
        
        # 输出层
        output_config = MLPConfig(
            input_dim=config.input_features,
            output_dim=config.output_features,
            hidden_dims=[128, 64],
            name="output"
        )
        output_block = MLPBlock(output_config)
        builder.add_block(output_block, "output")
        
        builder.auto_connect()
        self.model = builder.build()
        
        # 自适应池化
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播"""
        result = self.model(x, **kwargs)
        output = result['output']
        
        # 自适应池化到固定维度
        if len(output.shape) == 3:
            output = output.transpose(1, 2)
            output = self.adaptive_pooling(output).squeeze(-1)
        
        return output


class ExplainableNet(nn.Module):
    """
    可解释网络
    
    专门设计用于提供丰富解释信息的网络架构。
    """
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        
        builder = NetworkBuilder(config)
        
        # 输入处理层
        input_config = MLPConfig(
            input_dim=config.input_features,
            output_dim=256,
            hidden_dims=[512, 384],
            name="input_processor",
            explainable=True
        )
        input_processor = MLPBlock(input_config)
        builder.add_block(input_processor, "input_processor")
        
        # 可解释注意力层
        attention_config = BlockConfig(
            input_dim=256,
            output_dim=256,
            name="explainable_attention",
            explainable=True,
            attention_weights=True
        )
        attention_layer = AttentionBlock(attention_config, num_heads=4)
        builder.add_block(attention_layer, "explainable_attention")
        
        # 特征重要性层
        importance_config = MLPConfig(
            input_dim=256,
            output_dim=128,
            hidden_dims=[192],
            name="importance_layer",
            explainable=True
        )
        importance_layer = MLPBlock(importance_config)
        builder.add_block(importance_layer, "importance_layer")
        
        # 预测头
        prediction_config = MLPConfig(
            input_dim=128,
            output_dim=config.output_features,
            hidden_dims=[64],
            name="prediction_head"
        )
        prediction_head = MLPBlock(prediction_config)
        builder.add_block(prediction_head, "prediction_head")
        
        builder.auto_connect()
        self.model = builder.build()
        
        # 特征重要性计算
        self.importance_weights = nn.Parameter(torch.randn(128))
    
    def forward(self, x: torch.Tensor, return_explanations: bool = False, **kwargs):
        """
        前向传播
        
        Args:
            x: 输入
            return_explanations: 是否返回解释信息
        """
        result = self.model(x, **kwargs)
        output = result['output']
        
        if return_explanations:
            # 计算特征重要性
            intermediate = result['intermediate_outputs']
            importance_features = intermediate.get('importance_layer', None)
            
            if importance_features is not None:
                # 计算每个特征的重要性分数
                importance_scores = torch.abs(importance_features * self.importance_weights)
                importance_scores = F.softmax(importance_scores, dim=-1)
            else:
                importance_scores = None
            
            return {
                'prediction': output,
                'attention_weights': result['attention_weights'],
                'feature_importance': importance_scores,
                'intermediate_features': intermediate
            }
        else:
            return output


class RecommenderNet(nn.Module):
    """
    推荐系统专用网络
    
    集成了推荐系统常用的组件：嵌入、交叉特征、多任务等。
    """
    
    def __init__(self, config: RecommenderNetConfig):
        super().__init__()
        self.config = config
        
        # 用户和物品嵌入
        self.user_embedding = nn.Embedding(
            config.user_vocab_size if hasattr(config, 'user_vocab_size') else 100000,
            config.user_embedding_dim
        )
        self.item_embedding = nn.Embedding(
            config.item_vocab_size if hasattr(config, 'item_vocab_size') else 100000,
            config.item_embedding_dim
        )
        
        # 特征交叉层 (Deep & Cross Network风格)
        if config.use_dcn:
            self.cross_layers = nn.ModuleList([
                nn.Linear(config.feature_dim, config.feature_dim, bias=False)
                for _ in range(3)
            ])
            self.cross_bias = nn.ParameterList([
                nn.Parameter(torch.zeros(config.feature_dim))
                for _ in range(3)
            ])
        
        # 注意力机制
        if config.use_attention:
            self.user_item_attention = MultiHeadSelfAttention(
                config.user_embedding_dim + config.item_embedding_dim,
                num_heads=4
            )
        
        # 深度网络
        builder = NetworkBuilder(config)
        
        # 特征融合层
        fusion_input_dim = config.user_embedding_dim + config.item_embedding_dim
        if hasattr(config, 'context_features'):
            fusion_input_dim += config.context_features
        
        fusion_config = MLPConfig(
            input_dim=fusion_input_dim,
            output_dim=config.feature_dim,
            hidden_dims=[fusion_input_dim * 2, config.feature_dim * 2],
            name="feature_fusion"
        )
        fusion_layer = MLPBlock(fusion_config)
        builder.add_block(fusion_layer, "fusion")
        
        # 深度层
        deep_config = MLPConfig(
            input_dim=config.feature_dim,
            output_dim=config.feature_dim // 2,
            hidden_dims=[config.feature_dim, config.feature_dim // 2],
            name="deep_layers"
        )
        deep_layer = MLPBlock(deep_config)
        builder.add_block(deep_layer, "deep")
        
        # 输出层
        final_input_dim = config.feature_dim // 2
        if config.use_dcn:
            final_input_dim += config.feature_dim  # Cross network输出
        
        output_config = MLPConfig(
            input_dim=final_input_dim,
            output_dim=config.output_features,
            hidden_dims=[final_input_dim // 2],
            name="output"
        )
        output_layer = MLPBlock(output_config)
        builder.add_block(output_layer, "output")
        
        builder.auto_connect()
        self.deep_network = builder.build()
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                context_features: Optional[torch.Tensor] = None, **kwargs):
        """
        前向传播
        
        Args:
            user_ids: 用户ID
            item_ids: 物品ID  
            context_features: 上下文特征
        """
        # 嵌入查找
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # 特征拼接
        features = torch.cat([user_emb, item_emb], dim=-1)
        if context_features is not None:
            features = torch.cat([features, context_features], dim=-1)
        
        # 注意力机制
        if self.config.use_attention:
            if len(features.shape) == 2:
                features = features.unsqueeze(1)  # 添加序列维度
            features_att, _ = self.user_item_attention(features, features, features)
            features = features_att.squeeze(1)
        
        # Deep网络
        deep_result = self.deep_network(features, **kwargs)
        deep_output = deep_result['output']
        
        # Cross网络 (如果启用)
        if self.config.use_dcn:
            cross_output = features
            for i, (linear, bias) in enumerate(zip(self.cross_layers, self.cross_bias)):
                cross_temp = linear(cross_output)
                cross_output = features * cross_temp + bias + cross_output
            
            # 拼接Deep和Cross输出
            final_features = torch.cat([deep_output, cross_output], dim=-1)
        else:
            final_features = deep_output
        
        # 最终输出
        output_result = self.deep_network.blocks['output'](final_features, **kwargs)
        return output_result['output']


class MultiHeadSelfAttention(nn.Module):
    """自注意力机制辅助类"""
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.w_o(context)
        
        return output, attention_weights


# 导出所有架构
__all__ = [
    'SmallTransformerConfig', 'HybridCNNConfig', 'RecommenderNetConfig',
    'SmallTransformer', 'HybridCNN', 'AdaptiveRNN', 
    'ExplainableNet', 'RecommenderNet'
]
