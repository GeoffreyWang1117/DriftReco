"""
SmartNet基础积木块

提供各种常用的神经网络组件，包括Transformer、CNN、RNN、MLP等，
每个组件都支持可解释性和性能优化。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import math
import time
from dataclasses import dataclass

from .core import BaseBlock, BlockConfig

# ============================== 专用配置类 ==============================

@dataclass
class TransformerConfig(BlockConfig):
    """Transformer块配置"""
    num_heads: int = 8
    hidden_dim: int = 512
    ff_dim: int = 2048
    num_layers: int = 6
    max_seq_length: int = 512
    use_rotary: bool = False  # 是否使用旋转位置编码
    use_flash_attention: bool = False  # 是否使用Flash Attention

@dataclass
class CNNConfig(BlockConfig):
    """CNN块配置"""
    num_filters: int = 64
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    num_layers: int = 3
    pool_type: str = "max"  # max, avg, adaptive
    residual: bool = True

@dataclass
class RNNConfig(BlockConfig):
    """RNN块配置"""
    hidden_size: int = 256
    num_layers: int = 2
    rnn_type: str = "LSTM"  # LSTM, GRU, RNN
    bidirectional: bool = True
    return_sequences: bool = False

@dataclass
class MLPConfig(BlockConfig):
    """MLP块配置"""
    hidden_dims: List[int] = None
    num_layers: int = 3
    use_residual: bool = True
    
    def __post_init__(self):
        if self.hidden_dims is None:
            # 自动生成递减的隐藏层维度
            step = max(1, (self.input_dim - self.output_dim) // self.num_layers)
            self.hidden_dims = [
                max(self.output_dim, self.input_dim - i * step)
                for i in range(1, self.num_layers)
            ]

# ============================== Transformer 相关组件 ==============================

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    支持可解释性分析的注意力实现
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None  # 保存注意力权重用于解释
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            query: (batch_size, seq_len, d_model)
            key: (batch_size, seq_len, d_model)  
            value: (batch_size, seq_len, d_model)
            mask: 注意力掩码
            
        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = query.size()
        
        # 线性变换
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        self.attention_weights = attention_weights.detach()  # 保存用于解释
        
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        context = torch.matmul(attention_weights, V)
        
        # 重塑和输出投影
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.w_o(context)
        
        return output, self.attention_weights


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class TransformerBlock(BaseBlock):
    """
    Transformer块
    
    包含多头注意力和前馈网络的完整transformer层
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.config = config
        
        # 多头注意力
        self.self_attention = MultiHeadAttention(
            d_model=config.input_dim,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(config.input_dim, config.ff_dim),
            self.activation,
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.output_dim)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(config.input_dim)
        self.norm2 = nn.LayerNorm(config.output_dim)
        
        # 位置编码
        if hasattr(config, 'max_seq_length'):
            self.pos_encoding = PositionalEncoding(
                config.input_dim, config.max_seq_length
            )
        else:
            self.pos_encoding = None
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, d_model)
            
        Returns:
            包含output和attention权重的字典
        """
        start_time = time.time()
        
        # 添加位置编码
        if self.pos_encoding is not None and len(x.shape) == 3:
            x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # 自注意力机制
        residual = x
        x = self.norm1(x)
        attn_output, attention_weights = self.self_attention(x, x, x)
        x = residual + self.dropout(attn_output)
        
        # 前馈网络
        residual = x
        x = self.norm2(x)
        ff_output = self.feed_forward(x)
        x = residual + self.dropout(ff_output)
        
        # 更新统计信息
        self.forward_count += 1
        self.total_time += time.time() - start_time
        
        if self.explainable:
            self.attention_weights.append(attention_weights)
        
        return {
            'output': x,
            'attention': attention_weights if self.config.attention_weights else None,
            'features': ff_output if self.explainable else None
        }


# ============================== CNN 相关组件 ==============================

class CNNBlock(BaseBlock):
    """
    CNN块
    
    支持多层卷积、残差连接和各种池化方式
    """
    
    def __init__(self, config: CNNConfig):
        super().__init__(config)
        self.config = config
        
        layers = []
        in_channels = config.input_dim
        
        for i in range(config.num_layers):
            # 卷积层
            layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=config.num_filters,
                kernel_size=config.kernel_size,
                stride=config.stride,
                padding=config.padding
            ))
            
            # 批归一化
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(config.num_filters))
            
            # 激活函数
            layers.append(self.activation)
            
            # Dropout
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            
            in_channels = config.num_filters
        
        self.conv_layers = nn.Sequential(*layers)
        
        # 池化层
        if config.pool_type == "max":
            self.pooling = nn.AdaptiveMaxPool1d(1)
        elif config.pool_type == "avg":
            self.pooling = nn.AdaptiveAvgPool1d(1)
        else:
            self.pooling = nn.Identity()
        
        # 输出投影
        self.output_projection = nn.Linear(config.num_filters, config.output_dim)
        
        # 残差连接（如果输入输出维度匹配）
        if config.residual and config.input_dim == config.output_dim:
            self.residual_projection = nn.Identity()
        elif config.residual:
            self.residual_projection = nn.Linear(config.input_dim, config.output_dim)
        else:
            self.residual_projection = None
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, input_dim) 或 (batch_size, input_dim)
            
        Returns:
            包含output的字典
        """
        start_time = time.time()
        residual = x
        
        # 确保输入格式正确 (batch_size, channels, seq_len)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # 如果输入是 (batch_size, seq_len, input_dim)，需要转置
        if x.size(1) != self.config.input_dim and x.size(-1) == self.config.input_dim:
            x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        
        # 卷积处理
        conv_output = self.conv_layers(x)
        
        # 池化
        pooled_output = self.pooling(conv_output)
        
        # 展平
        if len(pooled_output.shape) > 2:
            pooled_output = pooled_output.squeeze(-1)  # 移除seq_len维度
        
        # 输出投影
        output = self.output_projection(pooled_output)
        
        # 残差连接
        if self.residual_projection is not None:
            if len(residual.shape) == 3:
                residual = residual.mean(dim=1)  # 平均池化处理序列维度
            residual_out = self.residual_projection(residual)
            output = output + residual_out
        
        # 更新统计信息
        self.forward_count += 1
        self.total_time += time.time() - start_time
        
        return {
            'output': output,
            'features': conv_output if self.explainable else None
        }


# ============================== RNN 相关组件 ==============================

class RNNBlock(BaseBlock):
    """
    RNN块
    
    支持LSTM、GRU和vanilla RNN，以及双向处理
    """
    
    def __init__(self, config: RNNConfig):
        super().__init__(config)
        self.config = config
        
        # 选择RNN类型
        if config.rnn_type.upper() == "LSTM":
            self.rnn = nn.LSTM(
                input_size=config.input_dim,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                bidirectional=config.bidirectional,
                dropout=config.dropout if config.num_layers > 1 else 0,
                batch_first=True
            )
        elif config.rnn_type.upper() == "GRU":
            self.rnn = nn.GRU(
                input_size=config.input_dim,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                bidirectional=config.bidirectional,
                dropout=config.dropout if config.num_layers > 1 else 0,
                batch_first=True
            )
        else:
            self.rnn = nn.RNN(
                input_size=config.input_dim,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                bidirectional=config.bidirectional,
                dropout=config.dropout if config.num_layers > 1 else 0,
                batch_first=True,
                nonlinearity='tanh'
            )
        
        # 输出维度计算
        rnn_output_dim = config.hidden_size * (2 if config.bidirectional else 1)
        
        # 输出投影层
        self.output_projection = nn.Linear(rnn_output_dim, config.output_dim)
        
        # 注意力机制（用于序列池化）
        self.use_attention = not config.return_sequences
        if self.use_attention:
            self.attention = nn.Linear(rnn_output_dim, 1)
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, input_dim)
            
        Returns:
            包含output的字典
        """
        start_time = time.time()
        
        # 确保输入是3D张量
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # 添加seq_len维度
        
        # RNN前向传播
        rnn_output, hidden = self.rnn(x)  # (batch_size, seq_len, hidden_size*directions)
        
        if self.config.return_sequences:
            # 返回所有时间步的输出
            output = self.output_projection(rnn_output)
        else:
            # 使用注意力机制池化序列
            if self.use_attention:
                # 计算注意力权重
                attention_weights = F.softmax(self.attention(rnn_output), dim=1)
                # 加权平均
                pooled_output = torch.sum(attention_weights * rnn_output, dim=1)
                attention_weights = attention_weights.squeeze(-1)
            else:
                # 简单平均池化
                pooled_output = rnn_output.mean(dim=1)
                attention_weights = None
            
            output = self.output_projection(pooled_output)
        
        # 更新统计信息
        self.forward_count += 1
        self.total_time += time.time() - start_time
        
        result = {'output': output}
        
        if self.explainable:
            result['features'] = rnn_output
            if not self.config.return_sequences and self.use_attention:
                result['attention'] = attention_weights
        
        return result


# ============================== MLP 相关组件 ==============================

class MLPBlock(BaseBlock):
    """
    多层感知机块
    
    支持残差连接、多种激活函数和归一化方式
    """
    
    def __init__(self, config: MLPConfig):
        super().__init__(config)
        self.config = config
        
        layers = []
        input_dim = config.input_dim
        
        # 构建隐藏层
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim, bias=config.bias))
            
            if config.layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            elif config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self.activation)
            
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(input_dim, config.output_dim, bias=config.bias))
        
        self.mlp = nn.Sequential(*layers)
        
        # 残差连接
        if config.use_residual and config.input_dim == config.output_dim:
            self.residual_projection = nn.Identity()
        elif config.use_residual:
            self.residual_projection = nn.Linear(config.input_dim, config.output_dim)
        else:
            self.residual_projection = None
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, input_dim) 或 (batch_size, seq_len, input_dim)
            
        Returns:
            包含output的字典
        """
        start_time = time.time()
        
        original_shape = x.shape
        residual = x
        
        # 如果是3D张量，展平最后两个维度
        if len(x.shape) == 3:
            batch_size, seq_len, _ = x.shape
            x = x.view(-1, x.size(-1))
        
        # MLP前向传播
        mlp_output = self.mlp(x)
        
        # 残差连接
        if self.residual_projection is not None:
            if len(original_shape) == 3:
                residual = residual.view(-1, residual.size(-1))
            residual_out = self.residual_projection(residual)
            mlp_output = mlp_output + residual_out
        
        # 恢复原始形状
        if len(original_shape) == 3:
            mlp_output = mlp_output.view(batch_size, seq_len, -1)
        
        # 更新统计信息
        self.forward_count += 1
        self.total_time += time.time() - start_time
        
        return {
            'output': mlp_output,
            'features': mlp_output if self.explainable else None
        }


# ============================== 其他常用组件 ==============================

class EmbeddingBlock(BaseBlock):
    """
    嵌入层块
    
    支持位置嵌入、特征嵌入等多种嵌入方式
    """
    
    def __init__(self, config: BlockConfig, vocab_size: int, 
                 padding_idx: Optional[int] = None):
        super().__init__(config)
        
        self.embedding = nn.Embedding(
            vocab_size, 
            config.output_dim,
            padding_idx=padding_idx
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入ID张量 (batch_size, seq_len)
            
        Returns:
            包含output的字典
        """
        start_time = time.time()
        
        embedded = self.embedding(x)
        output = self.dropout(embedded)
        
        # 更新统计信息
        self.forward_count += 1
        self.total_time += time.time() - start_time
        
        return {
            'output': output,
            'features': embedded if self.explainable else None
        }


class AttentionBlock(BaseBlock):
    """
    独立的注意力块
    
    可以单独使用的注意力机制
    """
    
    def __init__(self, config: BlockConfig, num_heads: int = 8):
        super().__init__(config)
        
        self.attention = MultiHeadAttention(
            d_model=config.input_dim,
            num_heads=num_heads,
            dropout=config.dropout
        )
        
        self.norm = nn.LayerNorm(config.input_dim)
        self.output_projection = nn.Linear(config.input_dim, config.output_dim)
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, input_dim)
            
        Returns:
            包含output和attention权重的字典
        """
        start_time = time.time()
        
        # 自注意力
        residual = x
        x = self.norm(x)
        attn_output, attention_weights = self.attention(x, x, x)
        x = residual + attn_output
        
        # 输出投影
        output = self.output_projection(x)
        
        # 更新统计信息
        self.forward_count += 1
        self.total_time += time.time() - start_time
        
        return {
            'output': output,
            'attention': attention_weights,
            'features': attn_output if self.explainable else None
        }


class FeedForwardBlock(BaseBlock):
    """
    前馈网络块
    
    标准的FFN实现，支持不同的激活函数和归一化
    """
    
    def __init__(self, config: BlockConfig, ff_dim: int = None):
        super().__init__(config)
        
        if ff_dim is None:
            ff_dim = config.input_dim * 4  # 默认为输入维度的4倍
        
        self.feed_forward = nn.Sequential(
            nn.Linear(config.input_dim, ff_dim),
            self.activation,
            nn.Dropout(config.dropout),
            nn.Linear(ff_dim, config.output_dim)
        )
        
        self.norm = nn.LayerNorm(config.input_dim)
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            包含output的字典
        """
        start_time = time.time()
        
        residual = x
        x = self.norm(x)
        ff_output = self.feed_forward(x)
        output = residual + ff_output
        
        # 更新统计信息
        self.forward_count += 1
        self.total_time += time.time() - start_time
        
        return {
            'output': output,
            'features': ff_output if self.explainable else None
        }


# 导出所有块
__all__ = [
    'TransformerConfig', 'CNNConfig', 'RNNConfig', 'MLPConfig',
    'TransformerBlock', 'CNNBlock', 'RNNBlock', 'MLPBlock',
    'EmbeddingBlock', 'AttentionBlock', 'FeedForwardBlock',
    'MultiHeadAttention', 'PositionalEncoding'
]
