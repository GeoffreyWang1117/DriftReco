"""
SmartNet核心组件

提供基础的抽象类和核心功能，为整个框架提供统一的接口和基础设施。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import numpy as np
from collections import OrderedDict
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BlockConfig:
    """
    网络块配置基类
    
    定义每个网络块的通用配置参数
    """
    input_dim: int
    output_dim: int
    name: str = "base_block"
    dropout: float = 0.1
    activation: str = "relu"  # relu, gelu, swish, tanh等
    batch_norm: bool = True
    layer_norm: bool = False
    bias: bool = True
    
    # 可解释性相关配置
    explainable: bool = True
    attention_weights: bool = False
    
    # 性能相关配置
    use_checkpoint: bool = False  # 梯度检查点
    memory_efficient: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'name': self.name,
            'dropout': self.dropout,
            'activation': self.activation,
            'batch_norm': self.batch_norm,
            'layer_norm': self.layer_norm,
            'bias': self.bias,
            'explainable': self.explainable,
            'attention_weights': self.attention_weights,
            'use_checkpoint': self.use_checkpoint,
            'memory_efficient': self.memory_efficient
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BlockConfig':
        """从字典创建配置"""
        return cls(**config_dict)


@dataclass  
class NetworkConfig:
    """
    网络整体配置类
    
    定义整个网络的架构和训练参数
    """
    name: str = "smartnet_model"
    input_features: int = 768
    output_features: int = 1
    max_seq_length: int = 512
    
    # 优化器相关
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"  # adam, adamw, sgd
    scheduler: str = "cosine"  # cosine, linear, constant
    
    # 训练相关
    batch_size: int = 32
    epochs: int = 100
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    
    # 设备和性能
    device: str = "auto"  # auto, cpu, cuda
    mixed_precision: bool = True
    compile_model: bool = False  # PyTorch 2.0编译
    
    # 可解释性设置
    enable_explainability: bool = True
    track_attention: bool = True
    save_intermediate: bool = False
    
    def validate(self) -> bool:
        """验证配置的有效性"""
        if self.input_features <= 0:
            raise ValueError("input_features必须为正数")
        if self.output_features <= 0:
            raise ValueError("output_features必须为正数")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate必须为正数")
        return True


class BaseBlock(nn.Module, ABC):
    """
    所有网络块的基类
    
    提供统一的接口和基础功能：
    - 标准化的forward接口
    - 可解释性支持
    - 配置管理
    - 性能监控
    """
    
    def __init__(self, config: BlockConfig):
        super().__init__()
        self.config = config
        self.name = config.name
        
        # 可解释性相关
        self.explainable = config.explainable
        self.attention_weights = []
        self.feature_importances = []
        
        # 性能统计
        self.forward_count = 0
        self.total_time = 0.0
        
        # 构建激活函数
        self.activation = self._build_activation(config.activation)
        
        # 构建归一化层
        if config.batch_norm:
            self.batch_norm = nn.BatchNorm1d(config.output_dim)
        else:
            self.batch_norm = None
            
        if config.layer_norm:
            self.layer_norm = nn.LayerNorm(config.output_dim)
        else:
            self.layer_norm = None
    
    def _build_activation(self, activation: str) -> nn.Module:
        """构建激活函数"""
        activation_map = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
        }
        
        if activation.lower() in activation_map:
            return activation_map[activation.lower()]
        else:
            logger.warning(f"未知激活函数 {activation}，使用ReLU")
            return nn.ReLU()
    
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量
            **kwargs: 额外参数
        
        Returns:
            Dict包含:
            - 'output': 主要输出张量
            - 'attention': 注意力权重(如果有)  
            - 'features': 中间特征(如果需要可解释性)
        """
        pass
    
    def apply_normalization(self, x: torch.Tensor) -> torch.Tensor:
        """应用归一化"""
        if self.batch_norm is not None and len(x.shape) == 2:
            x = self.batch_norm(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x
    
    def get_config(self) -> BlockConfig:
        """获取配置"""
        return self.config
    
    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return {
            'forward_count': self.forward_count,
            'total_time': self.total_time,
            'avg_time': self.total_time / max(1, self.forward_count),
            'attention_shapes': [att.shape for att in self.attention_weights[-5:]],
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.forward_count = 0
        self.total_time = 0.0
        self.attention_weights.clear()
        self.feature_importances.clear()


class NetworkBuilder:
    """
    网络构建器主类
    
    负责组装各种网络块，构建完整的神经网络架构
    """
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.blocks = OrderedDict()
        self.connections = []  # [(from_block, to_block, connection_type)]
        self.built_model = None
        
        logger.info(f"初始化NetworkBuilder: {config.name}")
    
    def add_block(self, block: BaseBlock, name: str = None) -> 'NetworkBuilder':
        """
        添加网络块
        
        Args:
            block: 网络块实例
            name: 块的名称，如果不指定则使用块的默认名称
        
        Returns:
            self (支持链式调用)
        """
        block_name = name or block.name or f"block_{len(self.blocks)}"
        
        if block_name in self.blocks:
            logger.warning(f"块 {block_name} 已存在，将被覆盖")
        
        self.blocks[block_name] = block
        logger.info(f"添加网络块: {block_name} ({block.__class__.__name__})")
        
        return self
    
    def connect(self, from_block: str, to_block: str, 
                connection_type: str = "sequential") -> 'NetworkBuilder':
        """
        连接两个网络块
        
        Args:
            from_block: 源块名称
            to_block: 目标块名称  
            connection_type: 连接类型 ("sequential", "residual", "attention")
        
        Returns:
            self (支持链式调用)
        """
        if from_block not in self.blocks:
            raise ValueError(f"源块 {from_block} 不存在")
        if to_block not in self.blocks:
            raise ValueError(f"目标块 {to_block} 不存在")
        
        self.connections.append((from_block, to_block, connection_type))
        logger.info(f"连接块: {from_block} -> {to_block} ({connection_type})")
        
        return self
    
    def build(self) -> nn.Module:
        """
        构建完整的神经网络
        
        Returns:
            完整的PyTorch模型
        """
        if not self.blocks:
            raise ValueError("没有添加任何网络块")
        
        # 创建智能网络包装器
        self.built_model = SmartNetwork(
            blocks=self.blocks,
            connections=self.connections,
            config=self.config
        )
        
        logger.info(f"成功构建网络: {self.config.name}")
        logger.info(f"包含 {len(self.blocks)} 个块, {len(self.connections)} 个连接")
        
        return self.built_model
    
    def auto_connect(self) -> 'NetworkBuilder':
        """
        自动连接所有块（按添加顺序顺序连接）
        
        Returns:
            self (支持链式调用)
        """
        block_names = list(self.blocks.keys())
        
        for i in range(len(block_names) - 1):
            self.connect(block_names[i], block_names[i + 1], "sequential")
        
        logger.info("自动连接完成")
        return self
    
    def get_summary(self) -> str:
        """获取网络架构摘要"""
        summary = [f"网络架构摘要: {self.config.name}"]
        summary.append("=" * 50)
        
        summary.append("网络块:")
        for name, block in self.blocks.items():
            config = block.get_config()
            summary.append(f"  {name}: {block.__class__.__name__} "
                         f"({config.input_dim} -> {config.output_dim})")
        
        summary.append("\n连接关系:")
        for from_b, to_b, conn_type in self.connections:
            summary.append(f"  {from_b} --{conn_type}--> {to_b}")
        
        return "\n".join(summary)


class SmartNetwork(nn.Module):
    """
    智能神经网络包装器
    
    将多个网络块组装成完整的网络，支持：
    - 灵活的块间连接
    - 自动梯度管理
    - 可解释性信息收集
    - 性能监控
    """
    
    def __init__(self, blocks: OrderedDict, connections: List[Tuple], 
                 config: NetworkConfig):
        super().__init__()
        self.config = config
        self.connections = connections
        
        # 注册所有块作为子模块
        self.blocks = nn.ModuleDict(blocks)
        
        # 构建执行图
        self.execution_graph = self._build_execution_graph()
        
        # 可解释性追踪
        self.track_explainability = config.enable_explainability
        self.intermediate_outputs = {}
        
    def _build_execution_graph(self) -> List[str]:
        """构建执行顺序图"""
        # 简化版本：按添加顺序执行
        # 未来可以实现更复杂的拓扑排序
        return list(self.blocks.keys())
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量
            **kwargs: 额外参数
        
        Returns:
            包含输出和可解释性信息的字典
        """
        current_output = x
        all_outputs = {'input': x}
        all_attention = {}
        
        # 按执行图顺序执行每个块
        for block_name in self.execution_graph:
            block = self.blocks[block_name]
            
            # 执行块的前向传播
            block_result = block(current_output, **kwargs)
            
            # 处理结果
            if isinstance(block_result, dict):
                current_output = block_result['output']
                if 'attention' in block_result and block_result['attention'] is not None:
                    all_attention[block_name] = block_result['attention']
            else:
                current_output = block_result
            
            # 存储中间输出（如果需要）
            if self.track_explainability:
                all_outputs[block_name] = current_output
        
        # 返回完整结果
        result = {
            'output': current_output,
            'intermediate_outputs': all_outputs,
            'attention_weights': all_attention
        }
        
        return result
    
    def get_explainability_info(self) -> Dict[str, Any]:
        """获取可解释性信息"""
        info = {
            'block_stats': {},
            'attention_summary': {},
            'parameter_count': sum(p.numel() for p in self.parameters()),
            'memory_usage': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        }
        
        # 收集每个块的统计信息
        for name, block in self.blocks.items():
            if hasattr(block, 'get_stats'):
                info['block_stats'][name] = block.get_stats()
        
        return info
    
    def save_config(self, path: str):
        """保存网络配置"""
        config_dict = {
            'network_config': self.config.__dict__,
            'blocks_config': {
                name: block.get_config().to_dict() 
                for name, block in self.blocks.items()
            },
            'connections': self.connections
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        
        logger.info(f"网络配置已保存到: {path}")


class SmartOptimizer:
    """
    智能优化器
    
    根据网络结构和训练数据自动调整优化策略
    """
    
    def __init__(self, model: nn.Module, config: NetworkConfig):
        self.model = model
        self.config = config
        self.optimizer = None
        self.scheduler = None
        self.scaler = None  # 用于混合精度训练
        
        self._build_optimizer()
        self._build_scheduler()
        
        if config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def _build_optimizer(self):
        """构建优化器"""
        params = self.model.parameters()
        
        if self.config.optimizer.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                params, 
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {self.config.optimizer}")
        
        logger.info(f"构建优化器: {self.config.optimizer}")
    
    def _build_scheduler(self):
        """构建学习率调度器"""
        if self.config.scheduler.lower() == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.epochs
            )
        elif self.config.scheduler.lower() == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.1,
                total_iters=self.config.epochs
            )
        elif self.config.scheduler.lower() == "constant":
            self.scheduler = None  # 不使用调度器
        else:
            logger.warning(f"未知调度器 {self.config.scheduler}，不使用调度器")
            self.scheduler = None
    
    def step(self, loss: torch.Tensor):
        """执行一步优化"""
        if self.config.mixed_precision and self.scaler:
            # 混合精度训练
            self.scaler.scale(loss).backward()
            
            if self.config.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # 标准训练
            loss.backward()
            
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )
            
            self.optimizer.step()
        
        # 清零梯度
        self.optimizer.zero_grad()
    
    def schedule_step(self):
        """学习率调度步进"""
        if self.scheduler:
            self.scheduler.step()
    
    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']


# 导出所有组件
__all__ = [
    'BlockConfig', 'NetworkConfig', 'BaseBlock', 'NetworkBuilder',
    'SmartNetwork', 'SmartOptimizer'
]
