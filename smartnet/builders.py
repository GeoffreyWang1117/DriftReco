"""
SmartNet智能构建器

基于可解释性信息(SHAP、Fisher)和性能指标自动构建和优化神经网络架构。
这是框架的核心智能组件，能够根据数据特性自动设计最优网络结构。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
import numpy as np
import json
import logging
from abc import ABC, abstractmethod
from collections import defaultdict

from .core import NetworkBuilder, NetworkConfig, BlockConfig, BaseBlock
from .blocks import (
    TransformerBlock, CNNBlock, RNNBlock, MLPBlock,
    TransformerConfig, CNNConfig, RNNConfig, MLPConfig
)
from .architectures import SmallTransformer, HybridCNN, RecommenderNet

logger = logging.getLogger(__name__)


# ============================== 构建器配置 ==============================

@dataclass
class BuilderConfig:
    """构建器基础配置"""
    max_layers: int = 10
    max_parameters: int = 10_000_000  # 最大参数数量
    target_accuracy: float = 0.85
    max_build_time: int = 3600  # 最大构建时间(秒)
    
    # 搜索策略
    search_strategy: str = "evolutionary"  # evolutionary, grid, random
    population_size: int = 20
    generations: int = 50
    mutation_rate: float = 0.1
    
    # 性能权重
    accuracy_weight: float = 0.4
    speed_weight: float = 0.3
    explainability_weight: float = 0.2
    parameter_weight: float = 0.1

@dataclass
class ArchitectureGene:
    """架构基因 - 用于进化算法"""
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    connections: List[Tuple[int, int, str]] = field(default_factory=list)
    fitness: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def mutate(self, mutation_rate: float = 0.1) -> 'ArchitectureGene':
        """基因变异"""
        new_gene = ArchitectureGene(
            blocks=self.blocks.copy(),
            connections=self.connections.copy()
        )
        
        if np.random.random() < mutation_rate:
            # 变异块配置
            if new_gene.blocks:
                idx = np.random.randint(len(new_gene.blocks))
                block = new_gene.blocks[idx].copy()
                
                # 随机改变某个参数
                if block['type'] == 'transformer':
                    if np.random.random() < 0.5:
                        block['num_heads'] = np.random.choice([4, 8, 12, 16])
                    else:
                        block['hidden_dim'] = np.random.choice([256, 512, 768, 1024])
                elif block['type'] == 'mlp':
                    block['num_layers'] = np.random.randint(2, 6)
                elif block['type'] == 'cnn':
                    block['num_filters'] = np.random.choice([32, 64, 128, 256])
                
                new_gene.blocks[idx] = block
        
        return new_gene
    
    def crossover(self, other: 'ArchitectureGene') -> 'ArchitectureGene':
        """基因交叉"""
        # 简单的单点交叉
        if not self.blocks or not other.blocks:
            return self
        
        crossover_point = np.random.randint(1, min(len(self.blocks), len(other.blocks)))
        
        new_blocks = self.blocks[:crossover_point] + other.blocks[crossover_point:]
        new_connections = self.connections  # 简化处理
        
        return ArchitectureGene(blocks=new_blocks, connections=new_connections)


# ============================== 基础构建器 ==============================

class BaseBuilder(ABC):
    """构建器基类"""
    
    def __init__(self, config: BuilderConfig):
        self.config = config
        self.build_history = []
        self.best_architectures = []
        
    @abstractmethod
    def build(self, **kwargs) -> nn.Module:
        """构建网络"""
        pass
    
    def evaluate_architecture(self, model: nn.Module, 
                            validation_data: Optional[torch.utils.data.DataLoader] = None) -> Dict[str, float]:
        """评估架构性能"""
        metrics = {}
        
        # 参数数量
        total_params = sum(p.numel() for p in model.parameters())
        metrics['parameters'] = total_params
        
        # 模型大小(MB)
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        metrics['model_size_mb'] = (param_size + buffer_size) / 1024 / 1024
        
        # 推理速度测试
        if torch.cuda.is_available():
            model = model.cuda()
            dummy_input = torch.randn(32, 512, 768).cuda()
        else:
            dummy_input = torch.randn(32, 512, 768)
        
        model.eval()
        with torch.no_grad():
            # 预热
            for _ in range(10):
                try:
                    _ = model(dummy_input)
                except:
                    # 处理维度不匹配的情况
                    dummy_input = torch.randn(32, 768)
                    if torch.cuda.is_available():
                        dummy_input = dummy_input.cuda()
                    _ = model(dummy_input)
                    break
            
            # 计时
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if torch.cuda.is_available():
                start_time.record()
                for _ in range(100):
                    _ = model(dummy_input)
                end_time.record()
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time) / 100  # ms per inference
            else:
                import time
                start_time = time.time()
                for _ in range(100):
                    _ = model(dummy_input)
                end_time = time.time()
                elapsed_time = (end_time - start_time) * 10  # ms per inference
            
            metrics['inference_time_ms'] = elapsed_time
        
        # 如果有验证数据，计算准确率
        if validation_data is not None:
            accuracy = self._evaluate_accuracy(model, validation_data)
            metrics['accuracy'] = accuracy
        
        return metrics
    
    def _evaluate_accuracy(self, model: nn.Module, 
                         validation_data: torch.utils.data.DataLoader) -> float:
        """评估模型准确率"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(validation_data):
                if batch_idx >= 10:  # 只评估前10个batch
                    break
                
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                
                try:
                    output = model(data)
                    if len(output.shape) > 1 and output.shape[1] > 1:
                        pred = output.argmax(dim=1)
                    else:
                        pred = (output > 0.5).long().squeeze()
                    
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
                except Exception as e:
                    logger.warning(f"评估时出错: {e}")
                    return 0.5  # 返回随机准确率
        
        return correct / total if total > 0 else 0.5
    
    def calculate_fitness(self, metrics: Dict[str, float]) -> float:
        """计算适应度分数"""
        fitness = 0.0
        
        # 准确率贡献
        if 'accuracy' in metrics:
            fitness += self.config.accuracy_weight * metrics['accuracy']
        
        # 速度贡献 (越快越好)
        if 'inference_time_ms' in metrics:
            speed_score = 1.0 / (1.0 + metrics['inference_time_ms'] / 10.0)
            fitness += self.config.speed_weight * speed_score
        
        # 参数效率贡献 (越少越好)
        if 'parameters' in metrics:
            param_score = 1.0 / (1.0 + metrics['parameters'] / 1000000.0)
            fitness += self.config.parameter_weight * param_score
        
        # 可解释性贡献 (简化处理)
        explainability_score = 0.5  # 基础分数
        fitness += self.config.explainability_weight * explainability_score
        
        return fitness


# ============================== SHAP基础构建器 ==============================

class ShapBasedBuilder(BaseBuilder):
    """
    基于SHAP值的网络构建器
    
    根据SHAP特征重要性信息自动设计网络架构，
    重要特征获得更多关注和计算资源。
    """
    
    def __init__(self, config: BuilderConfig, shap_values: Optional[np.ndarray] = None):
        super().__init__(config)
        self.shap_values = shap_values
        self.feature_importance_ranking = None
        
        if shap_values is not None:
            self._analyze_shap_values()
    
    def _analyze_shap_values(self):
        """分析SHAP值，确定特征重要性排序"""
        if self.shap_values is None:
            return
        
        # 计算全局特征重要性
        if len(self.shap_values.shape) == 2:
            # (samples, features)
            importance = np.abs(self.shap_values).mean(axis=0)
        elif len(self.shap_values.shape) == 3:
            # (samples, sequence, features)
            importance = np.abs(self.shap_values).mean(axis=(0, 1))
        else:
            importance = np.abs(self.shap_values).flatten()
        
        # 特征重要性排序
        self.feature_importance_ranking = np.argsort(importance)[::-1]
        
        logger.info(f"SHAP分析完成，识别出 {len(importance)} 个特征")
        logger.info(f"Top 5重要特征: {self.feature_importance_ranking[:5]}")
    
    def build(self, input_dim: int, output_dim: int, 
              validation_data: Optional[torch.utils.data.DataLoader] = None) -> nn.Module:
        """
        基于SHAP值构建网络
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            validation_data: 验证数据
            
        Returns:
            构建的神经网络
        """
        if self.shap_values is None:
            logger.warning("未提供SHAP值，使用默认架构")
            return self._build_default_architecture(input_dim, output_dim)
        
        # 基于SHAP重要性设计架构
        architecture = self._design_shap_based_architecture(input_dim, output_dim)
        
        # 构建网络
        network_config = NetworkConfig(
            input_features=input_dim,
            output_features=output_dim,
            name="shap_optimized_net"
        )
        
        builder = NetworkBuilder(network_config)
        
        # 添加架构中定义的层
        for layer_name, layer_config in architecture.items():
            if layer_config['type'] == 'attention':
                block_config = BlockConfig(
                    input_dim=layer_config['input_dim'],
                    output_dim=layer_config['output_dim'],
                    name=layer_name
                )
                from .blocks import AttentionBlock
                block = AttentionBlock(block_config, num_heads=layer_config.get('num_heads', 8))
                
            elif layer_config['type'] == 'transformer':
                trans_config = TransformerConfig(
                    input_dim=layer_config['input_dim'],
                    output_dim=layer_config['output_dim'],
                    num_heads=layer_config.get('num_heads', 8),
                    ff_dim=layer_config.get('ff_dim', layer_config['input_dim'] * 4),
                    name=layer_name
                )
                block = TransformerBlock(trans_config)
                
            elif layer_config['type'] == 'mlp':
                mlp_config = MLPConfig(
                    input_dim=layer_config['input_dim'],
                    output_dim=layer_config['output_dim'],
                    hidden_dims=layer_config.get('hidden_dims', [layer_config['input_dim'] // 2]),
                    name=layer_name
                )
                block = MLPBlock(mlp_config)
            
            else:
                continue
                
            builder.add_block(block, layer_name)
        
        builder.auto_connect()
        model = builder.build()
        
        # 评估性能
        if validation_data:
            metrics = self.evaluate_architecture(model, validation_data)
            logger.info(f"SHAP优化网络性能: {metrics}")
        
        return model
    
    def _design_shap_based_architecture(self, input_dim: int, output_dim: int) -> Dict[str, Dict]:
        """基于SHAP重要性设计架构"""
        architecture = {}
        
        if self.feature_importance_ranking is None:
            return self._get_default_layers(input_dim, output_dim)
        
        # 计算重要特征比例
        num_features = len(self.feature_importance_ranking)
        top_features = min(num_features // 4, 64)  # 前25%或最多64个特征
        
        current_dim = input_dim
        
        # 特征选择/重点关注层
        if num_features > 100:  # 如果特征很多，添加注意力层
            architecture['feature_attention'] = {
                'type': 'attention',
                'input_dim': current_dim,
                'output_dim': current_dim,
                'num_heads': min(8, current_dim // 64)
            }
        
        # 特征压缩层（重点保留重要特征信息）
        compressed_dim = max(output_dim * 4, min(512, current_dim // 2))
        architecture['feature_compression'] = {
            'type': 'mlp',
            'input_dim': current_dim,
            'output_dim': compressed_dim,
            'hidden_dims': [min(1024, current_dim), compressed_dim * 2]
        }
        current_dim = compressed_dim
        
        # 基于重要性的变换层
        if top_features > 32:
            # 如果有足够的重要特征，使用Transformer
            architecture['importance_transformer'] = {
                'type': 'transformer',
                'input_dim': current_dim,
                'output_dim': current_dim,
                'num_heads': min(8, current_dim // 64),
                'ff_dim': current_dim * 2
            }
        else:
            # 否则使用MLP
            architecture['importance_mlp'] = {
                'type': 'mlp',
                'input_dim': current_dim,
                'output_dim': current_dim // 2,
                'hidden_dims': [current_dim, current_dim // 2]
            }
            current_dim = current_dim // 2
        
        # 输出层
        architecture['output_layer'] = {
            'type': 'mlp',
            'input_dim': current_dim,
            'output_dim': output_dim,
            'hidden_dims': [max(output_dim * 2, current_dim // 2)]
        }
        
        return architecture
    
    def _build_default_architecture(self, input_dim: int, output_dim: int) -> nn.Module:
        """构建默认架构"""
        network_config = NetworkConfig(
            input_features=input_dim,
            output_features=output_dim,
            name="default_net"
        )
        
        builder = NetworkBuilder(network_config)
        
        # 简单的MLP架构
        mlp_config = MLPConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[min(512, input_dim * 2), min(256, input_dim)],
            name="default_mlp"
        )
        mlp_block = MLPBlock(mlp_config)
        builder.add_block(mlp_block, "default_mlp")
        
        return builder.build()
    
    def _get_default_layers(self, input_dim: int, output_dim: int) -> Dict[str, Dict]:
        """获取默认层配置"""
        return {
            'default_mlp': {
                'type': 'mlp',
                'input_dim': input_dim,
                'output_dim': output_dim,
                'hidden_dims': [min(512, input_dim * 2), min(256, input_dim)]
            }
        }


# ============================== Fisher信息构建器 ==============================

class FisherBasedBuilder(BaseBuilder):
    """
    基于Fisher信息的网络构建器
    
    根据Fisher信息矩阵指导网络设计，
    为敏感参数分配更多计算资源。
    """
    
    def __init__(self, config: BuilderConfig, fisher_info: Optional[Dict[str, float]] = None):
        super().__init__(config)
        self.fisher_info = fisher_info
        self.sensitivity_ranking = None
        
        if fisher_info:
            self._analyze_fisher_information()
    
    def _analyze_fisher_information(self):
        """分析Fisher信息，确定参数敏感性"""
        if not self.fisher_info:
            return
        
        # 按Fisher信息值排序参数
        fisher_items = list(self.fisher_info.items())
        fisher_items.sort(key=lambda x: x[1], reverse=True)
        
        self.sensitivity_ranking = fisher_items
        
        logger.info(f"Fisher分析完成，分析了 {len(fisher_items)} 个参数")
        logger.info(f"Top 5敏感参数: {[name for name, _ in fisher_items[:5]]}")
    
    def build(self, input_dim: int, output_dim: int,
              validation_data: Optional[torch.utils.data.DataLoader] = None) -> nn.Module:
        """
        基于Fisher信息构建网络
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            validation_data: 验证数据
            
        Returns:
            构建的神经网络
        """
        if not self.fisher_info:
            logger.warning("未提供Fisher信息，使用默认架构")
            return self._build_robust_architecture(input_dim, output_dim)
        
        # 基于Fisher信息设计架构
        architecture = self._design_fisher_based_architecture(input_dim, output_dim)
        
        # 构建网络
        network_config = NetworkConfig(
            input_features=input_dim,
            output_features=output_dim,
            name="fisher_optimized_net"
        )
        
        builder = NetworkBuilder(network_config)
        
        # 添加架构中定义的层
        for layer_name, layer_config in architecture.items():
            if layer_config['type'] == 'robust_mlp':
                mlp_config = MLPConfig(
                    input_dim=layer_config['input_dim'],
                    output_dim=layer_config['output_dim'],
                    hidden_dims=layer_config.get('hidden_dims', []),
                    dropout=layer_config.get('dropout', 0.1),
                    name=layer_name
                )
                block = MLPBlock(mlp_config)
                
            elif layer_config['type'] == 'stable_transformer':
                trans_config = TransformerConfig(
                    input_dim=layer_config['input_dim'],
                    output_dim=layer_config['output_dim'],
                    num_heads=layer_config.get('num_heads', 4),  # 较少的头数，更稳定
                    ff_dim=layer_config.get('ff_dim', layer_config['input_dim'] * 2),
                    dropout=layer_config.get('dropout', 0.2),  # 更高的dropout
                    name=layer_name
                )
                block = TransformerBlock(trans_config)
            
            else:
                continue
                
            builder.add_block(block, layer_name)
        
        builder.auto_connect()
        model = builder.build()
        
        # 评估性能
        if validation_data:
            metrics = self.evaluate_architecture(model, validation_data)
            logger.info(f"Fisher优化网络性能: {metrics}")
        
        return model
    
    def _design_fisher_based_architecture(self, input_dim: int, output_dim: int) -> Dict[str, Dict]:
        """基于Fisher信息设计鲁棒架构"""
        architecture = {}
        current_dim = input_dim
        
        # 根据Fisher信息的分布决定架构策略
        if self.sensitivity_ranking:
            top_sensitivity = [value for _, value in self.sensitivity_ranking[:10]]
            avg_sensitivity = np.mean(top_sensitivity)
            
            # 如果参数敏感性很高，使用更保守的架构
            if avg_sensitivity > 0.1:
                # 高敏感性 -> 更稳定的架构
                hidden_dim = min(256, current_dim // 2)
                architecture['stable_layer1'] = {
                    'type': 'robust_mlp',
                    'input_dim': current_dim,
                    'output_dim': hidden_dim,
                    'hidden_dims': [hidden_dim * 2, hidden_dim],
                    'dropout': 0.3  # 高dropout增加鲁棒性
                }
                current_dim = hidden_dim
                
                architecture['stable_layer2'] = {
                    'type': 'robust_mlp',
                    'input_dim': current_dim,
                    'output_dim': output_dim,
                    'hidden_dims': [current_dim // 2],
                    'dropout': 0.2
                }
                
            else:
                # 低敏感性 -> 可以使用更复杂的架构
                hidden_dim = min(512, current_dim)
                architecture['complex_layer1'] = {
                    'type': 'stable_transformer',
                    'input_dim': current_dim,
                    'output_dim': hidden_dim,
                    'num_heads': 8,
                    'dropout': 0.1
                }
                current_dim = hidden_dim
                
                architecture['complex_layer2'] = {
                    'type': 'robust_mlp',
                    'input_dim': current_dim,
                    'output_dim': output_dim,
                    'hidden_dims': [current_dim // 2],
                    'dropout': 0.1
                }
        
        return architecture
    
    def _build_robust_architecture(self, input_dim: int, output_dim: int) -> nn.Module:
        """构建鲁棒的默认架构"""
        network_config = NetworkConfig(
            input_features=input_dim,
            output_features=output_dim,
            name="robust_net"
        )
        
        builder = NetworkBuilder(network_config)
        
        # 鲁棒的MLP架构
        mlp_config = MLPConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[min(256, input_dim), min(128, input_dim // 2)],
            dropout=0.2,  # 适度的dropout
            name="robust_mlp"
        )
        mlp_block = MLPBlock(mlp_config)
        builder.add_block(mlp_block, "robust_mlp")
        
        return builder.build()


# ============================== 自动架构搜索 ==============================

class AutoArchBuilder(BaseBuilder):
    """
    自动架构搜索构建器
    
    使用进化算法自动搜索最优网络架构
    """
    
    def __init__(self, config: BuilderConfig):
        super().__init__(config)
        self.population = []
        self.generation = 0
    
    def build(self, input_dim: int, output_dim: int,
              validation_data: Optional[torch.utils.data.DataLoader] = None) -> nn.Module:
        """
        使用进化算法搜索最优架构
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度  
            validation_data: 验证数据（用于评估）
            
        Returns:
            搜索到的最优网络
        """
        logger.info(f"开始自动架构搜索: {self.config.search_strategy}")
        
        # 初始化种群
        self._initialize_population(input_dim, output_dim)
        
        best_gene = None
        best_fitness = 0.0
        
        for generation in range(self.config.generations):
            self.generation = generation
            logger.info(f"进化第 {generation + 1} 代...")
            
            # 评估种群
            for gene in self.population:
                model = self._gene_to_model(gene, input_dim, output_dim)
                if model is not None:
                    metrics = self.evaluate_architecture(model, validation_data)
                    gene.fitness = self.calculate_fitness(metrics)
                    gene.metrics = metrics
                    
                    if gene.fitness > best_fitness:
                        best_fitness = gene.fitness
                        best_gene = gene
                        logger.info(f"发现更优架构，适应度: {best_fitness:.4f}")
            
            # 选择、交叉、变异
            self.population = self._evolve_population()
            
            # 早停条件
            if best_fitness > 0.95:
                logger.info("达到目标适应度，提前停止")
                break
        
        # 构建最优模型
        if best_gene:
            logger.info(f"搜索完成，最佳适应度: {best_fitness:.4f}")
            logger.info(f"最佳架构指标: {best_gene.metrics}")
            return self._gene_to_model(best_gene, input_dim, output_dim)
        else:
            logger.warning("搜索失败，返回默认架构")
            return self._build_default_model(input_dim, output_dim)
    
    def _initialize_population(self, input_dim: int, output_dim: int):
        """初始化种群"""
        self.population = []
        
        for _ in range(self.config.population_size):
            gene = self._create_random_gene(input_dim, output_dim)
            self.population.append(gene)
        
        logger.info(f"初始化种群，大小: {len(self.population)}")
    
    def _create_random_gene(self, input_dim: int, output_dim: int) -> ArchitectureGene:
        """创建随机基因"""
        blocks = []
        current_dim = input_dim
        
        # 随机层数
        num_layers = np.random.randint(2, min(self.config.max_layers, 8))
        
        for i in range(num_layers - 1):
            # 随机选择层类型
            layer_type = np.random.choice(['mlp', 'transformer', 'cnn'], p=[0.5, 0.3, 0.2])
            
            if layer_type == 'mlp':
                output_dim_layer = np.random.randint(
                    max(output_dim, current_dim // 4),
                    min(current_dim * 2, 1024)
                )
                blocks.append({
                    'type': 'mlp',
                    'input_dim': current_dim,
                    'output_dim': output_dim_layer,
                    'num_layers': np.random.randint(2, 4),
                    'dropout': np.random.uniform(0.1, 0.3)
                })
                current_dim = output_dim_layer
                
            elif layer_type == 'transformer':
                blocks.append({
                    'type': 'transformer',
                    'input_dim': current_dim,
                    'output_dim': current_dim,  # 保持维度
                    'num_heads': np.random.choice([4, 8, 12, 16]),
                    'ff_dim': current_dim * np.random.choice([2, 4]),
                    'dropout': np.random.uniform(0.1, 0.2)
                })
                
            elif layer_type == 'cnn':
                output_dim_layer = np.random.randint(
                    max(64, current_dim // 2),
                    min(current_dim * 2, 512)
                )
                blocks.append({
                    'type': 'cnn',
                    'input_dim': current_dim,
                    'output_dim': output_dim_layer,
                    'num_filters': np.random.choice([32, 64, 128, 256]),
                    'num_layers': np.random.randint(2, 4),
                    'kernel_size': np.random.choice([3, 5, 7])
                })
                current_dim = output_dim_layer
        
        # 输出层
        blocks.append({
            'type': 'mlp',
            'input_dim': current_dim,
            'output_dim': output_dim,
            'num_layers': 1,
            'dropout': 0.0
        })
        
        # 简单的顺序连接
        connections = [(i, i + 1, 'sequential') for i in range(len(blocks) - 1)]
        
        return ArchitectureGene(blocks=blocks, connections=connections)
    
    def _gene_to_model(self, gene: ArchitectureGene, input_dim: int, output_dim: int) -> Optional[nn.Module]:
        """将基因转换为实际模型"""
        try:
            network_config = NetworkConfig(
                input_features=input_dim,
                output_features=output_dim,
                name=f"evolved_net_gen{self.generation}"
            )
            
            builder = NetworkBuilder(network_config)
            
            # 添加每个块
            for i, block_config in enumerate(gene.blocks):
                block_name = f"block_{i}"
                
                if block_config['type'] == 'mlp':
                    config = MLPConfig(
                        input_dim=block_config['input_dim'],
                        output_dim=block_config['output_dim'],
                        num_layers=block_config.get('num_layers', 2),
                        dropout=block_config.get('dropout', 0.1),
                        name=block_name
                    )
                    block = MLPBlock(config)
                    
                elif block_config['type'] == 'transformer':
                    config = TransformerConfig(
                        input_dim=block_config['input_dim'],
                        output_dim=block_config['output_dim'],
                        num_heads=block_config.get('num_heads', 8),
                        ff_dim=block_config.get('ff_dim', block_config['input_dim'] * 4),
                        dropout=block_config.get('dropout', 0.1),
                        name=block_name
                    )
                    block = TransformerBlock(config)
                    
                elif block_config['type'] == 'cnn':
                    config = CNNConfig(
                        input_dim=block_config['input_dim'],
                        output_dim=block_config['output_dim'],
                        num_filters=block_config.get('num_filters', 64),
                        num_layers=block_config.get('num_layers', 3),
                        kernel_size=block_config.get('kernel_size', 3),
                        name=block_name
                    )
                    block = CNNBlock(config)
                
                else:
                    continue
                
                builder.add_block(block, block_name)
            
            # 自动连接
            builder.auto_connect()
            model = builder.build()
            
            # 检查参数数量
            total_params = sum(p.numel() for p in model.parameters())
            if total_params > self.config.max_parameters:
                return None
            
            return model
            
        except Exception as e:
            logger.warning(f"基因转换失败: {e}")
            return None
    
    def _evolve_population(self) -> List[ArchitectureGene]:
        """进化种群"""
        # 按适应度排序
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # 选择前50%作为父代
        elite_size = len(self.population) // 2
        elite = self.population[:elite_size]
        
        new_population = elite.copy()  # 保留精英
        
        # 生成新个体
        while len(new_population) < self.config.population_size:
            # 锦标赛选择
            parent1 = self._tournament_selection(elite)
            parent2 = self._tournament_selection(elite)
            
            # 交叉
            child = parent1.crossover(parent2)
            
            # 变异
            child = child.mutate(self.config.mutation_rate)
            
            new_population.append(child)
        
        return new_population
    
    def _tournament_selection(self, candidates: List[ArchitectureGene], k: int = 3) -> ArchitectureGene:
        """锦标赛选择"""
        tournament = np.random.choice(candidates, min(k, len(candidates)), replace=False)
        return max(tournament, key=lambda x: x.fitness)
    
    def _build_default_model(self, input_dim: int, output_dim: int) -> nn.Module:
        """构建默认模型"""
        network_config = NetworkConfig(
            input_features=input_dim,
            output_features=output_dim,
            name="default_evolved_net"
        )
        
        builder = NetworkBuilder(network_config)
        
        mlp_config = MLPConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[512, 256, 128],
            name="default_mlp"
        )
        mlp_block = MLPBlock(mlp_config)
        builder.add_block(mlp_block, "default_mlp")
        
        return builder.build()


# ============================== 性能优化器 ==============================

class PerformanceOptimizer:
    """
    性能优化器
    
    对已构建的网络进行性能优化，包括剪枝、量化、知识蒸馏等
    """
    
    def __init__(self):
        self.optimization_history = []
    
    def optimize_model(self, model: nn.Module, optimization_type: str = "pruning") -> nn.Module:
        """
        优化模型性能
        
        Args:
            model: 待优化模型
            optimization_type: 优化类型 ('pruning', 'quantization', 'distillation')
            
        Returns:
            优化后的模型
        """
        if optimization_type == "pruning":
            return self._prune_model(model)
        elif optimization_type == "quantization":
            return self._quantize_model(model)
        else:
            logger.warning(f"不支持的优化类型: {optimization_type}")
            return model
    
    def _prune_model(self, model: nn.Module) -> nn.Module:
        """模型剪枝"""
        try:
            import torch.nn.utils.prune as prune
            
            # 简单的结构化剪枝
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=0.2)
                    prune.remove(module, 'weight')
            
            logger.info("模型剪枝完成")
            return model
            
        except Exception as e:
            logger.warning(f"模型剪枝失败: {e}")
            return model
    
    def _quantize_model(self, model: nn.Module) -> nn.Module:
        """模型量化"""
        try:
            # 动态量化
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
            logger.info("模型量化完成")
            return quantized_model
            
        except Exception as e:
            logger.warning(f"模型量化失败: {e}")
            return model


class ExplainabilityOptimizer:
    """
    可解释性优化器
    
    优化模型的可解释性，添加注意力机制、中间层输出等
    """
    
    def __init__(self):
        pass
    
    def add_explainability_hooks(self, model: nn.Module) -> nn.Module:
        """为模型添加可解释性钩子"""
        
        def save_attention_hook(name):
            def hook(module, input, output):
                if hasattr(module, 'attention_weights') and module.attention_weights is not None:
                    if not hasattr(model, 'saved_attentions'):
                        model.saved_attentions = {}
                    model.saved_attentions[name] = module.attention_weights.detach()
            return hook
        
        def save_features_hook(name):
            def hook(module, input, output):
                if not hasattr(model, 'saved_features'):
                    model.saved_features = {}
                if isinstance(output, dict) and 'features' in output:
                    model.saved_features[name] = output['features'].detach()
                elif isinstance(output, torch.Tensor):
                    model.saved_features[name] = output.detach()
            return hook
        
        # 为每个模块注册钩子
        for name, module in model.named_modules():
            if hasattr(module, 'attention_weights'):
                module.register_forward_hook(save_attention_hook(name))
            module.register_forward_hook(save_features_hook(name))
        
        logger.info("可解释性钩子添加完成")
        return model


# 导出所有构建器
__all__ = [
    'BuilderConfig', 'ArchitectureGene', 'BaseBuilder',
    'ShapBasedBuilder', 'FisherBasedBuilder', 'AutoArchBuilder',
    'PerformanceOptimizer', 'ExplainabilityOptimizer'
]
