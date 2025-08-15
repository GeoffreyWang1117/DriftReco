"""
SmartNet可解释性工具

提供专门的可解释性分析工具，与主框架的explainability模块互补，
专注于SmartNet构建的网络的解释和可视化。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import logging
from abc import ABC, abstractmethod
import json
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================== 可解释性配置 ==============================

@dataclass
class ExplainabilityConfig:
    """可解释性分析配置"""
    save_intermediate: bool = True
    track_gradients: bool = True
    compute_saliency: bool = True
    generate_plots: bool = True
    plot_save_dir: str = "explainability_plots"
    
    # 分析范围
    analyze_attention: bool = True
    analyze_features: bool = True
    analyze_gradients: bool = True
    
    # 可视化配置
    plot_format: str = "png"
    plot_dpi: int = 300
    colormap: str = "viridis"


# ============================== 基础可解释层 ==============================

class ExplainableLayer(nn.Module, ABC):
    """
    可解释层基类
    
    所有需要提供解释信息的层都应继承此类
    """
    
    def __init__(self, name: str = "explainable_layer"):
        super().__init__()
        self.name = name
        self.explanation_data = {}
        self.hooks = []
        
    @abstractmethod
    def get_explanation(self) -> Dict[str, Any]:
        """获取解释信息"""
        pass
    
    def register_explanation_hooks(self):
        """注册解释性钩子"""
        def forward_hook(module, input, output):
            self._collect_forward_info(input, output)
        
        def backward_hook(module, grad_input, grad_output):
            self._collect_backward_info(grad_input, grad_output)
        
        self.hooks.append(self.register_forward_hook(forward_hook))
        self.hooks.append(self.register_backward_hook(backward_hook))
    
    def _collect_forward_info(self, input, output):
        """收集前向传播信息"""
        if isinstance(input, tuple):
            input = input[0]
        
        self.explanation_data['forward'] = {
            'input_shape': input.shape if hasattr(input, 'shape') else None,
            'output_shape': output.shape if hasattr(output, 'shape') else None,
            'input_stats': self._tensor_stats(input) if torch.is_tensor(input) else None,
            'output_stats': self._tensor_stats(output) if torch.is_tensor(output) else None
        }
    
    def _collect_backward_info(self, grad_input, grad_output):
        """收集反向传播信息"""
        if grad_input is not None and grad_input[0] is not None:
            self.explanation_data['backward'] = {
                'grad_input_stats': self._tensor_stats(grad_input[0]),
                'grad_output_stats': self._tensor_stats(grad_output[0]) if grad_output[0] is not None else None
            }
    
    def _tensor_stats(self, tensor: torch.Tensor) -> Dict[str, float]:
        """计算张量统计信息"""
        if tensor is None or not torch.is_tensor(tensor):
            return {}
        
        return {
            'mean': float(tensor.mean()),
            'std': float(tensor.std()),
            'min': float(tensor.min()),
            'max': float(tensor.max()),
            'norm': float(tensor.norm())
        }
    
    def clear_hooks(self):
        """清理钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class ExplainableLinear(ExplainableLayer):
    """可解释的线性层"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, name: str = "explainable_linear"):
        super().__init__(name)
        self.linear = nn.Linear(in_features, out_features, bias)
        self.register_explanation_hooks()
        
        # 权重重要性追踪
        self.weight_importance = None
        self.bias_importance = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算权重重要性（基于激活值）
        with torch.no_grad():
            input_magnitude = torch.abs(x).mean(dim=0)
            weight_magnitude = torch.abs(self.linear.weight)
            self.weight_importance = weight_magnitude * input_magnitude.unsqueeze(0)
        
        return self.linear(x)
    
    def get_explanation(self) -> Dict[str, Any]:
        explanation = {
            'layer_type': 'linear',
            'parameters': {
                'in_features': self.linear.in_features,
                'out_features': self.linear.out_features,
                'has_bias': self.linear.bias is not None
            },
            'weight_stats': self._tensor_stats(self.linear.weight),
            'weight_importance': self.weight_importance.cpu().numpy() if self.weight_importance is not None else None
        }
        
        if self.linear.bias is not None:
            explanation['bias_stats'] = self._tensor_stats(self.linear.bias)
        
        explanation.update(self.explanation_data)
        return explanation


# ============================== 注意力可视化器 ==============================

class AttentionVisualizer:
    """
    注意力权重可视化器
    
    专门用于可视化Transformer和其他注意力机制的权重
    """
    
    def __init__(self, config: ExplainabilityConfig):
        self.config = config
        
    def visualize_attention_weights(self, attention_weights: torch.Tensor, 
                                  tokens: Optional[List[str]] = None,
                                  save_path: Optional[str] = None) -> str:
        """
        可视化注意力权重
        
        Args:
            attention_weights: 注意力权重张量 (batch, heads, seq, seq)
            tokens: 标记列表
            save_path: 保存路径
            
        Returns:
            保存的图片路径
        """
        if not self.config.generate_plots:
            return ""
        
        # 转换为numpy
        if torch.is_tensor(attention_weights):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # 取第一个样本和头的平均
        if len(attention_weights.shape) == 4:
            # (batch, heads, seq, seq) -> (seq, seq)
            attn_matrix = attention_weights[0].mean(axis=0)
        elif len(attention_weights.shape) == 3:
            # (heads, seq, seq) -> (seq, seq)
            attn_matrix = attention_weights.mean(axis=0)
        else:
            # (seq, seq)
            attn_matrix = attention_weights
        
        # 创建热力图
        plt.figure(figsize=(10, 8))
        
        # 如果有token标签，使用它们
        if tokens is not None and len(tokens) == attn_matrix.shape[0]:
            sns.heatmap(attn_matrix, 
                       xticklabels=tokens[:attn_matrix.shape[1]], 
                       yticklabels=tokens[:attn_matrix.shape[0]],
                       cmap=self.config.colormap,
                       cbar=True,
                       square=True)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
        else:
            sns.heatmap(attn_matrix, 
                       cmap=self.config.colormap,
                       cbar=True,
                       square=True)
        
        plt.title("Attention Weights Heatmap")
        plt.xlabel("Key Positions")
        plt.ylabel("Query Positions")
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            import os
            os.makedirs(self.config.plot_save_dir, exist_ok=True)
            save_path = f"{self.config.plot_save_dir}/attention_heatmap.{self.config.plot_format}"
        
        plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"注意力热力图已保存: {save_path}")
        return save_path
    
    def visualize_attention_heads(self, attention_weights: torch.Tensor,
                                save_path: Optional[str] = None) -> str:
        """
        可视化多个注意力头
        
        Args:
            attention_weights: 注意力权重 (batch, heads, seq, seq)
            save_path: 保存路径
            
        Returns:
            保存的图片路径
        """
        if not self.config.generate_plots:
            return ""
        
        if torch.is_tensor(attention_weights):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        if len(attention_weights.shape) != 4:
            logger.warning("注意力权重维度不正确，需要4D张量 (batch, heads, seq, seq)")
            return ""
        
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # 取第一个样本
        attn_heads = attention_weights[0]  # (heads, seq, seq)
        
        # 计算子图网格
        cols = min(4, num_heads)
        rows = (num_heads + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for head_idx in range(num_heads):
            ax = axes[head_idx] if num_heads > 1 else axes[0]
            
            im = ax.imshow(attn_heads[head_idx], cmap=self.config.colormap, aspect='auto')
            ax.set_title(f"Head {head_idx + 1}")
            ax.set_xlabel("Key Positions")
            ax.set_ylabel("Query Positions")
            
            # 添加颜色条
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 隐藏多余的子图
        for head_idx in range(num_heads, len(axes)):
            axes[head_idx].set_visible(False)
        
        plt.suptitle("Multi-Head Attention Visualization")
        plt.tight_layout()
        
        # 保存
        if save_path is None:
            import os
            os.makedirs(self.config.plot_save_dir, exist_ok=True)
            save_path = f"{self.config.plot_save_dir}/attention_heads.{self.config.plot_format}"
        
        plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"多头注意力可视化已保存: {save_path}")
        return save_path


# ============================== 特征重要性追踪器 ==============================

class FeatureImportanceTracker:
    """
    特征重要性追踪器
    
    追踪和分析网络中各个特征的重要性变化
    """
    
    def __init__(self, config: ExplainabilityConfig):
        self.config = config
        self.importance_history = []
        self.feature_names = None
        
    def compute_gradient_based_importance(self, model: nn.Module, 
                                        input_data: torch.Tensor,
                                        target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        基于梯度计算特征重要性
        
        Args:
            model: 模型
            input_data: 输入数据
            target: 目标值
            
        Returns:
            特征重要性字典
        """
        model.eval()
        input_data.requires_grad_(True)
        
        # 前向传播
        output = model(input_data)
        
        # 计算损失
        if output.dim() > 1 and output.size(1) > 1:
            # 多分类
            loss = F.cross_entropy(output, target)
        else:
            # 回归或二分类
            loss = F.mse_loss(output.squeeze(), target.float())
        
        # 反向传播
        loss.backward()
        
        # 计算输入梯度重要性
        input_gradients = input_data.grad
        if input_gradients is not None:
            # 绝对值梯度作为重要性指标
            importance = torch.abs(input_gradients).mean(dim=0)
        else:
            importance = torch.zeros_like(input_data[0])
        
        # 收集各层的重要性
        layer_importance = {}
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight.grad is not None:
                layer_importance[name] = torch.abs(module.weight.grad).mean()
        
        return {
            'input_importance': importance,
            'layer_importance': layer_importance
        }
    
    def compute_integrated_gradients(self, model: nn.Module,
                                   input_data: torch.Tensor,
                                   target: torch.Tensor,
                                   baseline: Optional[torch.Tensor] = None,
                                   steps: int = 50) -> torch.Tensor:
        """
        计算集成梯度
        
        Args:
            model: 模型
            input_data: 输入数据
            target: 目标值  
            baseline: 基线输入
            steps: 积分步数
            
        Returns:
            集成梯度重要性
        """
        if baseline is None:
            baseline = torch.zeros_like(input_data)
        
        # 创建插值路径
        alphas = torch.linspace(0, 1, steps + 1, device=input_data.device)
        
        gradients = []
        for alpha in alphas:
            # 插值输入
            interpolated_input = baseline + alpha * (input_data - baseline)
            interpolated_input.requires_grad_(True)
            
            # 前向传播
            output = model(interpolated_input)
            
            # 计算损失
            if output.dim() > 1 and output.size(1) > 1:
                loss = F.cross_entropy(output, target)
            else:
                loss = F.mse_loss(output.squeeze(), target.float())
            
            # 计算梯度
            grad = torch.autograd.grad(loss, interpolated_input, create_graph=False)[0]
            gradients.append(grad)
        
        # 集成梯度计算
        avg_gradients = torch.stack(gradients).mean(dim=0)
        integrated_gradients = (input_data - baseline) * avg_gradients
        
        return integrated_gradients
    
    def track_importance_over_time(self, importance: Dict[str, torch.Tensor], step: int):
        """追踪重要性随时间的变化"""
        importance_snapshot = {
            'step': step,
            'input_importance': importance.get('input_importance', torch.tensor(0.0)),
            'layer_importance': importance.get('layer_importance', {})
        }
        
        self.importance_history.append(importance_snapshot)
        
        # 限制历史记录长度
        if len(self.importance_history) > 1000:
            self.importance_history = self.importance_history[-500:]
    
    def visualize_importance_trend(self, save_path: Optional[str] = None) -> str:
        """可视化重要性趋势"""
        if not self.config.generate_plots or not self.importance_history:
            return ""
        
        steps = [h['step'] for h in self.importance_history]
        
        # 绘制输入重要性趋势
        plt.figure(figsize=(12, 8))
        
        # 输入重要性（取前几个维度）
        plt.subplot(2, 1, 1)
        input_importance_over_time = []
        for h in self.importance_history:
            imp = h['input_importance']
            if torch.is_tensor(imp):
                # 取前5个特征的重要性
                top_features = imp.flatten()[:5].cpu().numpy()
                input_importance_over_time.append(top_features)
            else:
                input_importance_over_time.append([0] * 5)
        
        input_importance_array = np.array(input_importance_over_time)
        for i in range(input_importance_array.shape[1]):
            plt.plot(steps, input_importance_array[:, i], label=f'Feature {i+1}', alpha=0.7)
        
        plt.title("Input Feature Importance Over Time")
        plt.xlabel("Training Step")
        plt.ylabel("Importance Score")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 层重要性趋势
        plt.subplot(2, 1, 2)
        
        # 收集所有层的名称
        all_layer_names = set()
        for h in self.importance_history:
            all_layer_names.update(h['layer_importance'].keys())
        
        # 绘制主要层的重要性
        main_layers = list(all_layer_names)[:5]  # 只显示前5个层
        
        for layer_name in main_layers:
            layer_importance_over_time = []
            for h in self.importance_history:
                layer_imp = h['layer_importance'].get(layer_name, torch.tensor(0.0))
                if torch.is_tensor(layer_imp):
                    layer_importance_over_time.append(float(layer_imp.cpu()))
                else:
                    layer_importance_over_time.append(0.0)
            
            plt.plot(steps, layer_importance_over_time, label=layer_name, alpha=0.7)
        
        plt.title("Layer Importance Over Time")
        plt.xlabel("Training Step")
        plt.ylabel("Average Weight Gradient")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        if save_path is None:
            import os
            os.makedirs(self.config.plot_save_dir, exist_ok=True)
            save_path = f"{self.config.plot_save_dir}/importance_trend.{self.config.plot_format}"
        
        plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"重要性趋势图已保存: {save_path}")
        return save_path
    
    def get_top_features(self, k: int = 10) -> List[Tuple[int, float]]:
        """获取最重要的特征"""
        if not self.importance_history:
            return []
        
        # 使用最新的重要性数据
        latest_importance = self.importance_history[-1]['input_importance']
        if torch.is_tensor(latest_importance):
            importance_values = latest_importance.flatten().cpu().numpy()
            
            # 获取top-k特征
            top_indices = np.argsort(importance_values)[::-1][:k]
            top_features = [(int(idx), float(importance_values[idx])) for idx in top_indices]
            
            return top_features
        
        return []


# ============================== 网络解释器 ==============================

class NetworkExplainer:
    """
    网络整体解释器
    
    对整个SmartNet网络提供全面的解释分析
    """
    
    def __init__(self, model: nn.Module, config: ExplainabilityConfig):
        self.model = model
        self.config = config
        self.attention_visualizer = AttentionVisualizer(config)
        self.feature_tracker = FeatureImportanceTracker(config)
        
        # 收集网络信息
        self.network_info = self._analyze_network_structure()
        
    def _analyze_network_structure(self) -> Dict[str, Any]:
        """分析网络结构"""
        info = {
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'layers': {},
            'connections': []
        }
        
        # 分析每个层
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # 叶子节点
                layer_info = {
                    'type': module.__class__.__name__,
                    'parameters': sum(p.numel() for p in module.parameters())
                }
                
                # 添加层特定信息
                if isinstance(module, nn.Linear):
                    layer_info.update({
                        'in_features': module.in_features,
                        'out_features': module.out_features,
                        'bias': module.bias is not None
                    })
                elif isinstance(module, nn.Conv1d):
                    layer_info.update({
                        'in_channels': module.in_channels,
                        'out_channels': module.out_channels,
                        'kernel_size': module.kernel_size,
                        'stride': module.stride,
                        'padding': module.padding
                    })
                
                info['layers'][name] = layer_info
        
        return info
    
    def explain_prediction(self, input_data: torch.Tensor, 
                          target: Optional[torch.Tensor] = None,
                          method: str = "gradient") -> Dict[str, Any]:
        """
        解释单个预测
        
        Args:
            input_data: 输入数据
            target: 目标值（如果可用）
            method: 解释方法 ("gradient", "integrated_gradients", "attention")
            
        Returns:
            解释结果字典
        """
        explanation = {
            'network_info': self.network_info,
            'input_shape': input_data.shape,
            'method': method
        }
        
        self.model.eval()
        
        # 前向传播获取预测和中间结果
        with torch.no_grad():
            if hasattr(self.model, 'forward') and 'return_explanations' in self.model.forward.__code__.co_varnames:
                # 如果模型支持返回解释
                result = self.model(input_data, return_explanations=True)
                if isinstance(result, dict):
                    explanation['prediction'] = result.get('prediction', result.get('output'))
                    explanation['attention_weights'] = result.get('attention_weights', {})
                    explanation['intermediate_features'] = result.get('intermediate_features', {})
                else:
                    explanation['prediction'] = result
            else:
                explanation['prediction'] = self.model(input_data)
        
        # 基于梯度的解释
        if method == "gradient" and target is not None:
            importance = self.feature_tracker.compute_gradient_based_importance(
                self.model, input_data, target
            )
            explanation['feature_importance'] = importance
            
        elif method == "integrated_gradients" and target is not None:
            integrated_grads = self.feature_tracker.compute_integrated_gradients(
                self.model, input_data, target
            )
            explanation['integrated_gradients'] = integrated_grads
        
        # 注意力解释
        if method == "attention" and 'attention_weights' in explanation:
            attention_paths = []
            for layer_name, attention in explanation['attention_weights'].items():
                if attention is not None:
                    # 保存注意力可视化
                    path = self.attention_visualizer.visualize_attention_weights(
                        attention, save_path=f"{self.config.plot_save_dir}/{layer_name}_attention.{self.config.plot_format}"
                    )
                    attention_paths.append(path)
            explanation['attention_visualizations'] = attention_paths
        
        return explanation
    
    def generate_comprehensive_report(self, input_data: torch.Tensor,
                                    target: Optional[torch.Tensor] = None,
                                    save_path: Optional[str] = None) -> str:
        """
        生成综合解释报告
        
        Args:
            input_data: 输入数据
            target: 目标值
            save_path: 报告保存路径
            
        Returns:
            报告文件路径
        """
        # 获取所有类型的解释
        gradient_explanation = self.explain_prediction(input_data, target, "gradient")
        attention_explanation = self.explain_prediction(input_data, target, "attention")
        
        # 生成报告内容
        report_lines = []
        report_lines.append("# SmartNet网络解释性报告")
        report_lines.append(f"生成时间: {np.datetime64('now')}")
        report_lines.append("")
        
        # 网络结构信息
        report_lines.append("## 网络结构分析")
        info = self.network_info
        report_lines.append(f"- 总参数量: {info['total_parameters']:,}")
        report_lines.append(f"- 可训练参数: {info['trainable_parameters']:,}")
        report_lines.append(f"- 层数: {len(info['layers'])}")
        report_lines.append("")
        
        # 主要层信息
        report_lines.append("### 主要网络层")
        for layer_name, layer_info in list(info['layers'].items())[:10]:
            report_lines.append(f"- **{layer_name}**: {layer_info['type']} "
                               f"({layer_info['parameters']:,} 参数)")
        report_lines.append("")
        
        # 预测结果
        prediction = gradient_explanation.get('prediction')
        if prediction is not None:
            if torch.is_tensor(prediction):
                pred_value = prediction.detach().cpu().numpy()
                if pred_value.ndim == 0:
                    report_lines.append(f"## 预测结果: {pred_value:.4f}")
                elif pred_value.ndim == 1 and len(pred_value) == 1:
                    report_lines.append(f"## 预测结果: {pred_value[0]:.4f}")
                else:
                    report_lines.append(f"## 预测结果: {pred_value}")
            else:
                report_lines.append(f"## 预测结果: {prediction}")
            report_lines.append("")
        
        # 特征重要性
        if 'feature_importance' in gradient_explanation:
            report_lines.append("## 特征重要性分析")
            importance = gradient_explanation['feature_importance']
            
            if 'input_importance' in importance:
                input_imp = importance['input_importance']
                if torch.is_tensor(input_imp):
                    top_features = self.feature_tracker.get_top_features(k=10)
                    if top_features:
                        report_lines.append("### Top 10重要特征")
                        for i, (feature_idx, importance_score) in enumerate(top_features):
                            report_lines.append(f"{i+1}. 特征 {feature_idx}: {importance_score:.6f}")
                        report_lines.append("")
        
        # 注意力分析
        if 'attention_weights' in attention_explanation and attention_explanation['attention_weights']:
            report_lines.append("## 注意力机制分析")
            attention_weights = attention_explanation['attention_weights']
            
            for layer_name, attention in attention_weights.items():
                if attention is not None:
                    if torch.is_tensor(attention):
                        attn_stats = {
                            'mean': float(attention.mean()),
                            'std': float(attention.std()),
                            'max': float(attention.max()),
                            'min': float(attention.min())
                        }
                        report_lines.append(f"### {layer_name}")
                        report_lines.append(f"- 平均注意力: {attn_stats['mean']:.4f}")
                        report_lines.append(f"- 注意力标准差: {attn_stats['std']:.4f}")
                        report_lines.append(f"- 最大注意力: {attn_stats['max']:.4f}")
                        report_lines.append("")
        
        # 建议和总结
        report_lines.append("## 解释性总结")
        report_lines.append("### 模型透明度评估")
        
        # 计算透明度分数
        transparency_score = 0.5  # 基础分数
        
        if 'attention_weights' in attention_explanation and attention_explanation['attention_weights']:
            transparency_score += 0.3  # 有注意力机制
        
        if 'feature_importance' in gradient_explanation:
            transparency_score += 0.2  # 有特征重要性
        
        transparency_score = min(1.0, transparency_score)
        
        report_lines.append(f"- 整体透明度评分: {transparency_score:.2f}/1.0")
        
        if transparency_score > 0.8:
            report_lines.append("- 评价: 高度可解释，决策过程清晰")
        elif transparency_score > 0.6:
            report_lines.append("- 评价: 中等可解释性，可识别关键特征")
        else:
            report_lines.append("- 评价: 有限可解释性，建议增加解释性组件")
        
        report_lines.append("")
        report_lines.append("### 改进建议")
        
        # 基于分析结果给出建议
        suggestions = []
        
        if transparency_score < 0.7:
            suggestions.append("考虑添加更多注意力机制以增强可解释性")
        
        if 'feature_importance' in gradient_explanation:
            importance = gradient_explanation['feature_importance']
            if 'layer_importance' in importance and len(importance['layer_importance']) > 10:
                suggestions.append("网络层数较多，考虑可视化中间层特征")
        
        if not suggestions:
            suggestions.append("当前模型具有良好的可解释性")
        
        for suggestion in suggestions:
            report_lines.append(f"- {suggestion}")
        
        # 保存报告
        report_content = "\n".join(report_lines)
        
        if save_path is None:
            import os
            os.makedirs(self.config.plot_save_dir, exist_ok=True)
            save_path = f"{self.config.plot_save_dir}/comprehensive_report.md"
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"综合解释报告已保存: {save_path}")
        return save_path


# 导出所有组件
__all__ = [
    'ExplainabilityConfig', 'ExplainableLayer', 'ExplainableLinear',
    'AttentionVisualizer', 'FeatureImportanceTracker', 'NetworkExplainer'
]
