# 📘 DriftRec: SmartNet可视化神经网络构建器

> 🧠 拖拽式神经网络构建 + 🚀 双RTX3090训练 + 🎯 智能架构推荐 + 📊 可解释性分析

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com)
[![GPU](https://img.shields.io/badge/GPU-RTX%203090%20x2-orange.svg)](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090)

---

## 🧭 Table of Contents

- [Overview](#overview) 
- [🌟 新增功能: Web可视化界面](#新增功能-web可视化界面)
- [Motivation](#motivation)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [🎨 Web界面使用指南](#web界面使用指南)
- [Supported Models](#supported-models)
- [Explainability Modules](#explainability-modules)
- [Experimental Results](#experimental-results)
- [Algorithm Comparison & Insights](#algorithm-comparison--insights)
- [Future Work](#future-work)
- [Citation](#citation)
- [License](#license)

---

## 🧩 Overview

# 🧠 SmartNet - 智能神经网络构建平台

[![开发状态](https://img.shields.io/badge/状态-开发中-yellow.svg)](https://github.com/GeoffreyWang1117/DriftReco)
[![版本](https://img.shields.io/badge/版本-v0.3.0--dev-blue.svg)](CHANGELOG.md)
[![平台支持](https://img.shields.io/badge/平台-Web%20%7C%20iPad%20%7C%20Mobile-green.svg)](#)

> **双模式神经网络构建平台** - 支持可视化拖拽和代码编程两种开发方式

## 🌟 项目特色

### 双重开发模式
- **🎨 可视化拖拽模式**: 直观的组件拖拽，适合快速原型设计
- **💻 代码编程模式**: 支持SmartNet DSL、PyTorch语法和YAML配置

### 全平台支持  
- **🖥️ 桌面端**: 完整功能体验
- **📱 iPad/平板**: 专门优化的触摸拖拽
- **📱 手机端**: 响应式移动界面

### 智能特性
- **🔧 自动接口匹配**: 智能验证组件间的参数兼容性
- **📊 实时参数估算**: 显示网络规模和内存需求
- **⚡ 语法实时检查**: 代码编写时的即时反馈

---

## 🚀 快速开始

### 环境要求
- Python 3.8+
- Flask 2.0+
- 现代浏览器 (Chrome, Firefox, Safari, Edge)

### 一键启动
```bash
# 克隆项目
git clone https://github.com/GeoffreyWang1117/DriftReco.git
cd DriftRec

# 启动应用
cd web_app
python -c "from app import create_app; app = create_app(); app.run(debug=True, port=5001)"
```

### 访问地址
- 🎨 **拖拽界面**: http://localhost:5001/
- 💻 **代码编辑器**: http://localhost:5001/code-editor
- 🧪 **DSL测试**: http://localhost:5001/dsl-test

---

## 📱 界面预览

### 拖拽模式界面
```
┌─────────────┬──────────────────────┬─────────────┐
│  组件库     │       画布区域        │  属性面板   │
│  ┌───────┐  │  ┌─────┐  ┌─────┐    │  ┌───────┐  │
│  │ MLP   │  │  │ FC1 │──│ FC2 │    │  │ 参数  │  │
│  │ CNN   │  │  └─────┘  └─────┘    │  │ 配置  │  │
│  │ RNN   │  │                      │  └───────┘  │
│  └───────┘  │                      │             │
└─────────────┴──────────────────────┴─────────────┘
```

### 代码编辑模式界面  
```
┌──────────────────┬──────────────────┬──────────────┐
│   代码编辑器     │    网络预览      │   语法帮助   │
│  ┌─────────────┐ │  ┌─────────────┐ │ ┌──────────┐ │
│  │ SmartNet    │ │  │ 网络结构图  │ │ │ DSL语法  │ │
│  │ DSL/PyTorch │ │  │ 参数统计    │ │ │ 示例代码 │ │
│  │ YAML        │ │  │ 错误提示    │ │ │ API参考  │ │
│  └─────────────┘ │  └─────────────┘ │ └──────────┘ │
└──────────────────┴──────────────────┴──────────────┘
```

---

## 🛠️ 核心技术栈

### 前端技术
- **HTML5/CSS3**: 响应式布局设计
- **JavaScript ES6+**: 现代化前端开发
- **CodeMirror**: 专业代码编辑器
- **Touch Events API**: 移动端触摸支持

### 后端技术
- **Flask**: 轻量级Python Web框架  
- **SmartNet**: 自研神经网络构建库
- **PyTorch**: 深度学习框架支持

### 开发工具
- **响应式设计**: 支持多种屏幕尺寸
- **实时热重载**: 开发时自动刷新
- **语法高亮**: 多语言代码支持

---

## 💡 SmartNet DSL 语言

### 基础语法示例
```smartnet
# 简单多层感知机
network SimpleMLP:
    input_dim: 128
    output_dim: 10
    learning_rate: 0.001
    
    layer fc1: Linear(128, 64)
    layer fc2: Linear(64, 32)  
    layer output: Linear(32, 10)
    
    connection: fc1 -> fc2 -> output
    activation: relu
    dropout: 0.1
```

### YAML配置格式
```yaml
name: TransformerNet
parameters:
  input_dim: 512
  learning_rate: 0.0001

layers:
  - name: embedding
    type: Linear
    parameters:
      arg_0: 512
      arg_1: 512
  - name: attention  
    type: MultiHeadAttention
    parameters:
      embed_dim: 512
      num_heads: 8

connections:
  - from: embedding
    to: attention
    type: sequential
```

### PyTorch语法支持
```python
import torch
import torch.nn as nn

class CustomNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

---

## 📊 项目状态

### 完成功能 ✅
- [x] 响应式Web界面设计
- [x] iPad/移动端触摸拖拽支持
- [x] 代码编辑器完整实现
- [x] SmartNet DSL语言设计
- [x] 多语言语法支持
- [x] 实时参数估算功能

### 开发中功能 🔄  
- [ ] **DSL解析器调试** (当前优先级)
- [ ] 网络构建API完善
- [ ] 训练工作流集成

### 规划功能 📋
- [ ] 高级DSL特性扩展
- [ ] 云端训练支持
- [ ] 协作开发功能
- [ ] 插件生态系统

---

## 🎯 开发进度

```
总体进度: ████████░░ 80%

模块完成情况:
├── 前端界面框架:    ██████████ 100%
├── 移动端适配:      ██████████ 100%  
├── 代码编辑器:      ████████░░  90%
├── DSL语言设计:     ████████░░  85%
├── DSL解析器:       ██████░░░░  60% ⚠️
├── 网络构建:        █████░░░░░  50%
└── 训练集成:        ███░░░░░░░  30%
```

**当前阻塞问题**: DSL解析器中 `parseYAML()` 函数返回undefined错误

---

## 📚 文档导航

### 开发者文档
- 📖 **[详细开发文档](README_DEVELOPMENT.md)** - 完整的开发指南和项目结构说明
- 🚀 **[快速问题解决](QUICK_START_GUIDE.md)** - 常见问题和调试方法  
- 🏗️ **[代码架构说明](CODE_ARCHITECTURE.md)** - 项目架构和模块关系
- 📝 **[版本历史记录](CHANGELOG.md)** - 开发日志和版本变更

### 用户文档
- 🎨 **拖拽界面使用**: 访问主页开始使用
- 💻 **代码编辑教程**: 在代码编辑器页面查看语法帮助
- 🧪 **DSL语法测试**: 使用测试页面验证代码

---

## 🔧 开发指南

### 当前开发重点
1. **🚨 紧急任务**: 修复DSL解析器undefined错误
2. **📍 调试入口**: http://localhost:5001/dsl-test  
3. **🎯 关键文件**: `web_app/static/js/smartnet-dsl.js`
4. **🔍 问题函数**: `parseYAML()` 和 `parseSmartNetDSL()`

### 开发环境设置
```bash
# 启动开发服务器
cd web_app
python -c "from app import create_app; app = create_app(); app.run(debug=True, port=5001)"

# 在浏览器中测试
# - 主界面: http://localhost:5001/
# - DSL测试: http://localhost:5001/dsl-test
# - 打开开发者工具查看控制台
```

### 代码贡献流程
1. 阅读 [开发文档](README_DEVELOPMENT.md) 了解项目结构
2. 使用 [快速指南](QUICK_START_GUIDE.md) 解决常见问题
3. 参考 [架构说明](CODE_ARCHITECTURE.md) 理解代码组织
4. 从当前优先级最高的任务开始开发
5. 提交前确保功能正常并更新相关文档

---

## 🌐 技术特色

### 🎨 智能UI设计
- **自适应布局**: 根据屏幕尺寸自动调整界面
- **触摸优化**: 专为触屏设备优化的交互体验  
- **视觉反馈**: 丰富的动画和状态提示

### 🧠 智能代码助手
- **语法高亮**: 多语言代码着色显示
- **实时检查**: 编写代码时的即时语法验证
- **智能提示**: 代码自动补全和错误修正建议

### ⚡ 高性能架构
- **模块化设计**: 松耦合的组件架构
- **异步处理**: 非阻塞的用户交互体验
- **缓存优化**: 智能的资源加载和缓存策略

---

## 🤝 社区与支持

### 贡献方式
- 🐛 **Bug报告**: 发现问题请创建Issue
- 💡 **功能建议**: 欢迎提出新的想法和改进建议  
- 🔧 **代码贡献**: Fork项目并提交Pull Request
- 📚 **文档完善**: 帮助改进项目文档

### 技术支持
- 📖 **文档首选**: 查阅项目文档获取帮助
- 🔍 **问题排查**: 使用调试工具和测试页面
- 💬 **社区讨论**: 在Issues中参与技术讨论

---

## 📄 开源协议

本项目采用 [Apache License 2.0](LICENSE) 开源协议。

---

## ⭐ 致谢

感谢所有为SmartNet项目做出贡献的开发者和用户！

### 核心技术致谢
- **Flask**: 提供了优秀的Python Web框架
- **CodeMirror**: 强大的代码编辑器组件
- **PyTorch**: 深度学习框架支持

---

*最后更新: 2025年8月15日*  
*开发者: Geoffrey Wang*  
*项目状态: 积极开发中*

🎨 **拖拽式可视化界面**：像搭积木一样构建神经网络
⚡ **GPU优化训练**：双RTX 3090并行训练，智能内存管理  
📊 **可解释性分析**：基于SHAP和Fisher信息的模型理解
🔄 **模块化架构**：支持多种网络组件的即插即用

### 🌟 新增功能: Web可视化界面

🎉 **最新亮点**：DriftRec现已集成完整的Web可视化界面！

#### 🚀 立即体验

```bash
# 启动Web应用
cd web_app
./start_server.sh

# 打开浏览器访问
open http://localhost:5000
```

#### ✨ 界面特性

- **🎨 拖拽构建**：可视化拖拽组件构建神经网络
- **⚡ 实时训练**：在线GPU训练，实时监控进度  
- **🔧 参数调整**：可视化参数配置和优化
- **📊 智能分析**：集成SHAP和Fisher信息分析
- **🎯 约束优化**：针对双RTX 3090的性能优化

#### 🧱 支持的组件类型

| 组件 | 描述 | 适用场景 |
|------|------|----------|
| 🟢 MLP层 | 多层感知机 | 结构化数据处理 |
| 🔵 Transformer层 | 多头注意力 | 序列建模 |
| 🟠 CNN层 | 卷积网络 | 图像和局部特征 |
| 🟣 RNN层 | 循环网络 | 时序数据建模 |

详细使用方法请参考：[📚 Web界面使用指南](/web_app/使用指南.md)

---

## 🎯 Motivation

Modern recommender systems often suffer from:
- Performance degradation under feature or behavior drift
- Lack of interpretability in prediction outcomes
- Inadequate handling of cold-start and sparse features

**DriftRec** aims to bridge this gap by providing:
- Modular implementations of baseline CTR models
- Uncertainty and robustness analysis tools
- Visual insight into model decision-making processes

---

## ✨ Key Features

### 🎯 模块化设计
- ✅ 统一的核心框架 (`driftrec/`)，消除代码冗余
- ✅ 即插即用的模型接口 (BaseModel, ModelTrainer)
- ✅ 灵活的配置管理系统 (ConfigManager)
- ✅ 重新组织的脚本结构，提高可维护性

### 📊 综合分析能力  
- ✅ 端到端推荐系统 + 可解释性分析流水线
- ✅ 集成SHAP值分析 (全局和局部解释)
- ✅ Fisher信息矩阵特征敏感性估计
- ✅ 多维度模型基准测试框架

### 🌊 高级检测功能
- ✅ 实时数据漂移检测和监控
- ✅ 时间序列模式和趋势分析  
- ✅ 异常检测和预警系统
- ✅ 季节性和周期性分析

### 📋 智能报告生成
- ✅ 自动化Markdown详细报告
- ✅ 交互式HTML仪表板
- ✅ 多模型对比可视化
- ✅ 执行洞察和改进建议

### 🔧 易于扩展
- ✅ 支持Amazon Reviews 2023 (All_Beauty)数据集
- ✅ 轻松扩展到其他模型 (DIN, xDeepFM, LLMs等)
- ✅ 模块化架构便于新功能集成

---

## 🧱 Project Structure

```bash
DriftRec/
├── web_app/                     # 🌐 Web可视化界面 (新增)
│   ├── app.py                   # Flask应用主程序
│   ├── templates/index.html     # 拖拽式界面模板
│   ├── static/                  # CSS/JS静态资源
│   ├── start_server.sh          # 后台启动脚本
│   ├── stop_server.sh           # 停止服务脚本
│   ├── status_server.sh         # 状态监控脚本
│   ├── test_client.py           # API测试客户端
│   └── 使用指南.md               # 详细使用文档
├── SmartNet/                    # 🧠 SmartNet核心框架 (新增)
│   ├── core/                    # 核心组件 (BaseComponent, NetworkBuilder)
│   ├── networks/                # 网络定义 (MLP, Transformer, CNN, RNN)
│   ├── training/                # 训练模块 (GPUTrainer, ProgressMonitor)
│   ├── analysis/                # 分析工具 (SHAP, Fisher, Explainer)
│   └── utils/                   # 工具函数 (参数估算, GPU检测)
├── driftrec/                    # 🎯 原始分析框架
│   ├── core/                    # 基础组件 (BaseModel, DataLoader, ConfigManager)
│   ├── explainability/          # 可解释性分析 (SHAP, Fisher, Attention)  
│   ├── benchmark/               # 基准测试框架 (MetricsCalculator, ModelComparator)
│   ├── drift/                   # 漂移检测 (DriftDetector, TemporalAnalyzer)
│   └── visualization/           # 可视化生成 (ReportGenerator, DashboardGenerator)
├── scripts/                     # 📁 分析脚本
│   ├── data_processing/         # 数据处理脚本
│   ├── training/                # 模型训练脚本  
│   ├── analysis/                # 分析评估脚本
│   └── experiments/             # 实验对比脚本
├── data/                        # Amazon Beauty数据集
├── models/                      # 训练好的模型文件
├── outputs/                     # 分析结果和可视化
├── definitions/                 # 模型定义
├── run_analysis.py              # 🚀 统一分析入口
└── README.md
```

---

## 🚀 Quick Start

### 🎨 方式1: Web可视化界面 (推荐)

```bash
# Step 1: 启动Web应用
cd web_app
./start_server.sh

# Step 2: 打开浏览器访问
open http://localhost:5000

# Step 3: 在Web界面中操作
# 1. 🎯 配置网络参数 (输入/输出维度)
# 2. 🧱 拖拽组件到画布 (MLP/Transformer/CNN/RNN)  
# 3. ⚙️ 调整组件参数
# 4. 🔨 构建网络架构
# 5. 🚀 开始GPU训练

# Step 4: 停止服务
./stop_server.sh
```

#### 🎭 界面预览
```
┌─────────────────────────────────────────────────────────────┐
│  🧠 SmartNet - 智能神经网络构建器                              │
│  [清空画布] [构建网络] [开始训练]                              │
├─────────────┬─────────────────────────┬─────────────────────┤
│  🧱 组件面板  │      � 网络画布        │    ⚙️ 属性面板      │
│             │                        │                    │
│ • MLP层     │   [拖拽组件到此构建网络]    │ 选择组件来编辑属性    │
│ • CNN层     │                        │                    │  
│ • Transform │     🔗 智能连接         │ 📊 参数实时预览      │
│ • RNN层     │                        │                    │
└─────────────┴─────────────────────────┴─────────────────────┘
│  🟢 系统就绪 | GPU: 2x RTX 3090 | 构建成功 | 训练中...     │
└─────────────────────────────────────────────────────────────┘
```

### �🔥 方式2: 统一分析框架

```bash
# Step 1: Clone the repo
git clone https://github.com/yourname/DriftRec.git
cd DriftRec

# Step 2: Install dependencies  
pip install -r requirements.txt

# Step 3: 运行完整分析 (一键执行)
python run_analysis.py --mode all --models dlrm dcnv2 autoint deepfm

# Step 4: 查看结果
# - 报告: outputs/reports/
# - 仪表板: outputs/dashboards/
# - 详细数据: outputs/[model_name]/
```

## 🎨 Web界面使用指南

### 🌟 核心功能

#### 1. 🎯 网络配置
- **网络名称**：为构建的网络命名
- **输入维度**：数据特征维度 (1-2048)  
- **输出维度**：预测输出维度 (1-100)
- **学习率**：训练学习率 (0.0001-0.1)

#### 2. 🧱 组件拖拽
支持4种核心组件类型：

| 组件 | 图标 | 功能描述 | 主要参数 |
|------|------|----------|----------|
| MLP层 | 🟢 | 多层感知机 | 输入/输出/隐藏维度, Dropout |
| Transformer层 | 🔵 | 多头注意力机制 | 注意力头数, FFN维度 |
| CNN层 | 🟠 | 卷积神经网络 | 卷积核大小, 通道数, 步长 |
| RNN层 | 🟣 | 循环神经网络 | 隐藏维度, 层数, 类型(LSTM/GRU) |

#### 3. ⚙️ 参数调整
- **实时编辑**：选择组件即可在右侧面板编辑参数
- **智能验证**：参数范围自动验证，防止配置错误
- **即时预览**：参数修改实时反映在界面上

#### 4. 🔨 网络构建
- **一键构建**：自动连接组件并验证架构
- **参数估算**：预估网络参数量和内存使用
- **约束检查**：确保符合双RTX 3090的限制

#### 5. 🚀 GPU训练  
- **训练配置**：轮数、批次大小、学习率设置
- **实时监控**：训练进度、损失曲线、GPU使用率
- **智能优化**：自动批次大小调整，内存优化

### 🎯 使用流程

```
开始 → 设置网络参数 → 拖拽组件 → 调整参数 → 构建网络 → 开始训练 → 查看结果
  ↑                                                                    ↓
  ← ← ← ← ← ← ← ← ← 保存模型/继续优化 ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ←
```

### 🔧 高级功能

#### 约束优化 (双RTX 3090)
- **最大深度**：10层网络限制
- **参数上限**：5000万参数限制  
- **批次推荐**：根据显存自动推荐32-128
- **内存预估**：构建前预估GPU内存使用

#### 性能监控
- **实时图表**：损失曲线和准确率变化
- **GPU状态**：温度、利用率、显存使用  
- **训练日志**：详细的训练过程记录
- **自动保存**：训练检查点自动保存

详细使用说明请参考：[📖 完整使用指南](web_app/使用指南.md)

### 🔧 模块化调用示例

```python
from driftrec.core import ConfigManager, ModelTrainer
from driftrec.explainability import ExplainabilityFramework
from driftrec.benchmark import BenchmarkRunner  
from driftrec.drift import DriftDetector
from driftrec.visualization import ReportGenerator

# 统一配置管理
config = ConfigManager().load_config('config.json')

# 可解释性分析
explainability = ExplainabilityFramework()
results = explainability.run_comprehensive_analysis(model, data)

# 基准测试
benchmark = BenchmarkRunner()
metrics = benchmark.run_benchmark(models_dict, test_data)

# 生成报告
report_gen = ReportGenerator('outputs/reports')
report_gen.generate_comparison_report(results)
```

### 📜 传统方式 (原始脚本)

```bash
# Step 1: 数据预处理
python scripts/data_processing/step1_load_beauty_parquet.py
python scripts/data_processing/step1d5_preprocess_full.py

# Step 2: 训练模型
python scripts/training/step3_train_deepfm_ms.py

# Step 3: 分析评估
python scripts/analysis/step4_analyze_deepfm_ms_full.py
python scripts/analysis/step4e_export_deepfm_analysis.py
```

---

## 🧠 Supported Models

| Model     | Feature Interaction       | Interpretability | Use Case            |
|-----------|---------------------------|------------------|---------------------|
| DeepFM    | FM + MLP                  | Medium           | Sparse CTR datasets |
| AutoInt   | Self-attention-based      | High             | Cold-start & drift  |
| (planned) xDeepFM | CIN + DNN         | Medium           | High-order modeling |

---

## 🔍 Explainability Modules

### SHAP Analysis
- Measures per-feature contribution to model prediction
- Visualizes both local (single sample) and global importance

### Fisher Information Matrix
- Estimates feature sensitivity and model robustness
- Useful for detecting potential overfitting and drift-prone features

*(Visualizations will be shown here, e.g. SHAP summary plots and Fisher bar charts)*

---

## 📊 Experimental Results

| Model     | AUC   | LogLoss | Notes                  |
|-----------|-------|---------|------------------------|
| DeepFM    | 0.742 | 0.511   | Baseline               |
| AutoInt   | 0.755 | 0.498   | Strong under cold-start|

*(Insert SHAP and Fisher visualization images under `/results` folder)*

---

## 📘 Algorithm Comparison & Insights

```markdown
| Model     | Interaction Method | Interpretability | Training Cost | Notes                      |
|-----------|--------------------|------------------|----------------|----------------------------|
| DeepFM    | FM + DNN           | Medium           | Medium         | Good for sparse features   |
| AutoInt   | Attention layers   | High             | High           | Better at structure learning |
| xDeepFM   | CIN + DNN          | Medium           | High           | Planned extension          |
```

---

## � 新增功能 (2025年8月更新)

### 📈 Benchmark框架
- ✅ 综合性能评估系统 (`step6_benchmark_framework.py`)
- ✅ 多维度模型对比 (准确性、可解释性、效率、鲁棒性)
- ✅ 自动化报告生成

### 🔍 高级可解释性分析
- ✅ 集成多种可解释性方法 (`step7_advanced_explainability.py`)
- ✅ SHAP + Fisher信息 + 注意力权重分析
- ✅ 组合洞察生成和可视化

### 🌊 漂移检测与时间分析
- ✅ 数据漂移实时监控 (`step8_drift_detection.py`)
- ✅ 时间序列模式分析
- ✅ 季节性和趋势检测

### 📊 交互式仪表板
- ✅ HTML可视化界面 (`step9_dashboard_generator.py`)
- ✅ 多模型对比视图
- ✅ 实时洞察和建议

### 🤖 一键执行系统
- ✅ 主执行脚本 (`run_full_analysis.py`)
- ✅ 模块化执行选项
- ✅ 自动依赖检查和报告生成

## 🛠️ 使用新功能

```bash
# 运行完整的benchmark和可解释性分析
python scripts/run_full_analysis.py --mode full

# 只运行benchmark分析
python scripts/run_full_analysis.py --mode benchmark

# 只运行高级可解释性分析
python scripts/run_full_analysis.py --mode explainability

# 运行漂移检测
python scripts/run_full_analysis.py --mode drift

# 生成交互式仪表板
python scripts/run_full_analysis.py --mode dashboard
```

## 📊 多种可解释性结合方案

### 1. 分层解释策略
- **全局层面**: Fisher信息矩阵识别关键参数
- **特征层面**: SHAP值分析特征重要性
- **实例层面**: 局部SHAP值解释单个预测

### 2. 跨模型一致性分析
- 比较不同模型的特征重要性
- 识别模型间的共同模式和差异
- 提供鲁棒的特征洞察

### 3. 时间动态监控
- 实时检测特征重要性漂移
- 监控模型稳定性变化
- 预警系统自动告警

### 4. 交互式探索
- 可视化仪表板支持深度探索
- 多维度对比分析
- 用户友好的洞察呈现

## 🛠️ Future Work

- [ ] Add DIN和xDeepFM作为新基线模型
- [ ] 集成FAISS的混合排序检索
- [ ] 使用开源LLM的提示式解释
- [ ] 扩展到MIND或Criteo数据集
- [ ] 添加A/B测试框架用于在线评估
- [ ] 开发模型压缩和加速技术

---

## 📚 Citation

```bibtex
@misc{driftrec2025,
  author = {Your Name},
  title = {DriftRec: Uncertainty-Aware and Explainable Recommender Framework},
  year = {2025},
  note = {GitHub project},
  url = {https://github.com/yourname/DriftRec}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
