# SmartNet 开发文档

## 🎯 项目概述

SmartNet 是一个**双模式神经网络构建平台**，支持：
1. **可视化拖拽模式** - 适合快速原型和直观设计
2. **代码编辑模式** - 支持 SmartNet DSL 和 PyTorch 语法

### 核心特性
- ✅ iPad/移动端触摸拖拽支持
- ✅ 响应式设计（支持多种屏幕尺寸）
- ✅ SmartNet 领域专用语言（DSL）
- ✅ 实时语法检查和代码转换
- ✅ 组件接口自动匹配验证
- 🔄 DSL解析器调试中

---

## 📁 项目结构

```
DriftRec/
├── web_app/                    # Flask Web应用
│   ├── app.py                 # 主应用入口
│   ├── templates/             # HTML模板
│   │   ├── index.html        # 主页（拖拽界面）
│   │   ├── code-editor.html  # 代码编辑器
│   │   └── dsl-test.html     # DSL测试页面
│   └── static/               # 静态资源
│       ├── css/
│       │   ├── style.css     # 主样式（包含响应式设计）
│       │   └── code-editor.css  # 代码编辑器样式
│       └── js/
│           ├── app.js        # 主应用逻辑
│           ├── touch-support.js      # 触摸拖拽支持
│           ├── code-editor.js        # 代码编辑器控制器
│           └── smartnet-dsl.js       # DSL解析器（调试中）
├── smartnet/                  # 核心SmartNet库
├── data/                     # 数据集
├── models/                   # 训练好的模型
├── outputs/                  # 分析结果输出
└── scripts/                  # 训练和分析脚本
```

---

## 🚀 当前开发状态

### ✅ 已完成功能

#### 1. 移动端/iPad 支持
- **文件**: `web_app/static/js/touch-support.js`
- **功能**: 完整的 TouchDragSupport 类
- **特性**: 
  - 长按检测（150ms触发）
  - 拖拽指示器
  - 设备自适应优化
  - 三种屏幕尺寸适配

**关键代码示例**:
```javascript
class TouchDragSupport {
    constructor(app) {
        this.app = app;
        this.touchStartTime = 0;
        this.longPressThreshold = 150; // ms
        this.isDragging = false;
        this.dragIndicator = null;
        this.setupTouchHandlers();
    }
}
```

#### 2. 响应式设计
- **文件**: `web_app/static/css/style.css`
- **断点**:
  - iPad Pro: 1024-1366px
  - iPad 标准: 768-1023px  
  - 移动端: <768px

**CSS媒体查询示例**:
```css
@media screen and (max-width: 768px) {
    .component-library { width: 100%; }
    .canvas-container { padding: 5px; }
}
```

#### 3. 代码编辑器界面
- **文件**: `web_app/templates/code-editor.html`
- **功能**: 三面板布局，支持多语言切换
- **集成**: CodeMirror 编辑器，语法高亮

#### 4. SmartNet DSL 语言设计
- **语法示例**:
```dsl
network SimpleMLP:
    input_dim: 128
    output_dim: 10
    
    layer fc1: Linear(128, 64)
    layer fc2: Linear(64, 10)
    
    connection: fc1 -> fc2
    activation: relu
```

### 🔄 开发中功能

#### DSL解析器 (`smartnet-dsl.js`)
- **状态**: 基础框架完成，正在调试解析逻辑
- **问题**: 解析YAML格式时出现 "undefined" 错误
- **位置**: `parseYAML()` 和 `parseSmartNetDSL()` 函数

---

## 🐛 当前已知问题

### 1. DSL解析器错误
**现象**: 点击解析按钮显示 "构建失败，undefined"
**位置**: `web_app/static/js/smartnet-dsl.js`
**可能原因**: 
- parseYAML 函数实现不完整
- 参数解析逻辑有bug
- 错误处理机制缺失

**调试入口**: 访问 `http://localhost:5001/dsl-test` 测试

### 2. Flask应用依赖问题
**现象**: SmartNet核心库导入失败
**原因**: 缺少 PyTorch 依赖
**影响**: 网络构建和训练功能受限

---

## 🛠 开发环境设置

### 启动开发服务器
```bash
cd /home/coder-gw/DriftRec/web_app
python -c "
from app import create_app
app = create_app()
app.run(debug=True, host='0.0.0.0', port=5001)
"
```

### 访问地址
- 主页（拖拽界面）: http://localhost:5001/
- 代码编辑器: http://localhost:5001/code-editor  
- DSL测试页面: http://localhost:5001/dsl-test

---

## 🎯 下一步开发计划

### 优先级1: 修复DSL解析器
**目标**: 完善 `smartnet-dsl.js` 中的解析逻辑

**关键函数需要修复**:
1. `parseYAML(code)` - YAML格式解析
2. `parseSmartNetDSL(code)` - SmartNet DSL解析  
3. `parseParameters(paramsStr)` - 参数解析
4. `parseValue(value)` - 值类型解析

**测试用例**:
```yaml
name: SimpleMLP
parameters:
  input_dim: 128
  output_dim: 10
layers:
  - name: fc1
    type: Linear
    parameters:
      arg_0: 128
      arg_1: 64
```

### 优先级2: 完善错误处理
**任务**:
- 添加详细的错误信息显示
- 实现语法错误定位
- 增加解析过程日志

### 优先级3: 增强DSL功能
**扩展**:
- 支持更多层类型
- 添加激活函数配置
- 实现优化器参数设置

---

## 💡 开发技巧和注意事项

### 1. 调试DSL解析器
```javascript
// 在 smartnet-dsl.js 中添加调试日志
parse(code) {
    console.log('Parsing code:', code);
    try {
        const result = /* 解析逻辑 */;
        console.log('Parse result:', result);
        return result;
    } catch (error) {
        console.error('Parse error:', error);
        throw error;
    }
}
```

### 2. 测试响应式设计
- 使用浏览器开发者工具设备模拟
- 测试三种屏幕尺寸断点
- 验证触摸拖拽功能

### 3. 代码编辑器扩展
- CodeMirror 配置在 `code-editor.js`
- 语法高亮规则可自定义
- 主题切换功能已实现

---

## 📚 技术栈说明

### 前端技术
- **HTML5/CSS3**: 响应式布局
- **JavaScript ES6+**: 现代JS语法
- **CodeMirror**: 代码编辑器组件
- **Responsive Design**: 移动端适配

### 后端技术  
- **Flask**: Python Web框架
- **PyTorch**: 深度学习框架（待安装）
- **SmartNet**: 自研神经网络构建库

### 开发工具
- **VS Code**: 推荐的开发环境
- **浏览器开发者工具**: 前端调试
- **Flask调试模式**: 后端热重载

---

## 🔧 常用命令

### 启动应用
```bash
# 方法1: 直接运行
cd web_app && python app.py

# 方法2: 使用create_app (推荐)
cd web_app && python -c "from app import create_app; app = create_app(); app.run(debug=True, port=5001)"
```

### 检查语法错误
```bash
# 检查Python语法
python -m py_compile web_app/app.py

# 检查JavaScript (需要nodejs)
node -c web_app/static/js/smartnet-dsl.js
```

### 查看运行日志
```bash
# Flask应用日志会在终端显示
# 浏览器控制台可查看前端日志
```

---

## 📝 代码规范

### JavaScript
- 使用ES6+ 语法
- 类名使用 PascalCase
- 函数名使用 camelCase
- 常量使用 UPPER_CASE

### Python  
- 遵循PEP8规范
- 函数名使用snake_case
- 类名使用PascalCase
- 添加类型提示(推荐)

### CSS
- 使用BEM命名规范
- 响应式优先设计
- 组件化样式编写

---

## 🚨 紧急问题排查

### DSL解析器显示undefined
1. 打开浏览器开发者工具
2. 查看Console标签页错误信息
3. 检查 `smartnet-dsl.js` 第162行的 `parse()` 函数
4. 验证 `parseYAML()` 函数返回值

### 触摸拖拽不工作
1. 确认设备支持触摸事件
2. 检查 `touch-support.js` 是否正确加载
3. 验证 `TouchDragSupport` 类初始化

### 样式布局异常
1. 检查CSS媒体查询断点
2. 验证flexbox属性支持
3. 清除浏览器缓存

---

## 📞 联系方式

如有问题请查阅此文档或检查代码注释。关键文件都包含详细的功能说明和使用示例。

**开发重点**: 当前最紧急的任务是修复DSL解析器的undefined错误，建议从 `parseYAML()` 函数开始调试。

---

*文档更新时间: 2025年8月15日*
*开发者: Geoffrey Wang*
