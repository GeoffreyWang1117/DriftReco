# SmartNet 代码架构图

## 🏗️ 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    SmartNet 平台                             │
├─────────────────────────────────────────────────────────────┤
│  前端 (Browser)                                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   拖拽模式      │  │   代码编辑模式   │  │  DSL测试    │ │
│  │   (index.html)  │  │(code-editor.html)│  │(dsl-test.html)│ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
│           │                     │                   │       │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              JavaScript 层                             │ │
│  │  ┌─────────┐ ┌─────────────┐ ┌─────────────────────────┐ │ │
│  │  │ app.js  │ │touch-support│ │     smartnet-dsl.js     │ │ │
│  │  │(主逻辑) │ │   .js       │ │      (DSL解析器)        │ │ │
│  │  └─────────┘ └─────────────┘ └─────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  后端 (Flask)                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                   app.py                               │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │ │
│  │  │    路由     │  │  API接口    │  │   网络构建      │ │ │
│  │  │   处理      │  │   处理      │  │     逻辑        │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘ │ │
│  └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  SmartNet 核心库                                          │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  NetworkBuilder │ MLPBlock │ TransformerBlock │ etc.   │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧩 前端组件详解

### 1. 主界面 (index.html)
```
index.html
├── 组件库面板 (Component Library)
│   ├── MLP组件
│   ├── Transformer组件
│   ├── CNN组件
│   └── RNN组件
├── 画布区域 (Canvas)
│   ├── 拖拽目标区域
│   ├── 组件实例显示
│   └── 连接线绘制
└── 控制面板 (Control Panel)
    ├── 网络配置
    ├── 训练参数
    └── 系统信息
```

**关键JavaScript文件**:
- `app.js`: 主应用逻辑，拖拽处理
- `touch-support.js`: iPad/移动端触摸支持

### 2. 代码编辑器 (code-editor.html)
```
code-editor.html
├── 编辑器面板 (Editor Panel)
│   ├── 语言选择 (SmartNet DSL / PyTorch / YAML)
│   ├── CodeMirror编辑器
│   └── 语法高亮
├── 预览面板 (Preview Panel)
│   ├── 网络结构图
│   ├── 参数统计
│   └── 错误显示
└── 帮助面板 (Help Panel)
    ├── 语法说明
    ├── 示例代码
    └── API参考
```

**关键JavaScript文件**:
- `code-editor.js`: 编辑器控制器
- `smartnet-dsl.js`: DSL解析引擎

---

## 🔧 核心模块分析

### SmartNet DSL 解析器 (`smartnet-dsl.js`)

#### 类结构
```javascript
class SmartNetDSL {
    // 构造函数
    constructor()
    
    // === 主要解析方法 ===
    parse(code)                    // 主解析入口
    parseSmartNetDSL(code)         // SmartNet DSL解析
    parseYAML(code)                // YAML格式解析
    
    // === 辅助解析方法 ===
    parseParameters(paramsStr)      // 参数解析
    parseValue(value)              // 值类型解析
    parseLayer(line, lineNum)      // 层定义解析
    parseConnection(line, lineNum)  // 连接解析
    
    // === 工具方法 ===
    isYAMLFormat(code)             // 格式检测
    validateNetwork(network)        // 网络验证
    estimateParameters(network)     // 参数估算
    
    // === 示例和帮助 ===
    getExamples()                  // 获取示例代码
    getSyntaxHelp(language)        // 获取语法帮助
}
```

#### 当前问题定位
```javascript
// 🚨 问题函数 (需要调试)
parseYAML(code) {
    // 这个函数可能没有正确返回网络对象
    // 导致显示 "undefined" 错误
}

parseSmartNetDSL(code) {
    // 这个函数的实现可能不完整
    // 需要检查所有代码路径是否都有返回值
}
```

---

## 📱 响应式设计架构

### CSS断点系统
```css
/* 桌面端 */
@media screen and (min-width: 1367px) {
    .container { max-width: 1200px; }
    .component-library { width: 250px; }
}

/* iPad Pro */
@media screen and (max-width: 1366px) and (min-width: 1024px) {
    .component-library { width: 220px; }
    .canvas-container { padding: 15px; }
}

/* iPad 标准 */
@media screen and (max-width: 1023px) and (min-width: 768px) {
    .component-library { width: 200px; }
    .properties-panel { width: 100%; }
}

/* 移动端 */
@media screen and (max-width: 767px) {
    .component-library { width: 100%; }
    .canvas-container { padding: 5px; }
}
```

### 触摸支持层次
```javascript
TouchDragSupport
├── 设备检测
│   ├── isMobileDevice()
│   ├── isTablet() 
│   └── hasTouch()
├── 事件处理
│   ├── handleTouchStart()
│   ├── handleTouchMove()
│   └── handleTouchEnd()
└── UI适配
    ├── createDragIndicator()
    ├── optimizeForMobile()
    └── adjustSensitivity()
```

---

## 🌐 后端API架构

### Flask路由结构
```
app.py
├── / (GET)                    # 主页
├── /code-editor (GET)         # 代码编辑器页面
├── /dsl-test (GET)           # DSL测试页面
├── /api/components (GET)      # 获取组件列表
├── /api/build (POST)         # 构建网络
├── /api/train (POST)         # 开始训练
├── /api/training/<id> (GET)  # 训练状态
└── /api/system_info (GET)    # 系统信息
```

### 数据流向
```
浏览器 ──HTTP请求──> Flask路由 ──调用──> SmartNet核心库
   │                                          │
   └──────JSON响应──────────────────────────────┘
```

---

## 📄 文件依赖关系

### HTML模板依赖
```
templates/
├── index.html
│   ├── → static/css/style.css
│   ├── → static/js/app.js  
│   └── → static/js/touch-support.js
├── code-editor.html
│   ├── → static/css/code-editor.css
│   ├── → static/js/code-editor.js
│   └── → static/js/smartnet-dsl.js
└── dsl-test.html
    └── → static/js/smartnet-dsl.js
```

### JavaScript模块依赖
```
app.js (主应用)
├── 依赖: 无外部依赖
└── 调用: TouchDragSupport

touch-support.js (触摸支持)
├── 依赖: app.js中的SmartNetApp实例
└── 调用: 无

code-editor.js (代码编辑器)
├── 依赖: CodeMirror库, smartnet-dsl.js
└── 调用: SmartNetDSL

smartnet-dsl.js (DSL解析器)
├── 依赖: 无
└── 调用: 无 (独立模块)
```

---

## 🧪 测试架构

### 单元测试结构 (规划中)
```
tests/
├── frontend/
│   ├── test_dsl_parser.js    # DSL解析器测试
│   ├── test_touch_support.js # 触摸功能测试
│   └── test_ui_components.js # UI组件测试
├── backend/
│   ├── test_api_routes.py    # API路由测试
│   ├── test_network_build.py # 网络构建测试
│   └── test_training.py      # 训练功能测试
└── integration/
    ├── test_full_workflow.py # 完整工作流测试
    └── test_cross_browser.js # 跨浏览器测试
```

### 当前手动测试流程
1. **启动应用**: `python -c "from app import create_app; app = create_app(); app.run(debug=True, port=5001)"`
2. **访问测试页面**: http://localhost:5001/dsl-test
3. **测试DSL解析**: 输入代码点击解析按钮
4. **检查浏览器控制台**: 查看错误信息
5. **验证响应式布局**: 调整浏览器窗口大小

---

## 🔄 开发工作流程图

```
开始开发
    │
    ├── 启动Flask应用
    │   └── python -c "from app import create_app; ..."
    │
    ├── 打开浏览器测试页面  
    │   ├── 主页: localhost:5001/
    │   ├── 编辑器: localhost:5001/code-editor
    │   └── 测试: localhost:5001/dsl-test
    │
    ├── 修改代码
    │   ├── 前端: web_app/static/
    │   └── 后端: web_app/app.py
    │
    ├── 测试功能
    │   ├── 刷新浏览器
    │   ├── 检查控制台
    │   └── 验证功能
    │
    └── 提交代码 (当功能完成时)
```

---

## 📊 项目状态总览

### 完成度统计
```
总体进度: ████████░░ 80%

模块完成情况:
├── 前端UI框架:      ██████████ 100%
├── 响应式设计:      ██████████ 100%
├── 触摸拖拽:       ██████████ 100%
├── 代码编辑器:      ████████░░  90%
├── DSL语言设计:     ████████░░  85%
├── DSL解析器:       ██████░░░░  60% ⚠️
├── 网络构建API:     █████░░░░░  50%
└── 训练集成:       ███░░░░░░░  30%
```

### 下个里程碑
**目标**: 完成DSL解析器调试，实现完整的代码编辑->解析->构建流程
**预估时间**: 1-2天
**关键任务**: 修复 `parseYAML()` 和 `parseSmartNetDSL()` 函数

---

*架构文档更新: 2025年8月15日*
*这份文档应该能帮助您快速理解整个项目的结构和当前开发状态*
*作者: Geoffrey Wang*
