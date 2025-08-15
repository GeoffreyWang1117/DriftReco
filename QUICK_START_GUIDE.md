# SmartNet 快速问题解决指南

## 🚨 当前紧急问题：DSL解析器返回undefined

### 问题现象
- 在DSL测试页面 (http://localhost:5001/dsl-test) 点击"解析代码"
- 显示"构建失败，undefined"错误
- 浏览器控制台可能显示JavaScript错误

### 问题定位
文件位置：`web_app/static/js/smartnet-dsl.js`
关键函数：
- `parse()` (第162行) - 主解析入口
- `parseYAML()` (第182行) - YAML格式解析
- `parseSmartNetDSL()` (约第270行) - SmartNet DSL解析

### 快速调试步骤

#### 1. 检查浏览器控制台
```javascript
// 在浏览器控制台运行
console.log(window.smartNetDSL); // 应该显示SmartNetDSL类实例
```

#### 2. 测试基础解析功能
```javascript
// 在浏览器控制台测试
try {
    const result = window.smartNetDSL.parse("name: test");
    console.log(result);
} catch (error) {
    console.error("Parse error:", error);
}
```

#### 3. 检查parseYAML函数实现
当前问题可能在于parseYAML函数没有正确返回网络对象结构。

#### 4. 修复建议
在 `smartnet-dsl.js` 中找到 `parseYAML()` 函数，确保它返回如下结构：
```javascript
return {
    name: '网络名称',
    parameters: {}, // 网络参数
    layers: [],     // 层数组
    connections: [], // 连接数组
    errors: []      // 错误数组
};
```

### 开发环境检查清单

#### Flask应用状态
```bash
# 检查应用是否运行
curl http://localhost:5001/dsl-test
# 应该返回HTML页面内容

# 检查JavaScript文件加载
curl http://localhost:5001/static/js/smartnet-dsl.js
# 应该返回JavaScript代码
```

#### 文件完整性检查
```bash
# 检查关键文件是否存在
ls -la web_app/static/js/smartnet-dsl.js
ls -la web_app/templates/dsl-test.html
```

---

## 🛠 开发工作流

### 每次开发开始前
1. 启动Flask应用
```bash
cd /home/coder-gw/DriftRec/web_app
python -c "from app import create_app; app = create_app(); app.run(debug=True, port=5001)"
```

2. 在浏览器中打开测试页面
- 主界面: http://localhost:5001/
- 代码编辑器: http://localhost:5001/code-editor
- DSL测试: http://localhost:5001/dsl-test

3. 打开浏览器开发者工具查看控制台

### 修改代码后
1. 保存文件（Flask自动重启）
2. 刷新浏览器页面
3. 检查控制台是否有新错误
4. 测试相关功能

---

## 📋 功能测试检查表

### 基础功能
- [ ] Flask应用正常启动
- [ ] 主页面可以访问
- [ ] 代码编辑器页面可以访问
- [ ] DSL测试页面可以访问

### 拖拽功能（桌面端）
- [ ] 可以从组件库拖拽组件到画布
- [ ] 拖拽时有视觉反馈
- [ ] 组件可以在画布上移动
- [ ] 可以连接组件

### 触摸功能（移动端）
- [ ] 长按可以触发拖拽
- [ ] 触摸拖拽有指示器显示
- [ ] 在iPad和手机上布局正常

### DSL解析功能
- [ ] 可以解析SmartNet DSL语法
- [ ] 可以解析YAML格式
- [ ] 显示解析结果和错误信息
- [ ] 参数估算功能正常

### 代码编辑器功能
- [ ] 语法高亮正常显示
- [ ] 可以切换不同语言模式
- [ ] 实时预览功能
- [ ] 代码转换功能

---

## 🔍 常见错误及解决方案

### 错误1: "SmartNet导入失败"
**原因**: 缺少PyTorch依赖
**解决**: 暂时忽略，不影响前端开发

### 错误2: "Port 5000 is in use"
**解决**: 使用端口5001或杀死占用进程
```bash
lsof -ti:5000 | xargs kill -9
```

### 错误3: JavaScript "undefined"
**常见原因**:
1. 函数没有返回值
2. 变量未正确初始化
3. 异步操作没有正确处理

**调试方法**:
1. 在函数开始添加 `console.log`
2. 检查所有 `return` 语句
3. 验证变量定义

### 错误4: CSS样式不生效
**解决**:
1. 清除浏览器缓存 (Ctrl+Shift+R)
2. 检查CSS文件路径
3. 验证媒体查询断点

---

## 📊 项目开发进度

### 已完成 (✅)
- [x] 基础Flask应用框架
- [x] 响应式UI设计
- [x] 触摸拖拽支持
- [x] 代码编辑器界面
- [x] DSL语法定义
- [x] 基础解析器框架

### 进行中 (🔄)
- [ ] DSL解析器调试
- [ ] 错误处理完善
- [ ] 组件接口验证

### 待开发 (📋)
- [ ] 网络训练集成
- [ ] 模型导出功能
- [ ] 高级DSL特性
- [ ] 性能优化

---

## 💻 推荐开发工具设置

### VS Code 扩展
- Python
- JavaScript (ES6) code snippets
- HTML CSS Support
- Flask Snippets
- Bracket Pair Colorizer

### 浏览器调试
- Chrome DevTools
- 移动设备模拟器
- Console日志监控
- Network面板检查

### 文件监控
```bash
# 监控文件变化
find web_app -name "*.py" -o -name "*.js" -o -name "*.css" | entr -r python web_app/app.py
```

---

## 📝 下次开发时从这里开始

1. **启动开发环境**
   ```bash
   cd /home/coder-gw/DriftRec/web_app
   python -c "from app import create_app; app = create_app(); app.run(debug=True, port=5001)"
   ```

2. **打开测试页面**
   - http://localhost:5001/dsl-test

3. **重点检查文件**
   - `web_app/static/js/smartnet-dsl.js` (parseYAML函数)
   - 浏览器控制台错误信息

4. **测试用例**
   ```yaml
   name: SimpleMLP
   parameters:
     input_dim: 128
   layers:
     - name: fc1
       type: Linear
   ```

5. **成功标志**
   - 解析器不再显示"undefined"
   - 能正确显示网络结构信息
   - 参数估算功能正常

**记住**: 当前最关键的是修复DSL解析器，其他功能已基本完善。

---

*快速指南更新: 2025年8月15日*
*作者: Geoffrey Wang*
