// 代码编辑器主控制器

class CodeEditor {
    constructor() {
        this.currentLanguage = 'smartnet';
        this.editor = null;
        this.currentNetwork = null;
        this.autoSaveInterval = null;
        this.lastSaveTime = Date.now();
        
        this.initializeEditor();
        this.setupEventListeners();
        this.loadSyntaxHelp();
        this.startAutoSave();
        this.checkSystemStatus();
    }

    // 初始化编辑器
    initializeEditor() {
        const textarea = document.getElementById('codeEditor');
        
        // 配置CodeMirror
        this.editor = CodeMirror.fromTextArea(textarea, {
            lineNumbers: true,
            mode: 'python', // 默认使用Python语法高亮
            theme: 'dracula',
            autoCloseBrackets: true,
            matchBrackets: true,
            styleActiveLine: true,
            indentUnit: 4,
            indentWithTabs: false,
            lineWrapping: true,
            foldGutter: true,
            gutters: ['CodeMirror-linenumbers', 'CodeMirror-foldgutter']
        });

        // 设置默认代码
        this.loadExample('simple_mlp');

        // 监听编辑器变化
        this.editor.on('change', () => {
            this.onCodeChange();
        });

        // 自适应高度
        this.editor.setSize(null, '100%');
    }

    // 设置事件监听器
    setupEventListeners() {
        // 标签页切换
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchLanguage(e.target.dataset.lang);
            });
        });

        // 主题切换
        document.getElementById('themeSelector').addEventListener('change', (e) => {
            this.changeTheme(e.target.value);
        });

        // 窗口大小变化
        window.addEventListener('resize', () => {
            this.editor.refresh();
        });
    }

    // 切换语言模式
    switchLanguage(language) {
        this.currentLanguage = language;
        
        // 更新标签页状态
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.lang === language);
        });

        // 更新编辑器模式
        let mode;
        switch (language) {
            case 'smartnet':
                mode = 'python'; // 使用Python语法高亮作为基础
                break;
            case 'pytorch':
                mode = 'python';
                break;
            case 'yaml':
                mode = 'yaml';
                break;
        }
        
        this.editor.setOption('mode', mode);
        this.loadSyntaxHelp();
        
        // 加载对应的示例代码
        if (language === 'pytorch' && this.currentNetwork) {
            this.convertToPyTorch();
        } else if (language === 'yaml' && this.currentNetwork) {
            this.convertToYAML();
        }
    }

    // 更改主题
    changeTheme(theme) {
        this.editor.setOption('theme', theme);
    }

    // 代码变化处理
    onCodeChange() {
        const code = this.editor.getValue();
        this.updateCodeStats(code);
        this.validateSyntax(code);
        this.updatePreview(code);
        this.markUnsaved();
    }

    // 更新代码统计
    updateCodeStats(code) {
        const lines = code.split('\n').length;
        const chars = code.length;
        
        document.getElementById('lineCount').textContent = lines;
        document.getElementById('charCount').textContent = chars;
    }

    // 语法验证
    validateSyntax(code) {
        const statusElement = document.getElementById('syntaxStatus');
        
        try {
            if (this.currentLanguage === 'smartnet') {
                const network = window.smartNetDSL.parse(code);
                
                if (network.errors.length > 0) {
                    this.showSyntaxErrors(network.errors);
                    statusElement.innerHTML = '<i class="fas fa-exclamation-triangle text-danger"></i> 语法错误';
                    statusElement.className = 'status-indicator error';
                } else {
                    statusElement.innerHTML = '<i class="fas fa-check-circle text-success"></i> 语法正确';
                    statusElement.className = 'status-indicator success';
                    this.currentNetwork = network;
                    
                    // 启用构建按钮
                    document.querySelector('[onclick="parseAndBuild()"]').disabled = false;
                }
            } else {
                // 对于PyTorch代码进行基本语法检查
                statusElement.innerHTML = '<i class="fas fa-code text-info"></i> ' + this.currentLanguage.toUpperCase();
                statusElement.className = 'status-indicator';
            }
        } catch (error) {
            statusElement.innerHTML = '<i class="fas fa-times-circle text-danger"></i> 解析错误';
            statusElement.className = 'status-indicator error';
        }
    }

    // 显示语法错误
    showSyntaxErrors(errors) {
        const errorDetails = errors.map(error => 
            `<div class="error-item">
                <strong>行 ${error.line}:</strong> ${error.message}
                <small class="error-type">(${error.type})</small>
            </div>`
        ).join('');
        
        document.getElementById('errorDetails').innerHTML = errorDetails;
    }

    // 更新预览
    updatePreview(code) {
        const previewElement = document.getElementById('networkPreview');
        
        if (this.currentLanguage === 'smartnet' && this.currentNetwork && this.currentNetwork.errors.length === 0) {
            this.renderNetworkPreview(this.currentNetwork);
            this.updateNetworkStats(this.currentNetwork);
        } else {
            previewElement.innerHTML = `
                <div class="preview-placeholder">
                    <i class="fas fa-code-branch"></i>
                    <p>${this.getPlaceholderText()}</p>
                </div>
            `;
            this.resetNetworkStats();
        }
    }

    // 渲染网络预览
    renderNetworkPreview(network) {
        const previewElement = document.getElementById('networkPreview');
        previewElement.innerHTML = ''; // 清空现有内容
        previewElement.style.position = 'relative';

        const visualComponents = window.smartNetDSL.toVisualComponents(network);
        
        // 渲染组件节点
        visualComponents.components.forEach((component, index) => {
            const nodeElement = document.createElement('div');
            nodeElement.className = 'preview-node';
            nodeElement.innerHTML = `
                <div class="node-icon">${component.icon}</div>
                <div class="node-name">${component.name}</div>
                <div class="node-type">${component.type}</div>
            `;
            
            // 设置位置
            nodeElement.style.left = (index * 120 + 20) + 'px';
            nodeElement.style.top = '50px';
            nodeElement.style.width = '100px';
            nodeElement.style.height = '80px';
            
            previewElement.appendChild(nodeElement);
        });

        // 渲染连接线
        this.renderConnections(network.connections, visualComponents.components);
    }

    // 渲染连接线
    renderConnections(connections, components) {
        const previewElement = document.getElementById('networkPreview');
        
        connections.forEach(conn => {
            const fromComponent = components.find(c => c.name === conn.from);
            const toComponent = components.find(c => c.name === conn.to);
            
            if (fromComponent && toComponent) {
                const line = document.createElement('div');
                line.className = 'preview-connection';
                
                const fromIndex = components.indexOf(fromComponent);
                const toIndex = components.indexOf(toComponent);
                
                const startX = fromIndex * 120 + 120;
                const endX = toIndex * 120 + 20;
                const y = 90; // 节点中心高度
                
                line.style.left = startX + 'px';
                line.style.top = y + 'px';
                line.style.width = (endX - startX) + 'px';
                
                previewElement.appendChild(line);
            }
        });
    }

    // 更新网络统计
    updateNetworkStats(network) {
        const stats = window.smartNetDSL.estimateParameters(network);
        
        document.getElementById('paramCount').textContent = 
            stats.total_parameters.toLocaleString();
        document.getElementById('layerCount').textContent = stats.total_layers;
        document.getElementById('memoryEst').textContent = stats.memory_estimate;
    }

    // 重置网络统计
    resetNetworkStats() {
        document.getElementById('paramCount').textContent = '-';
        document.getElementById('layerCount').textContent = '-';
        document.getElementById('memoryEst').textContent = '-';
    }

    // 获取占位符文本
    getPlaceholderText() {
        switch (this.currentLanguage) {
            case 'smartnet':
                return '在左侧编辑SmartNet DSL代码，这里将显示网络架构预览';
            case 'pytorch':
                return 'PyTorch代码编辑模式，切换到SmartNet DSL查看预览';
            case 'yaml':
                return 'YAML配置编辑模式，切换到SmartNet DSL查看预览';
            default:
                return '代码预览区域';
        }
    }

    // 加载语法帮助
    loadSyntaxHelp() {
        const helpElement = document.getElementById('syntaxHelp');
        const help = window.smartNetDSL.getSyntaxHelp(this.currentLanguage);
        
        if (help) {
            helpElement.innerHTML = `
                <h4>${help.title}</h4>
                ${help.sections.map(section => `
                    <div class="syntax-section">
                        <h5>${section.title}</h5>
                        <pre><code>${section.content}</code></pre>
                    </div>
                `).join('')}
            `;
        }
    }

    // 加载示例
    loadExample(exampleType = null) {
        if (!exampleType) {
            // 显示示例选择对话框
            this.showExampleDialog();
            return;
        }
        
        const examples = window.smartNetDSL.examples;
        if (examples[exampleType]) {
            this.editor.setValue(examples[exampleType]);
            this.onCodeChange();
        }
    }

    // 格式化代码
    formatCode() {
        const code = this.editor.getValue();
        // 简单的代码格式化
        const formatted = code.split('\n').map(line => {
            let indent = 0;
            const trimmed = line.trim();
            
            if (trimmed.startsWith('layer') || trimmed.startsWith('connection') || 
                trimmed.startsWith('input_dim') || trimmed.startsWith('output_dim')) {
                indent = 1;
            } else if (trimmed.startsWith('branch')) {
                indent = 1;
            }
            
            return '    '.repeat(indent) + trimmed;
        }).join('\n');
        
        this.editor.setValue(formatted);
    }

    // 导出代码
    exportCode() {
        const code = this.editor.getValue();
        const filename = `smartnet_${this.currentLanguage}_${Date.now()}.${this.getFileExtension()}`;
        
        const blob = new Blob([code], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // 获取文件扩展名
    getFileExtension() {
        switch (this.currentLanguage) {
            case 'smartnet': return 'snet';
            case 'pytorch': return 'py';
            case 'yaml': return 'yaml';
            default: return 'txt';
        }
    }

    // 转换为PyTorch代码
    convertToPyTorch() {
        if (!this.currentNetwork) return;
        
        const network = this.currentNetwork;
        let pyTorchCode = `import torch
import torch.nn as nn
import torch.nn.functional as F

class ${network.name || 'GeneratedNetwork'}(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Network parameters
        self.input_dim = ${network.parameters.input_dim || 128}
        self.output_dim = ${network.parameters.output_dim || 10}
        
        # Define layers
`;
        
        // 添加层定义
        network.layers.forEach(layer => {
            pyTorchCode += `        self.${layer.name} = ${this.convertLayerToPyTorch(layer)}\n`;
        });
        
        pyTorchCode += `
    def forward(self, x):
        # Forward pass
`;
        
        // 添加前向传播逻辑
        if (network.connections.length > 0) {
            let currentVar = 'x';
            network.connections.forEach(conn => {
                if (conn.from !== 'input') {
                    pyTorchCode += `        ${currentVar} = self.${conn.to}(${currentVar})\n`;
                    if (network.parameters.activation) {
                        pyTorchCode += `        ${currentVar} = F.${network.parameters.activation}(${currentVar})\n`;
                    }
                }
            });
        }
        
        pyTorchCode += `        return ${currentVar || 'x'}`;
        
        this.editor.setValue(pyTorchCode);
    }

    // 转换层到PyTorch格式
    convertLayerToPyTorch(layer) {
        switch (layer.type) {
            case 'Linear':
                const inDim = layer.parameters.input_dim || layer.parameters.arg_0 || 128;
                const outDim = layer.parameters.output_dim || layer.parameters.arg_1 || 64;
                return `nn.Linear(${inDim}, ${outDim})`;
                
            case 'Conv2d':
                const inCh = layer.parameters.in_channels || layer.parameters.arg_0 || 3;
                const outCh = layer.parameters.out_channels || layer.parameters.arg_1 || 32;
                const kernel = layer.parameters.kernel_size || layer.parameters.arg_2 || 3;
                return `nn.Conv2d(${inCh}, ${outCh}, ${kernel})`;
                
            case 'LSTM':
                const inputSize = layer.parameters.input_size || layer.parameters.arg_0 || 128;
                const hiddenSize = layer.parameters.hidden_size || layer.parameters.arg_1 || 256;
                return `nn.LSTM(${inputSize}, ${hiddenSize})`;
                
            default:
                return `nn.${layer.type}()`;
        }
    }

    // 转换为YAML配置
    convertToYAML() {
        if (!this.currentNetwork) return;
        
        const network = this.currentNetwork;
        let yamlCode = `# SmartNet网络配置
name: ${network.name || 'GeneratedNetwork'}

parameters:
  input_dim: ${network.parameters.input_dim || 128}
  output_dim: ${network.parameters.output_dim || 10}
  learning_rate: ${network.parameters.learning_rate || 0.001}

layers:
`;
        
        network.layers.forEach(layer => {
            yamlCode += `  - name: ${layer.name}\n`;
            yamlCode += `    type: ${layer.type}\n`;
            yamlCode += `    parameters:\n`;
            
            Object.entries(layer.parameters).forEach(([key, value]) => {
                yamlCode += `      ${key}: ${value}\n`;
            });
        });
        
        yamlCode += `
connections:
`;
        
        network.connections.forEach(conn => {
            yamlCode += `  - from: ${conn.from}\n`;
            yamlCode += `    to: ${conn.to}\n`;
            yamlCode += `    type: ${conn.type}\n`;
        });
        
        this.editor.setValue(yamlCode);
    }

    // 自动保存
    startAutoSave() {
        this.autoSaveInterval = setInterval(() => {
            this.autoSave();
        }, 30000); // 每30秒自动保存
    }

    autoSave() {
        const code = this.editor.getValue();
        localStorage.setItem('smartnet_code_backup', code);
        localStorage.setItem('smartnet_language', this.currentLanguage);
        this.lastSaveTime = Date.now();
        
        document.getElementById('saveStatus').textContent = '自动保存';
        setTimeout(() => {
            document.getElementById('saveStatus').textContent = '已保存';
        }, 2000);
    }

    // 标记未保存
    markUnsaved() {
        document.getElementById('saveStatus').textContent = '未保存';
    }

    // 检查系统状态
    async checkSystemStatus() {
        try {
            const response = await fetch('/api/system_info');
            const data = await response.json();
            
            if (data.success) {
                const gpuInfo = data.gpu_info;
                document.getElementById('gpuStatus').innerHTML = 
                    `GPU: ${gpuInfo.device_count}x ${gpuInfo.devices[0]?.name || 'Unknown'}`;
            }
        } catch (error) {
            document.getElementById('gpuStatus').innerHTML = 'GPU: 检测失败';
        }
    }

    // 刷新预览
    refreshPreview() {
        this.onCodeChange();
    }
}

// 全局函数
function switchToVisualMode() {
    window.location.href = '/';
}

function parseAndBuild() {
    if (window.codeEditor.currentNetwork) {
        const visualComponents = window.smartNetDSL.toVisualComponents(window.codeEditor.currentNetwork);
        
        // 发送到后端构建
        fetch('/api/build', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                network_name: window.codeEditor.currentNetwork.name || 'CodeGeneratedNetwork',
                components: visualComponents.components,
                connections: visualComponents.connections,
                parameters: visualComponents.parameters
            })
        }).then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('网络构建成功！\n' + 
                      '参数量: ' + data.total_parameters.toLocaleString() + '\n' +
                      '内存估计: ' + data.memory_estimate);
                
                // 启用训练按钮
                document.getElementById('trainBtn').disabled = false;
                document.getElementById('networkStatus').textContent = '网络: 已构建';
            } else {
                alert('构建失败: ' + data.message);
            }
        }).catch(error => {
            alert('构建错误: ' + error.message);
        });
    }
}

function trainFromCode() {
    const params = {
        epochs: 10,
        batch_size: 32,
        learning_rate: window.codeEditor.currentNetwork?.parameters?.learning_rate || 0.001
    };
    
    // 显示训练进度对话框
    document.getElementById('trainingModal').classList.add('show');
    
    fetch('/api/train', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(params)
    }).then(response => response.json())
    .then(data => {
        if (data.success) {
            // 开始监控训练进度
            monitorTrainingProgress();
        } else {
            alert('训练启动失败: ' + data.message);
            document.getElementById('trainingModal').classList.remove('show');
        }
    });
}

function monitorTrainingProgress() {
    // 模拟训练进度更新
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 5;
        if (progress >= 100) {
            progress = 100;
            clearInterval(interval);
            
            setTimeout(() => {
                document.getElementById('trainingModal').classList.remove('show');
                alert('训练完成！');
            }, 1000);
        }
        
        document.getElementById('trainingProgress').style.width = progress + '%';
        document.getElementById('trainingPercent').textContent = Math.round(progress) + '%';
        document.getElementById('trainingStatus').textContent = 
            progress < 100 ? '训练中...' : '训练完成';
    }, 500);
}

function stopTraining() {
    // TODO: 实现停止训练的逻辑
    document.getElementById('trainingModal').classList.remove('show');
}

function exportToVisual() {
    if (window.codeEditor.currentNetwork) {
        // 将当前网络保存到localStorage，供可视化界面使用
        localStorage.setItem('imported_network', JSON.stringify(window.codeEditor.currentNetwork));
        window.location.href = '/?import=true';
    }
}

function closeErrorModal() {
    document.getElementById('errorModal').classList.remove('show');
}

function fixError() {
    // TODO: 实现自动错误修复
    alert('自动修复功能正在开发中...');
    closeErrorModal();
}

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    window.codeEditor = new CodeEditor();
    
    // 恢复上次的代码
    const savedCode = localStorage.getItem('smartnet_code_backup');
    const savedLanguage = localStorage.getItem('smartnet_language');
    
    if (savedCode) {
        window.codeEditor.editor.setValue(savedCode);
    }
    
    if (savedLanguage) {
        window.codeEditor.switchLanguage(savedLanguage);
    }
});
