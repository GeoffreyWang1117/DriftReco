// SmartNet DSL 解析器和语法支持

class SmartNetDSL {
    constructor() {
        this.keywords = [
            'network', 'layer', 'connection', 'input_dim', 'output_dim', 
            'hidden_dim', 'dropout', 'activation', 'learning_rate',
            'MLP', 'CNN', 'RNN', 'Transformer', 'LSTM', 'GRU',
            'Linear', 'Conv2d', 'MaxPool2d', 'BatchNorm', 'LayerNorm',
            'Attention', 'MultiHeadAttention', 'FeedForward'
        ];
        
        this.activations = [
            'relu', 'sigmoid', 'tanh', 'leaky_relu', 'gelu', 'swish'
        ];
        
        this.examples = {
            simple_mlp: `# 简单多层感知机
network SimpleMLP:
    input_dim: 128
    output_dim: 10
    learning_rate: 0.001
    
    layer input: Linear(128, 64)
    layer hidden1: Linear(64, 32)
    layer output: Linear(32, 10)
    
    connection: input -> hidden1 -> output
    activation: relu
    dropout: 0.1`,

            transformer: `# Transformer网络
network TransformerNet:
    input_dim: 512
    output_dim: 10
    learning_rate: 0.0001
    
    layer embedding: Linear(input_dim, 512)
    layer transformer: MultiHeadAttention(
        embed_dim: 512,
        num_heads: 8,
        dropout: 0.1
    )
    layer ffn: FeedForward(512, 2048, 512)
    layer classifier: Linear(512, output_dim)
    
    connection: embedding -> transformer -> ffn -> classifier
    norm: LayerNorm`,

            cnn_rnn: `# CNN + RNN 混合网络
network CNNRNNNet:
    input_dim: [3, 224, 224]  # 图像输入
    sequence_length: 10
    output_dim: 5
    learning_rate: 0.001
    
    # CNN特征提取
    layer conv1: Conv2d(3, 32, kernel_size=3)
    layer conv2: Conv2d(32, 64, kernel_size=3)
    layer pool: MaxPool2d(kernel_size=2)
    layer flatten: Flatten()
    
    # RNN序列处理
    layer rnn: LSTM(
        input_size: 64*55*55,
        hidden_size: 256,
        num_layers: 2,
        dropout: 0.2
    )
    
    # 分类器
    layer classifier: Linear(256, output_dim)
    
    connection: conv1 -> conv2 -> pool -> flatten -> rnn -> classifier
    activation: relu`,

            complex: `# 复杂多分支网络
network ComplexNet:
    input_dim: 1024
    output_dim: 20
    learning_rate: 0.0005
    
    # 主干网络
    layer backbone: Linear(input_dim, 512)
    
    # 分支1: 注意力机制
    branch attention_branch:
        layer attention: MultiHeadAttention(512, num_heads=8)
        layer attn_norm: LayerNorm(512)
        layer attn_ffn: FeedForward(512, 1024, 512)
    
    # 分支2: 卷积特征
    branch conv_branch:
        layer conv_proj: Linear(512, 256*8*8)  # 重塑为特征图
        layer conv1: Conv2d(256, 128, 3, padding=1)
        layer conv2: Conv2d(128, 64, 3, padding=1)
        layer global_pool: AdaptiveAvgPool2d(1)
        layer conv_out: Linear(64, 256)
    
    # 特征融合
    layer fusion: Linear(768, 256)  # 512 + 256
    layer classifier: Linear(256, output_dim)
    
    # 连接定义
    connection main: backbone -> [attention_branch, conv_branch]
    connection merge: [attention_branch, conv_branch] -> fusion -> classifier
    
    # 全局配置
    activation: gelu
    dropout: 0.15
    batch_norm: true`
        };
        
        this.syntaxRules = [
            {
                name: 'network_definition',
                pattern: /^network\s+(\w+):/,
                description: '定义网络名称',
                example: 'network MyNetwork:'
            },
            {
                name: 'parameter_setting',
                pattern: /^\s*(input_dim|output_dim|learning_rate|dropout):\s*(.+)/,
                description: '设置网络参数',
                example: 'input_dim: 128'
            },
            {
                name: 'layer_definition',
                pattern: /^\s*layer\s+(\w+):\s*(.+)/,
                description: '定义网络层',
                example: 'layer hidden: Linear(128, 64)'
            },
            {
                name: 'connection_definition',
                pattern: /^\s*connection:\s*(.+)/,
                description: '定义连接关系',
                example: 'connection: input -> hidden -> output'
            },
            {
                name: 'branch_definition',
                pattern: /^\s*branch\s+(\w+):/,
                description: '定义网络分支',
                example: 'branch encoder_branch:'
            }
        ];
        
        this.componentMapping = {
            'Linear': { type: 'MLP', icon: '🟢' },
            'Conv2d': { type: 'CNN', icon: '🟠' },
            'LSTM': { type: 'RNN', icon: '🟣' },
            'GRU': { type: 'RNN', icon: '🟣' },
            'MultiHeadAttention': { type: 'Transformer', icon: '🔵' },
            'FeedForward': { type: 'MLP', icon: '🟢' },
            'BatchNorm': { type: 'MLP', icon: '🟢' },
            'LayerNorm': { type: 'Transformer', icon: '🔵' },
            'MaxPool2d': { type: 'CNN', icon: '🟠' },
            'Dropout': { type: 'MLP', icon: '🟢' }
        };
    }
    
    // 解析DSL代码 (支持SmartNet DSL和YAML)
    parse(code) {
        // 检测代码格式
        const trimmedCode = code.trim();
        
        if (this.isYAMLFormat(trimmedCode)) {
            return this.parseYAML(trimmedCode);
        } else {
            return this.parseSmartNetDSL(trimmedCode);
        }
    }
    
    // 检测是否为YAML格式
    isYAMLFormat(code) {
        // YAML特征：以name:开头，包含layers:，connections:等
        return code.includes('name:') && 
               code.includes('layers:') && 
               code.includes('- name:');
    }
    
    // 解析YAML格式
    parseYAML(code) {
        const network = {
            name: '',
            parameters: {},
            layers: [],
            connections: [],
            branches: {},
            errors: []
        };
        
        try {
            // 简单YAML解析 (基于正则表达式)
            const lines = code.split('\n');
            let currentSection = null;
            let currentLayer = null;
            let currentConnection = null;
            
            for (let i = 0; i < lines.length; i++) {
                const line = lines[i];
                const trimmed = line.trim();
                
                if (!trimmed || trimmed.startsWith('#')) continue;
                
                // 顶级字段
                if (trimmed.startsWith('name:')) {
                    network.name = trimmed.split(':', 2)[1].trim();
                } else if (trimmed === 'parameters:') {
                    currentSection = 'parameters';
                } else if (trimmed === 'layers:') {
                    currentSection = 'layers';
                } else if (trimmed === 'connections:') {
                    currentSection = 'connections';
                } else if (currentSection === 'parameters') {
                    // 解析参数
                    const paramMatch = trimmed.match(/^(\w+):\s*(.+)$/);
                    if (paramMatch) {
                        const [, key, value] = paramMatch;
                        network.parameters[key] = this.parseValue(value);
                    }
                } else if (currentSection === 'layers') {
                    // 解析层
                    if (trimmed.startsWith('- name:')) {
                        currentLayer = {
                            name: trimmed.split(':', 2)[1].trim(),
                            type: '',
                            parameters: {},
                            raw: ''
                        };
                        network.layers.push(currentLayer);
                    } else if (currentLayer && trimmed.startsWith('type:')) {
                        currentLayer.type = trimmed.split(':', 2)[1].trim();
                    } else if (currentLayer && trimmed === 'parameters:') {
                        // 下面的行是参数
                    } else if (currentLayer && line.match(/^\s{6,}\w+:/)) {
                        // 层参数 (6个或更多空格缩进)
                        const paramMatch = trimmed.match(/^(\w+):\s*(.+)$/);
                        if (paramMatch) {
                            const [, key, value] = paramMatch;
                            currentLayer.parameters[key] = this.parseValue(value);
                        }
                    }
                } else if (currentSection === 'connections') {
                    // 解析连接
                    if (trimmed.startsWith('- from:')) {
                        currentConnection = {
                            from: trimmed.split(':', 2)[1].trim(),
                            to: '',
                            type: 'sequential'
                        };
                        network.connections.push(currentConnection);
                    } else if (currentConnection && trimmed.startsWith('to:')) {
                        currentConnection.to = trimmed.split(':', 2)[1].trim();
                    } else if (currentConnection && trimmed.startsWith('type:')) {
                        currentConnection.type = trimmed.split(':', 2)[1].trim();
                    }
                }
            }
            
        } catch (error) {
            network.errors.push({
                line: 0,
                message: `YAML解析错误: ${error.message}`,
                type: 'yaml_parse_error'
            });
        }
        
        // 验证网络结构
        this.validateNetwork(network);
        
        return network;
    }
    
    // 解析SmartNet DSL代码
    parseSmartNetDSL(code) {
        const lines = code.split('\n');
        const network = {
            name: '',
            parameters: {},
            layers: [],
            connections: [],
            branches: {},
            errors: []
        };
        
        let currentBranch = null;
        let lineNumber = 0;
        
        for (const line of lines) {
            lineNumber++;
            const trimmed = line.trim();
            
            // 跳过空行和注释
            if (!trimmed || trimmed.startsWith('#')) continue;
            
            try {
                // 网络定义
                const networkMatch = trimmed.match(/^network\s+(\w+):/);
                if (networkMatch) {
                    network.name = networkMatch[1];
                    continue;
                }
                
                // 参数设置
                const paramMatch = trimmed.match(/^\s*(input_dim|output_dim|learning_rate|dropout|activation|batch_norm):\s*(.+)/);
                if (paramMatch) {
                    const [, key, value] = paramMatch;
                    network.parameters[key] = this.parseValue(value);
                    continue;
                }
                
                // 分支定义
                const branchMatch = trimmed.match(/^\s*branch\s+(\w+):/);
                if (branchMatch) {
                    currentBranch = branchMatch[1];
                    network.branches[currentBranch] = { layers: [], connections: [] };
                    continue;
                }
                
                // 层定义
                const layerMatch = trimmed.match(/^\s*layer\s+(\w+):\s*(.+)/);
                if (layerMatch) {
                    const [, name, definition] = layerMatch;
                    const layer = this.parseLayer(name, definition);
                    
                    if (currentBranch) {
                        network.branches[currentBranch].layers.push(layer);
                    } else {
                        network.layers.push(layer);
                    }
                    continue;
                }
                
                // 连接定义
                const connectionMatch = trimmed.match(/^\s*connection(?:\s+(\w+))?:\s*(.+)/);
                if (connectionMatch) {
                    const [, name, definition] = connectionMatch;
                    const connections = this.parseConnections(definition);
                    
                    if (currentBranch) {
                        network.branches[currentBranch].connections.push(...connections);
                    } else {
                        network.connections.push(...connections);
                    }
                    continue;
                }
                
                // 未识别的语法
                if (trimmed) {
                    network.errors.push({
                        line: lineNumber,
                        message: `未识别的语法: ${trimmed}`,
                        type: 'syntax_error'
                    });
                }
                
            } catch (error) {
                network.errors.push({
                    line: lineNumber,
                    message: error.message,
                    type: 'parse_error'
                });
            }
        }
        
        // 验证网络结构
        this.validateNetwork(network);
        
        return network;
    }
    
    // 解析层定义
    parseLayer(name, definition) {
        const layer = {
            name: name,
            type: '',
            parameters: {},
            raw: definition
        };
        
        // 解析层类型和参数
        const match = definition.match(/^(\w+)\s*(?:\((.*)\))?/);
        if (match) {
            const [, layerType, paramsStr] = match;
            layer.type = layerType;
            
            if (paramsStr) {
                layer.parameters = this.parseParameters(paramsStr);
            }
        }
        
        return layer;
    }
    
    // 解析参数
    parseParameters(paramsStr) {
        const params = {};
        
        if (!paramsStr || paramsStr.trim() === '') {
            return params;
        }
        
        // 处理多行参数（括号内换行）
        const cleanParamsStr = paramsStr.replace(/\s+/g, ' ').trim();
        
        // 分割参数，支持逗号分隔
        const paramPairs = this.splitParameters(cleanParamsStr);
        
        for (let i = 0; i < paramPairs.length; i++) {
            const pair = paramPairs[i].trim();
            
            // 带名称的参数 (key: value)
            const colonMatch = pair.match(/^(\w+):\s*(.+)$/);
            if (colonMatch) {
                const [, key, value] = colonMatch;
                params[key] = this.parseValue(value);
            } else if (pair) {
                // 位置参数
                params[`arg_${i}`] = this.parseValue(pair);
            }
        }
        
        return params;
    }
    
    // 智能分割参数，处理嵌套括号
    splitParameters(paramsStr) {
        const params = [];
        let current = '';
        let depth = 0;
        let inQuotes = false;
        
        for (let i = 0; i < paramsStr.length; i++) {
            const char = paramsStr[i];
            
            if (char === '"' || char === "'") {
                inQuotes = !inQuotes;
                current += char;
            } else if (!inQuotes) {
                if (char === '(' || char === '[' || char === '{') {
                    depth++;
                    current += char;
                } else if (char === ')' || char === ']' || char === '}') {
                    depth--;
                    current += char;
                } else if (char === ',' && depth === 0) {
                    params.push(current.trim());
                    current = '';
                } else {
                    current += char;
                }
            } else {
                current += char;
            }
        }
        
        if (current.trim()) {
            params.push(current.trim());
        }
        
        return params;
    }
    
    // 解析值
    parseValue(value) {
        if (!value) return null;
        
        const trimmed = value.trim();
        
        // 空值
        if (trimmed === '') return null;
        
        // 数字 (包括负数和小数)
        if (/^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(trimmed)) {
            return parseFloat(trimmed);
        }
        
        // 布尔值
        if (trimmed === 'true') return true;
        if (trimmed === 'false') return false;
        
        // null和undefined
        if (trimmed === 'null') return null;
        if (trimmed === 'undefined') return undefined;
        
        // 数组
        if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
            try {
                const arrayStr = trimmed.slice(1, -1);
                if (arrayStr.trim() === '') return [];
                
                // 使用智能分割处理嵌套数组
                const items = this.splitParameters(arrayStr);
                return items.map(item => this.parseValue(item.trim()));
            } catch (error) {
                console.warn('数组解析失败:', trimmed, error);
                return trimmed; // 解析失败时返回原字符串
            }
        }
        
        // 对象 (简单支持)
        if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
            try {
                return JSON.parse(trimmed);
            } catch (error) {
                console.warn('对象解析失败:', trimmed, error);
                return trimmed;
            }
        }
        
        // 字符串 (移除引号)
        if ((trimmed.startsWith('"') && trimmed.endsWith('"')) ||
            (trimmed.startsWith("'") && trimmed.endsWith("'"))) {
            return trimmed.slice(1, -1);
        }
        
        // 特殊值处理
        if (trimmed === 'input_dim' || trimmed === 'output_dim') {
            return trimmed; // 保持变量引用
        }
        
        // 默认返回字符串
        return trimmed;
    }
    
    // 解析连接
    parseConnections(definition) {
        const connections = [];
        
        // 简单连接: a -> b -> c
        if (definition.includes('->')) {
            const parts = definition.split('->').map(p => p.trim());
            for (let i = 0; i < parts.length - 1; i++) {
                connections.push({
                    from: parts[i],
                    to: parts[i + 1],
                    type: 'sequential'
                });
            }
        }
        
        // 分支连接: a -> [b, c]
        const branchMatch = definition.match(/(\w+)\s*->\s*\[([^\]]+)\]/);
        if (branchMatch) {
            const [, from, toList] = branchMatch;
            const targets = toList.split(',').map(t => t.trim());
            targets.forEach(target => {
                connections.push({
                    from: from,
                    to: target,
                    type: 'branch'
                });
            });
        }
        
        // 合并连接: [a, b] -> c
        const mergeMatch = definition.match(/\[([^\]]+)\]\s*->\s*(\w+)/);
        if (mergeMatch) {
            const [, fromList, to] = mergeMatch;
            const sources = fromList.split(',').map(s => s.trim());
            sources.forEach(source => {
                connections.push({
                    from: source,
                    to: to,
                    type: 'merge'
                });
            });
        }
        
        return connections;
    }
    
    // 验证网络结构
    validateNetwork(network) {
        // 检查必要参数
        if (!network.parameters.input_dim) {
            network.errors.push({
                line: 0,
                message: '缺少 input_dim 参数',
                type: 'validation_error'
            });
        }
        
        if (!network.parameters.output_dim) {
            network.errors.push({
                line: 0,
                message: '缺少 output_dim 参数',
                type: 'validation_error'
            });
        }
        
        // 检查层连接的有效性
        const layerNames = new Set(network.layers.map(l => l.name));
        
        for (const conn of network.connections) {
            if (!layerNames.has(conn.from) && conn.from !== 'input') {
                network.errors.push({
                    line: 0,
                    message: `未定义的层: ${conn.from}`,
                    type: 'validation_error'
                });
            }
            
            if (!layerNames.has(conn.to) && conn.to !== 'output') {
                network.errors.push({
                    line: 0,
                    message: `未定义的层: ${conn.to}`,
                    type: 'validation_error'
                });
            }
        }
    }
    
    // 转换为可视化组件
    toVisualComponents(network) {
        const components = [];
        let position = { x: 100, y: 100 };
        
        // 添加网络层
        for (const layer of network.layers) {
            const mapping = this.componentMapping[layer.type];
            if (mapping) {
                components.push({
                    id: layer.name,
                    type: mapping.type,
                    icon: mapping.icon,
                    name: layer.name,
                    parameters: this.convertParameters(layer),
                    position: { ...position }
                });
                
                position.x += 150;
                if (position.x > 600) {
                    position.x = 100;
                    position.y += 120;
                }
            }
        }
        
        return {
            components: components,
            connections: network.connections,
            parameters: network.parameters
        };
    }
    
    // 转换参数格式
    convertParameters(layer) {
        const params = { ...layer.parameters };
        
        // 根据层类型转换参数名称
        switch (layer.type) {
            case 'Linear':
                if (params.arg_0) params.input_dim = params.arg_0;
                if (params.arg_1) params.output_dim = params.arg_1;
                break;
            case 'Conv2d':
                if (params.arg_0) params.in_channels = params.arg_0;
                if (params.arg_1) params.out_channels = params.arg_1;
                if (params.arg_2) params.kernel_size = params.arg_2;
                break;
            case 'LSTM':
            case 'GRU':
                if (params.arg_0) params.input_size = params.arg_0;
                if (params.arg_1) params.hidden_size = params.arg_1;
                break;
        }
        
        return params;
    }
    
    // 估算网络参数
    estimateParameters(network) {
        let totalParams = 0;
        let totalLayers = network.layers.length;
        
        for (const layer of network.layers) {
            switch (layer.type) {
                case 'Linear':
                    const input_dim = layer.parameters.input_dim || layer.parameters.arg_0 || 128;
                    const output_dim = layer.parameters.output_dim || layer.parameters.arg_1 || 64;
                    totalParams += input_dim * output_dim + output_dim; // weights + bias
                    break;
                case 'Conv2d':
                    const in_ch = layer.parameters.in_channels || layer.parameters.arg_0 || 3;
                    const out_ch = layer.parameters.out_channels || layer.parameters.arg_1 || 32;
                    const kernel = layer.parameters.kernel_size || layer.parameters.arg_2 || 3;
                    totalParams += in_ch * out_ch * kernel * kernel + out_ch;
                    break;
                case 'MultiHeadAttention':
                    const embed_dim = layer.parameters.embed_dim || 512;
                    totalParams += embed_dim * embed_dim * 4; // Q, K, V, O projections
                    break;
                case 'LSTM':
                    const input_size = layer.parameters.input_size || layer.parameters.arg_0 || 128;
                    const hidden_size = layer.parameters.hidden_size || layer.parameters.arg_1 || 256;
                    const num_layers = layer.parameters.num_layers || 1;
                    totalParams += num_layers * (4 * (input_size + hidden_size + 1) * hidden_size);
                    break;
            }
        }
        
        return {
            total_parameters: totalParams,
            total_layers: totalLayers,
            memory_estimate: this.estimateMemory(totalParams)
        };
    }
    
    // 估算内存使用
    estimateMemory(params) {
        // 简单估算：参数 * 4 bytes (float32) * 3 (parameters + gradients + optimizer states)
        const bytes = params * 4 * 3;
        
        if (bytes < 1024 * 1024) {
            return (bytes / 1024).toFixed(1) + ' KB';
        } else if (bytes < 1024 * 1024 * 1024) {
            return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        } else {
            return (bytes / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
        }
    }
    
    // 获取语法帮助
    getSyntaxHelp(language = 'smartnet') {
        if (language === 'smartnet') {
            return {
                title: 'SmartNet DSL 语法',
                sections: [
                    {
                        title: '网络定义',
                        content: 'network NetworkName:\n    # 网络配置'
                    },
                    {
                        title: '参数设置',
                        content: 'input_dim: 128\noutput_dim: 10\nlearning_rate: 0.001'
                    },
                    {
                        title: '层定义',
                        content: 'layer name: LayerType(param1, param2)\nlayer hidden: Linear(128, 64)'
                    },
                    {
                        title: '连接定义',
                        content: 'connection: layer1 -> layer2 -> layer3\nconnection: input -> [branch1, branch2]\nconnection: [branch1, branch2] -> output'
                    }
                ]
            };
        } else if (language === 'pytorch') {
            return {
                title: 'PyTorch 语法',
                sections: [
                    {
                        title: '导入模块',
                        content: 'import torch\nimport torch.nn as nn\nimport torch.nn.functional as F'
                    },
                    {
                        title: '定义网络',
                        content: 'class MyNetwork(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.linear1 = nn.Linear(128, 64)'
                    },
                    {
                        title: '前向传播',
                        content: 'def forward(self, x):\n    x = F.relu(self.linear1(x))\n    return x'
                    }
                ]
            };
        }
    }
    
    estimateParameters(network) {
        // 估算网络参数量和内存使用
        let totalParams = 0;
        let memoryMB = 0;
        
        for (const layer of network.layers) {
            const layerParams = this.estimateLayerParameters(layer);
            totalParams += layerParams;
        }
        
        // 估算内存使用 (参数 + 梯度 + 优化器状态)
        memoryMB = Math.ceil(totalParams * 4 * 3 / 1024 / 1024); // 假设float32
        
        return {
            total_parameters: totalParams,
            total_layers: network.layers.length,
            memory_estimate: `${memoryMB} MB`,
            training_feasible: totalParams <= 50_000_000
        };
    }
    
    estimateLayerParameters(layer) {
        // 估算单层参数量
        const type = layer.type;
        const params = layer.parameters;
        
        switch (type) {
            case 'Linear': {
                const inputSize = params.arg_0 || params.input_size || 128;
                const outputSize = params.arg_1 || params.output_size || 64;
                return inputSize * outputSize + outputSize; // 权重 + 偏置
            }
                
            case 'Conv2d': {
                const inChannels = params.arg_0 || params.in_channels || 3;
                const outChannels = params.arg_1 || params.out_channels || 32;
                const kernelSize = params.kernel_size || params.arg_2 || 3;
                return inChannels * outChannels * kernelSize * kernelSize + outChannels;
            }
                
            case 'MultiHeadAttention': {
                const embedDim = params.embed_dim || params.arg_0 || 512;
                const numHeads = params.num_heads || params.arg_1 || 8;
                // Q, K, V projections + output projection + bias
                return 4 * embedDim * embedDim + 4 * embedDim;
            }
                
            case 'LSTM':
            case 'GRU': {
                const inputSize = params.input_size || params.arg_0 || 128;
                const hiddenSize = params.hidden_size || params.arg_1 || 64;
                const numLayers = params.num_layers || 1;
                // 简化估算: 4个门 * (input_size * hidden_size + hidden_size * hidden_size)
                return numLayers * 4 * (inputSize * hiddenSize + hiddenSize * hiddenSize + hiddenSize);
            }
                
            case 'BatchNorm':
            case 'LayerNorm': {
                const numFeatures = params.num_features || params.arg_0 || 64;
                return numFeatures * 2; // weight + bias
            }
                
            default:
                return 1000; // 默认估算
        }
    }
}

// 导出全局实例
window.smartNetDSL = new SmartNetDSL();
