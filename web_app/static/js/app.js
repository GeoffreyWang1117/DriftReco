// SmartNet Web应用 JavaScript主文件

class SmartNetApp {
    constructor() {
        this.components = [];
        this.networkNodes = [];
        this.connections = [];
        this.selectedNode = null;
        this.draggedComponent = null;
        this.nodeCounter = 0;
        this.currentTrainingId = null;
        
        // DOM元素引用
        this.canvas = null;
        this.propertiesPanel = null;
        this.buildBtn = null;
        this.trainBtn = null;
        this.clearBtn = null;
        
        // 绑定方法上下文
        this.handleDrop = this.handleDrop.bind(this);
        this.handleDragOver = this.handleDragOver.bind(this);
        this.handleNodeClick = this.handleNodeClick.bind(this);
        this.handleNodeDrag = this.handleNodeDrag.bind(this);
    }
    
    // 初始化应用
    async init() {
        console.log('🚀 初始化SmartNet Web应用');
        
        this.initDOMElements();
        this.initEventListeners();
        await this.loadComponents();
        await this.loadSystemInfo();
        
        console.log('✅ SmartNet应用初始化完成');
    }
    
    // 初始化DOM元素引用
    initDOMElements() {
        this.canvas = document.getElementById('networkCanvas');
        this.propertiesPanel = document.getElementById('propertiesContent');
        this.buildBtn = document.getElementById('buildBtn');
        this.trainBtn = document.getElementById('trainBtn');
        this.clearBtn = document.getElementById('clearBtn');
        
        if (!this.canvas) {
            console.error('❌ 无法找到画布元素');
            return;
        }
        
        console.log('✅ DOM元素初始化完成');
    }
    
    // 初始化事件监听器
    initEventListeners() {
        // 画布拖拽事件
        this.canvas.addEventListener('drop', this.handleDrop);
        this.canvas.addEventListener('dragover', this.handleDragOver);
        
        // 按钮事件
        this.buildBtn?.addEventListener('click', () => this.buildNetwork());
        this.trainBtn?.addEventListener('click', () => this.showTrainingDialog());
        this.clearBtn?.addEventListener('click', () => this.clearCanvas());
        
        // 模态对话框事件
        this.initModalEvents();
        
        console.log('✅ 事件监听器初始化完成');
    }
    
    // 初始化模态对话框事件
    initModalEvents() {
        // 关闭按钮事件
        document.querySelectorAll('.close, .close-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const modalId = btn.getAttribute('data-modal') || 
                               btn.closest('.modal')?.id;
                if (modalId) {
                    this.hideModal(modalId);
                }
            });
        });
        
        // 训练相关按钮
        document.getElementById('proceedTrainBtn')?.addEventListener('click', 
            () => this.showTrainingDialog());
        document.getElementById('startTrainBtn')?.addEventListener('click', 
            () => this.startTraining());
        document.getElementById('stopTrainBtn')?.addEventListener('click', 
            () => this.stopTraining());
    }
    
    // 加载可用组件列表
    async loadComponents() {
        try {
            const response = await fetch('/api/components');
            const data = await response.json();
            
            if (data.success) {
                this.components = data.components;
                this.renderComponents();
                console.log(`✅ 已加载 ${this.components.length} 个组件`);
            } else {
                console.error('❌ 加载组件失败:', data.error);
                this.showNotification('加载组件失败', 'error');
            }
        } catch (error) {
            console.error('❌ 组件加载出错:', error);
            this.showNotification('组件加载出错', 'error');
        }
    }
    
    // 渲染组件列表
    renderComponents() {
        const basicContainer = document.getElementById('basicComponents');
        const advancedContainer = document.getElementById('advancedComponents');
        
        if (!basicContainer || !advancedContainer) return;
        
        // 清空容器
        basicContainer.innerHTML = '';
        advancedContainer.innerHTML = '';
        
        this.components.forEach(component => {
            const element = this.createComponentElement(component);
            
            if (component.category === 'basic') {
                basicContainer.appendChild(element);
            } else {
                advancedContainer.appendChild(element);
            }
        });
    }
    
    // 创建组件元素
    createComponentElement(component) {
        const element = document.createElement('div');
        element.className = 'component-item';
        element.draggable = true;
        element.setAttribute('data-component-id', component.id);
        
        element.innerHTML = `
            <div class="component-header">
                <div class="component-icon" style="background-color: ${component.color}">
                    <i class="fas fa-cube"></i>
                </div>
                <div class="component-name">${component.name}</div>
            </div>
            <div class="component-description">${component.description}</div>
        `;
        
        // 拖拽事件
        element.addEventListener('dragstart', (e) => {
            this.draggedComponent = component;
            element.classList.add('dragging');
        });
        
        element.addEventListener('dragend', () => {
            element.classList.remove('dragging');
            this.draggedComponent = null;
        });
        
        return element;
    }
    
    // 处理拖拽放置
    handleDrop(e) {
        e.preventDefault();
        this.canvas.classList.remove('drag-over');
        
        if (!this.draggedComponent) return;
        
        // 获取放置位置（相对于画布）
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left - 100; // 调整为节点中心
        const y = e.clientY - rect.top - 50;
        
        // 创建网络节点
        this.createNetworkNode(this.draggedComponent, x, y);
        
        // 隐藏提示
        document.getElementById('canvasHint').style.display = 'none';
        
        this.updateCanvasStats();
    }
    
    // 处理拖拽悬停
    handleDragOver(e) {
        e.preventDefault();
        this.canvas.classList.add('drag-over');
        
        // 延迟移除样式
        clearTimeout(this.dragOverTimeout);
        this.dragOverTimeout = setTimeout(() => {
            this.canvas.classList.remove('drag-over');
        }, 100);
    }
    
    // 创建网络节点
    createNetworkNode(component, x, y) {
        const nodeId = `node_${++this.nodeCounter}`;
        const node = {
            id: nodeId,
            component: { ...component },
            x: Math.max(0, x),
            y: Math.max(0, y),
            params: { ...component.params }
        };
        
        // 创建DOM元素
        const element = document.createElement('div');
        element.className = 'network-node';
        element.setAttribute('data-node-id', nodeId);
        element.style.left = `${node.x}px`;
        element.style.top = `${node.y}px`;
        
        element.innerHTML = `
            <div class="node-header">
                <div class="node-icon" style="background-color: ${component.color}">
                    <i class="fas fa-cube"></i>
                </div>
                <div class="node-title">${component.name}</div>
            </div>
            <div class="node-description">${component.description}</div>
            <div class="node-params" id="${nodeId}_params">
                ${this.formatNodeParams(node.params)}
            </div>
            <button class="node-delete" title="删除节点">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        // 添加事件监听
        element.addEventListener('click', () => this.handleNodeClick(node));
        element.querySelector('.node-delete').addEventListener('click', (e) => {
            e.stopPropagation();
            this.deleteNode(nodeId);
        });
        
        // 使节点可拖拽
        this.makeNodeDraggable(element, node);
        
        // 添加到DOM和数组
        this.canvas.appendChild(element);
        this.networkNodes.push(node);
        
        console.log(`✅ 创建节点: ${nodeId} (${component.name})`);
    }
    
    // 格式化节点参数显示
    formatNodeParams(params) {
        const paramStrings = Object.entries(params).map(([key, value]) => {
            if (typeof value.default !== 'undefined') {
                return `${key}: ${value.default}`;
            }
            return `${key}: ${value}`;
        }).slice(0, 3); // 只显示前3个参数
        
        return paramStrings.join(', ');
    }
    
    // 使节点可拖拽
    makeNodeDraggable(element, node) {
        let isDragging = false;
        let startX, startY, initialX, initialY;
        
        const startDrag = (e) => {
            isDragging = true;
            element.classList.add('dragging');
            
            const clientX = e.clientX || (e.touches && e.touches[0].clientX);
            const clientY = e.clientY || (e.touches && e.touches[0].clientY);
            
            startX = clientX;
            startY = clientY;
            initialX = node.x;
            initialY = node.y;
            
            document.addEventListener('mousemove', drag);
            document.addEventListener('mouseup', stopDrag);
            document.addEventListener('touchmove', drag);
            document.addEventListener('touchend', stopDrag);
            
            e.preventDefault();
        };
        
        const drag = (e) => {
            if (!isDragging) return;
            
            const clientX = e.clientX || (e.touches && e.touches[0].clientX);
            const clientY = e.clientY || (e.touches && e.touches[0].clientY);
            
            const deltaX = clientX - startX;
            const deltaY = clientY - startY;
            
            node.x = Math.max(0, initialX + deltaX);
            node.y = Math.max(0, initialY + deltaY);
            
            element.style.left = `${node.x}px`;
            element.style.top = `${node.y}px`;
            
            e.preventDefault();
        };
        
        const stopDrag = () => {
            isDragging = false;
            element.classList.remove('dragging');
            
            document.removeEventListener('mousemove', drag);
            document.removeEventListener('mouseup', stopDrag);
            document.removeEventListener('touchmove', drag);
            document.removeEventListener('touchend', stopDrag);
        };
        
        element.addEventListener('mousedown', startDrag);
        element.addEventListener('touchstart', startDrag);
    }
    
    // 处理节点点击
    handleNodeClick(node) {
        // 移除其他节点的选中状态
        document.querySelectorAll('.network-node').forEach(el => {
            el.classList.remove('selected');
        });
        
        // 选中当前节点
        const element = document.querySelector(`[data-node-id="${node.id}"]`);
        element?.classList.add('selected');
        
        this.selectedNode = node;
        this.showNodeProperties(node);
    }
    
    // 显示节点属性
    showNodeProperties(node) {
        const container = this.propertiesPanel;
        if (!container) return;
        
        container.innerHTML = `
            <div class="property-group">
                <h4>基本信息</h4>
                <div class="property-item">
                    <label>节点ID:</label>
                    <input type="text" value="${node.id}" readonly>
                </div>
                <div class="property-item">
                    <label>组件类型:</label>
                    <input type="text" value="${node.component.name}" readonly>
                </div>
            </div>
            
            <div class="property-group">
                <h4>参数配置</h4>
                ${this.renderParameterInputs(node)}
            </div>
            
            <div class="property-group">
                <button class="btn btn-primary" onclick="smartNetApp.applyNodeChanges('${node.id}')">
                    <i class="fas fa-check"></i> 应用更改
                </button>
                <button class="btn btn-danger" onclick="smartNetApp.deleteNode('${node.id}')">
                    <i class="fas fa-trash"></i> 删除节点
                </button>
            </div>
        `;
    }
    
    // 渲染参数输入框
    renderParameterInputs(node) {
        const params = node.component.params || {};
        
        return Object.entries(params).map(([key, config]) => {
            const currentValue = node.params[key]?.default || node.params[key] || config.default;
            
            let input = '';
            if (config.type === 'int' || config.type === 'float') {
                input = `<input type="number" id="param_${key}" value="${currentValue}" 
                         min="${config.min || 0}" max="${config.max || 9999}" 
                         step="${config.type === 'float' ? '0.0001' : '1'}">`;
            } else if (config.type === 'select') {
                const options = config.options?.map(opt => 
                    `<option value="${opt}" ${opt === currentValue ? 'selected' : ''}>${opt}</option>`
                ).join('') || '';
                input = `<select id="param_${key}">${options}</select>`;
            } else if (config.type === 'list') {
                input = `<input type="text" id="param_${key}" value="${JSON.stringify(currentValue)}" 
                         placeholder="[值1, 值2, ...]">`;
            } else {
                input = `<input type="text" id="param_${key}" value="${currentValue}">`;
            }
            
            return `
                <div class="property-item">
                    <label>${key}:</label>
                    ${input}
                    ${config.min !== undefined || config.max !== undefined ? 
                      `<small>范围: ${config.min || 0} - ${config.max || '∞'}</small>` : ''}
                </div>
            `;
        }).join('');
    }
    
    // 应用节点参数更改
    applyNodeChanges(nodeId) {
        const node = this.networkNodes.find(n => n.id === nodeId);
        if (!node) return;
        
        const params = node.component.params || {};
        
        // 更新参数值
        Object.keys(params).forEach(key => {
            const input = document.getElementById(`param_${key}`);
            if (input) {
                const config = params[key];
                let value = input.value;
                
                // 类型转换
                if (config.type === 'int') {
                    value = parseInt(value) || config.default;
                } else if (config.type === 'float') {
                    value = parseFloat(value) || config.default;
                } else if (config.type === 'list') {
                    try {
                        value = JSON.parse(value);
                    } catch {
                        value = config.default;
                    }
                }
                
                node.params[key] = value;
            }
        });
        
        // 更新节点显示
        const element = document.querySelector(`[data-node-id="${nodeId}"]`);
        const paramsElement = element?.querySelector(`#${nodeId}_params`);
        if (paramsElement) {
            paramsElement.textContent = this.formatNodeParams(node.params);
        }
        
        this.updateCanvasStats();
        this.showNotification('参数已更新', 'success');
    }
    
    // 删除节点
    deleteNode(nodeId) {
        // 从数组中移除
        this.networkNodes = this.networkNodes.filter(n => n.id !== nodeId);
        
        // 从DOM中移除
        const element = document.querySelector(`[data-node-id="${nodeId}"]`);
        element?.remove();
        
        // 如果是当前选中的节点，清除属性面板
        if (this.selectedNode?.id === nodeId) {
            this.selectedNode = null;
            this.clearPropertiesPanel();
        }
        
        this.updateCanvasStats();
        
        // 如果没有节点了，显示提示
        if (this.networkNodes.length === 0) {
            document.getElementById('canvasHint').style.display = 'block';
        }
        
        console.log(`✅ 删除节点: ${nodeId}`);
    }
    
    // 清空属性面板
    clearPropertiesPanel() {
        if (this.propertiesPanel) {
            this.propertiesPanel.innerHTML = `
                <div class="no-selection">
                    <i class="fas fa-mouse-pointer"></i>
                    <p>选择一个组件来查看和编辑其属性</p>
                </div>
            `;
        }
    }
    
    // 更新画布统计信息
    updateCanvasStats() {
        const statsElement = document.getElementById('canvasStats');
        if (statsElement) {
            const paramCount = this.estimateParameters();
            statsElement.textContent = 
                `组件: ${this.networkNodes.length} | 连接: ${this.connections.length} | 参数: ${paramCount.toLocaleString()}`;
        }
    }
    
    // 估算参数数量
    estimateParameters() {
        let total = 0;
        
        this.networkNodes.forEach(node => {
            const type = node.component.id;
            const params = node.params;
            
            // 简化参数估算
            if (type === 'mlp') {
                const inputDim = params.input_dim || 128;
                const outputDim = params.output_dim || 64;
                const hiddenDims = params.hidden_dims || [96];
                
                let prevDim = inputDim;
                hiddenDims.forEach(dim => {
                    total += prevDim * dim + dim;
                    prevDim = dim;
                });
                total += prevDim * outputDim + outputDim;
                
            } else if (type === 'transformer') {
                const dModel = params.input_dim || 256;
                const ffDim = params.ff_dim || 512;
                total += 3 * dModel * dModel + 2 * dModel * ffDim + 4 * dModel;
                
            } else if (type === 'cnn') {
                const inChannels = params.input_channels || 3;
                const outChannels = params.output_channels || 32;
                const kernelSize = params.kernel_size || 3;
                total += inChannels * outChannels * kernelSize * kernelSize + outChannels;
                
            } else if (type === 'rnn') {
                const inputSize = params.input_size || 128;
                const hiddenSize = params.hidden_size || 64;
                const numLayers = params.num_layers || 1;
                total += numLayers * 4 * (inputSize * hiddenSize + hiddenSize * hiddenSize + hiddenSize);
            }
        });
        
        return total;
    }
    
    // 清空画布
    clearCanvas() {
        if (this.networkNodes.length === 0) return;
        
        if (!confirm('确定要清空所有组件吗？此操作无法撤销。')) return;
        
        // 清除所有节点
        this.networkNodes = [];
        this.connections = [];
        this.selectedNode = null;
        
        // 清除DOM元素
        document.querySelectorAll('.network-node').forEach(el => el.remove());
        document.querySelectorAll('.connection-line').forEach(el => el.remove());
        
        // 显示提示
        document.getElementById('canvasHint').style.display = 'block';
        
        // 清空属性面板
        this.clearPropertiesPanel();
        
        // 更新统计
        this.updateCanvasStats();
        
        // 禁用训练按钮
        if (this.trainBtn) {
            this.trainBtn.disabled = true;
        }
        
        console.log('✅ 画布已清空');
    }
    
    // 构建网络
    async buildNetwork() {
        if (this.networkNodes.length === 0) {
            this.showNotification('请先添加网络组件', 'warning');
            return;
        }
        
        // 获取网络配置
        const networkConfig = {
            name: document.getElementById('networkName')?.value || 'my_network',
            input_features: parseInt(document.getElementById('inputFeatures')?.value) || 128,
            output_features: parseInt(document.getElementById('outputFeatures')?.value) || 1,
            learning_rate: parseFloat(document.getElementById('learningRate')?.value) || 0.0001
        };
        
        // 准备组件数据
        const components = this.networkNodes.map((node, index) => ({
            id: node.id,
            type: node.component.id,
            params: node.params,
            position: { x: node.x, y: node.y },
            order: index
        }));
        
        const buildData = {
            components,
            connections: this.connections,
            config: networkConfig
        };
        
        console.log('🔧 开始构建网络:', buildData);
        
        try {
            this.buildBtn.disabled = true;
            this.buildBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 构建中...';
            
            const response = await fetch('/api/build', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(buildData)
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showBuildResult(result);
                this.trainBtn.disabled = false;
                console.log('✅ 网络构建成功');
            } else {
                this.showBuildError(result.error);
                console.error('❌ 网络构建失败:', result.error);
            }
            
        } catch (error) {
            console.error('❌ 构建请求失败:', error);
            this.showNotification('网络构建出错', 'error');
        } finally {
            this.buildBtn.disabled = false;
            this.buildBtn.innerHTML = '<i class="fas fa-cogs"></i> 构建网络';
        }
    }
    
    // 显示构建结果
    showBuildResult(result) {
        const modalBody = document.getElementById('buildModalBody');
        modalBody.innerHTML = `
            <div class="build-result success">
                <i class="fas fa-check-circle"></i>
                <h4>网络构建成功！</h4>
                
                <div class="build-stats">
                    <div class="stat-card">
                        <div class="value">${result.parameter_count.toLocaleString()}</div>
                        <div class="label">参数数量</div>
                    </div>
                    <div class="stat-card">
                        <div class="value">${result.memory_estimate || 0}MB</div>
                        <div class="label">内存估算</div>
                    </div>
                    <div class="stat-card">
                        <div class="value">${result.training_feasible ? '是' : '否'}</div>
                        <div class="label">可训练</div>
                    </div>
                </div>
                
                <p>网络已成功构建并验证，可以开始训练。</p>
                ${!result.training_feasible ? 
                    '<p class="warning">⚠️ 网络过大，可能无法在双3090上正常训练</p>' : ''}
            </div>
        `;
        
        document.getElementById('proceedTrainBtn').style.display = 
            result.training_feasible ? 'inline-block' : 'none';
        
        this.showModal('buildModal');
    }
    
    // 显示构建错误
    showBuildError(error) {
        const modalBody = document.getElementById('buildModalBody');
        modalBody.innerHTML = `
            <div class="build-result error">
                <i class="fas fa-exclamation-circle"></i>
                <h4>网络构建失败</h4>
                <p>${error}</p>
                
                <div class="error-suggestions">
                    <h5>建议：</h5>
                    <ul>
                        <li>检查网络深度是否超过10层</li>
                        <li>确保参数数量在5000万以内</li>
                        <li>验证组件参数配置是否正确</li>
                        <li>确保输入输出维度匹配</li>
                    </ul>
                </div>
            </div>
        `;
        
        document.getElementById('proceedTrainBtn').style.display = 'none';
        this.showModal('buildModal');
    }
    
    // 显示训练配置对话框
    async showTrainingDialog() {
        this.hideModal('buildModal');
        
        // 加载GPU信息
        await this.loadGPUInfo();
        
        this.showModal('trainModal');
    }
    
    // 加载GPU信息
    async loadGPUInfo() {
        try {
            const response = await fetch('/api/system_info');
            const data = await response.json();
            
            if (data.success && data.system.gpu_info) {
                this.renderGPUOptions(data.system.gpu_info);
            }
        } catch (error) {
            console.error('❌ 加载GPU信息失败:', error);
        }
    }
    
    // 渲染GPU选项
    renderGPUOptions(gpuInfo) {
        const container = document.getElementById('gpuOptions');
        if (!container) return;
        
        container.innerHTML = gpuInfo.map((gpu, index) => `
            <div class="gpu-option">
                <input type="checkbox" id="gpu_${index}" value="${index}" 
                       ${index < 2 ? 'checked' : ''}>
                <div class="gpu-info">
                    <div class="gpu-name">GPU ${index}: ${gpu.name}</div>
                    <div class="gpu-memory">${(gpu.memory_total / 1024).toFixed(1)}GB 显存</div>
                </div>
            </div>
        `).join('');
    }
    
    // 开始训练
    async startTraining() {
        const trainingConfig = {
            epochs: parseInt(document.getElementById('trainEpochs')?.value) || 10,
            batch_size: parseInt(document.getElementById('trainBatchSize')?.value) || 32,
            learning_rate: parseFloat(document.getElementById('trainLearningRate')?.value) || 0.0001,
            gpus: Array.from(document.querySelectorAll('#gpuOptions input:checked'))
                       .map(input => parseInt(input.value))
        };
        
        console.log('🎯 开始训练:', trainingConfig);
        
        try {
            const response = await fetch('/api/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(trainingConfig)
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.currentTrainingId = result.training_id;
                this.hideModal('trainModal');
                this.showTrainingProgress();
                this.startProgressMonitoring();
                console.log(`✅ 训练开始: ${result.training_id}`);
            } else {
                this.showNotification(result.error, 'error');
                console.error('❌ 训练启动失败:', result.error);
            }
            
        } catch (error) {
            console.error('❌ 训练请求失败:', error);
            this.showNotification('训练启动出错', 'error');
        }
    }
    
    // 显示训练进度
    showTrainingProgress() {
        this.showModal('progressModal');
        
        // 重置进度显示
        document.getElementById('currentEpoch').textContent = '0';
        document.getElementById('currentLoss').textContent = '--';
        document.getElementById('currentAccuracy').textContent = '--';
        document.getElementById('progressFill').style.width = '0%';
        document.getElementById('progressText').textContent = '训练已开始...';
        
        // 清空日志
        const logContainer = document.getElementById('trainingLog');
        logContainer.innerHTML = '<div class="log-entry info">训练任务已启动...</div>';
    }
    
    // 开始进度监控
    startProgressMonitoring() {
        if (!this.currentTrainingId) return;
        
        this.progressInterval = setInterval(async () => {
            try {
                const response = await fetch(`/api/training/${this.currentTrainingId}`);
                const data = await response.json();
                
                if (data.success) {
                    this.updateTrainingProgress(data.job);
                    
                    if (data.job.status === 'completed' || data.job.status === 'failed') {
                        this.stopProgressMonitoring();
                        this.onTrainingComplete(data.job);
                    }
                } else {
                    console.error('❌ 获取训练状态失败:', data.error);
                }
            } catch (error) {
                console.error('❌ 进度查询出错:', error);
            }
        }, 2000); // 每2秒查询一次
    }
    
    // 更新训练进度
    updateTrainingProgress(job) {
        // 更新进度数据
        document.getElementById('currentEpoch').textContent = job.current_epoch;
        document.getElementById('totalEpochs').textContent = job.epochs;
        document.getElementById('currentLoss').textContent = 
            job.loss ? job.loss.toFixed(4) : '--';
        document.getElementById('currentAccuracy').textContent = 
            job.accuracy ? (job.accuracy * 100).toFixed(2) + '%' : '--';
        
        // 更新进度条
        const progress = (job.current_epoch / job.epochs) * 100;
        document.getElementById('progressFill').style.width = `${progress}%`;
        document.getElementById('progressText').textContent = 
            `训练进行中... ${progress.toFixed(1)}%`;
        
        // 添加日志条目
        this.addTrainingLog('info', 
            `Epoch ${job.current_epoch}/${job.epochs} - Loss: ${job.loss?.toFixed(4) || 'N/A'} - Acc: ${job.accuracy ? (job.accuracy * 100).toFixed(2) + '%' : 'N/A'}`);
    }
    
    // 添加训练日志
    addTrainingLog(type, message) {
        const logContainer = document.getElementById('trainingLog');
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry ${type}`;
        logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        
        logContainer.appendChild(logEntry);
        logContainer.scrollTop = logContainer.scrollHeight;
        
        // 限制日志条目数量
        const entries = logContainer.querySelectorAll('.log-entry');
        if (entries.length > 50) {
            entries[0].remove();
        }
    }
    
    // 停止进度监控
    stopProgressMonitoring() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
    }
    
    // 训练完成处理
    onTrainingComplete(job) {
        if (job.status === 'completed') {
            document.getElementById('progressText').textContent = '训练完成！';
            this.addTrainingLog('success', '🎉 训练成功完成！');
            this.showNotification('训练完成', 'success');
        } else {
            document.getElementById('progressText').textContent = '训练失败';
            this.addTrainingLog('error', '❌ 训练失败或被中断');
            this.showNotification('训练失败', 'error');
        }
        
        // 启用关闭按钮
        document.getElementById('closeProgressBtn').disabled = false;
        document.getElementById('stopTrainBtn').style.display = 'none';
    }
    
    // 停止训练
    stopTraining() {
        if (!confirm('确定要停止当前训练吗？')) return;
        
        this.stopProgressMonitoring();
        this.addTrainingLog('warning', '⚠️ 用户手动停止训练');
        document.getElementById('progressText').textContent = '训练已停止';
        document.getElementById('closeProgressBtn').disabled = false;
        document.getElementById('stopTrainBtn').style.display = 'none';
        
        console.log('🛑 训练已停止');
    }
    
    // 加载系统信息
    async loadSystemInfo() {
        try {
            const response = await fetch('/api/system_info');
            const data = await response.json();
            
            if (data.success) {
                this.updateSystemStatus(data.system);
            }
        } catch (error) {
            console.error('❌ 加载系统信息失败:', error);
        }
    }
    
    // 更新系统状态显示
    updateSystemStatus(system) {
        const gpuInfo = document.getElementById('gpuInfo');
        if (gpuInfo) {
            if (system.gpu_available) {
                gpuInfo.innerHTML = `GPU: ${system.gpu_count}x RTX 3090 (${system.gpu_count * 24}GB)`;
                gpuInfo.style.color = '#27ae60';
            } else {
                gpuInfo.innerHTML = 'GPU: 不可用';
                gpuInfo.style.color = '#e74c3c';
            }
        }
    }
    
    // 显示模态对话框
    showModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.add('show');
            document.body.style.overflow = 'hidden';
        }
    }
    
    // 隐藏模态对话框
    hideModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.remove('show');
            document.body.style.overflow = 'auto';
        }
    }
    
    // 显示通知
    showNotification(message, type = 'info') {
        // 创建通知元素
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'success' ? '#2ecc71' : 
                        type === 'error' ? '#e74c3c' : 
                        type === 'warning' ? '#f39c12' : '#3498db'};
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            z-index: 10000;
            font-weight: 600;
            animation: slideInRight 0.3s ease-out;
        `;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // 自动移除
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease-out';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
}

// 创建全局应用实例
const smartNetApp = new SmartNetApp();

// 导出到全局作用域
window.SmartNetApp = SmartNetApp;
window.smartNetApp = smartNetApp;

// 添加样式动画
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOutRight {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);
