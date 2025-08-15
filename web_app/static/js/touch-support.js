// 移动端触摸拖拽支持
class TouchDragSupport {
    constructor(app) {
        this.app = app;
        this.isDragging = false;
        this.draggedElement = null;
        this.touchStartPos = { x: 0, y: 0 };
        this.dragThreshold = 10; // 移动多少像素才开始拖拽
        this.longPressDelay = 150; // 长按延迟(毫秒)
        this.longPressTimer = null;
        this.setupTouchEvents();
    }

    setupTouchEvents() {
        // 为组件面板添加触摸事件
        document.querySelectorAll('.component-item').forEach(item => {
            this.addTouchListeners(item);
        });

        // 为画布添加触摸事件
        this.app.canvas.addEventListener('touchstart', this.handleCanvasTouchStart.bind(this), { passive: false });
        this.app.canvas.addEventListener('touchmove', this.handleCanvasTouchMove.bind(this), { passive: false });
        this.app.canvas.addEventListener('touchend', this.handleCanvasTouchEnd.bind(this), { passive: false });

        // 为网络节点添加触摸事件（动态添加）
        this.observeNetworkNodes();
    }

    addTouchListeners(element) {
        element.addEventListener('touchstart', this.handleTouchStart.bind(this), { passive: false });
        element.addEventListener('touchmove', this.handleTouchMove.bind(this), { passive: false });
        element.addEventListener('touchend', this.handleTouchEnd.bind(this), { passive: false });
        element.addEventListener('touchcancel', this.handleTouchEnd.bind(this), { passive: false });
    }

    handleTouchStart(e) {
        // 阻止默认行为，防止页面滚动
        e.preventDefault();
        
        const touch = e.touches[0];
        const element = e.currentTarget;
        
        this.touchStartPos = { x: touch.clientX, y: touch.clientY };
        this.draggedElement = element;

        // 设置长按定时器
        this.longPressTimer = setTimeout(() => {
            this.startDrag(element, touch);
        }, this.longPressDelay);

        // 添加视觉反馈
        element.style.transform = 'scale(0.95)';
        element.style.transition = 'transform 0.1s ease';
    }

    handleTouchMove(e) {
        if (!this.draggedElement) return;

        const touch = e.touches[0];
        const deltaX = Math.abs(touch.clientX - this.touchStartPos.x);
        const deltaY = Math.abs(touch.clientY - this.touchStartPos.y);

        // 如果移动距离超过阈值，取消长按并开始拖拽
        if (deltaX > this.dragThreshold || deltaY > this.dragThreshold) {
            if (this.longPressTimer) {
                clearTimeout(this.longPressTimer);
                this.longPressTimer = null;
            }

            if (!this.isDragging) {
                this.startDrag(this.draggedElement, touch);
            }

            if (this.isDragging) {
                e.preventDefault();
                this.updateDragVisual(touch);
            }
        }
    }

    handleTouchEnd(e) {
        // 清除定时器
        if (this.longPressTimer) {
            clearTimeout(this.longPressTimer);
            this.longPressTimer = null;
        }

        // 恢复元素状态
        if (this.draggedElement) {
            this.draggedElement.style.transform = '';
            this.draggedElement.style.transition = '';
        }

        if (this.isDragging) {
            e.preventDefault();
            this.endDrag(e.changedTouches[0]);
        }

        this.resetDragState();
    }

    startDrag(element, touch) {
        this.isDragging = true;
        
        // 获取组件类型
        const componentType = element.dataset.componentType;
        if (componentType) {
            this.app.draggedComponent = this.app.components.find(c => c.type === componentType);
        }

        // 添加视觉效果
        element.classList.add('dragging');
        this.app.canvas.classList.add('drag-over');

        // 创建拖拽指示器
        this.createDragIndicator(touch);

        // 触觉反馈 (如果支持)
        if (navigator.vibrate) {
            navigator.vibrate(50);
        }
    }

    updateDragVisual(touch) {
        // 更新拖拽指示器位置
        if (this.dragIndicator) {
            this.dragIndicator.style.left = (touch.clientX - 40) + 'px';
            this.dragIndicator.style.top = (touch.clientY - 40) + 'px';
        }

        // 检查是否在画布上
        const canvasRect = this.app.canvas.getBoundingClientRect();
        const isOverCanvas = touch.clientX >= canvasRect.left && 
                           touch.clientX <= canvasRect.right &&
                           touch.clientY >= canvasRect.top && 
                           touch.clientY <= canvasRect.bottom;

        if (isOverCanvas) {
            this.app.canvas.classList.add('drag-over');
        } else {
            this.app.canvas.classList.remove('drag-over');
        }
    }

    endDrag(touch) {
        // 检查是否在画布上释放
        const canvasRect = this.app.canvas.getBoundingClientRect();
        const isOverCanvas = touch.clientX >= canvasRect.left && 
                           touch.clientX <= canvasRect.right &&
                           touch.clientY >= canvasRect.top && 
                           touch.clientY <= canvasRect.bottom;

        if (isOverCanvas && this.app.draggedComponent) {
            // 计算画布内的相对位置
            const x = touch.clientX - canvasRect.left;
            const y = touch.clientY - canvasRect.top;
            
            // 创建网络节点
            this.app.createNetworkNode(this.app.draggedComponent, x, y);

            // 成功反馈
            if (navigator.vibrate) {
                navigator.vibrate([50, 100, 50]);
            }
        }

        // 清理拖拽指示器
        this.removeDragIndicator();
    }

    createDragIndicator(touch) {
        this.dragIndicator = document.createElement('div');
        this.dragIndicator.className = 'drag-indicator';
        this.dragIndicator.style.cssText = `
            position: fixed;
            width: 80px;
            height: 80px;
            background: rgba(52, 152, 219, 0.8);
            border: 2px solid #3498db;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 0.8rem;
            z-index: 10000;
            pointer-events: none;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            animation: dragPulse 1s ease-in-out infinite;
        `;
        
        if (this.app.draggedComponent) {
            this.dragIndicator.textContent = this.app.draggedComponent.icon + ' ' + this.app.draggedComponent.name;
        }

        // 添加CSS动画
        if (!document.querySelector('#drag-animations')) {
            const style = document.createElement('style');
            style.id = 'drag-animations';
            style.textContent = `
                @keyframes dragPulse {
                    0%, 100% { transform: scale(1); }
                    50% { transform: scale(1.1); }
                }
            `;
            document.head.appendChild(style);
        }

        document.body.appendChild(this.dragIndicator);
        
        // 初始位置
        this.dragIndicator.style.left = (touch.clientX - 40) + 'px';
        this.dragIndicator.style.top = (touch.clientY - 40) + 'px';
    }

    removeDragIndicator() {
        if (this.dragIndicator) {
            document.body.removeChild(this.dragIndicator);
            this.dragIndicator = null;
        }
    }

    resetDragState() {
        this.isDragging = false;
        this.draggedElement = null;
        this.app.draggedComponent = null;
        this.app.canvas.classList.remove('drag-over');
        
        // 清理所有拖拽样式
        document.querySelectorAll('.dragging').forEach(el => {
            el.classList.remove('dragging');
        });
    }

    // 处理画布上的触摸事件
    handleCanvasTouchStart(e) {
        const touch = e.touches[0];
        const element = document.elementFromPoint(touch.clientX, touch.clientY);
        
        if (element && element.classList.contains('network-node')) {
            e.preventDefault();
            this.setupNodeDrag(element, touch);
        }
    }

    handleCanvasTouchMove(e) {
        if (this.draggingNode) {
            e.preventDefault();
            const touch = e.touches[0];
            const canvasRect = this.app.canvas.getBoundingClientRect();
            const x = touch.clientX - canvasRect.left - 40; // 减去节点宽度的一半
            const y = touch.clientY - canvasRect.top - 40;
            
            this.draggingNode.style.left = Math.max(0, Math.min(x, canvasRect.width - 80)) + 'px';
            this.draggingNode.style.top = Math.max(0, Math.min(y, canvasRect.height - 80)) + 'px';
        }
    }

    handleCanvasTouchEnd(e) {
        if (this.draggingNode) {
            this.draggingNode.classList.remove('dragging');
            this.draggingNode = null;
        }
    }

    setupNodeDrag(node, touch) {
        this.draggingNode = node;
        node.classList.add('dragging');
        
        // 触觉反馈
        if (navigator.vibrate) {
            navigator.vibrate(30);
        }
    }

    // 观察新添加的网络节点
    observeNetworkNodes() {
        const observer = new MutationObserver(mutations => {
            mutations.forEach(mutation => {
                mutation.addedNodes.forEach(node => {
                    if (node.nodeType === 1 && node.classList.contains('network-node')) {
                        // 为新节点添加触摸支持
                        this.addNodeTouchSupport(node);
                    }
                });
            });
        });

        observer.observe(this.app.canvas, {
            childList: true,
            subtree: true
        });
    }

    addNodeTouchSupport(node) {
        // 为网络节点添加特殊的触摸处理
        node.addEventListener('touchstart', (e) => {
            e.preventDefault();
            node.style.transform = 'scale(1.1)';
            node.style.zIndex = '1000';
        }, { passive: false });

        node.addEventListener('touchend', (e) => {
            e.preventDefault();
            node.style.transform = '';
            node.style.zIndex = '';
        }, { passive: false });
    }

    // 检测设备类型
    static isMobileDevice() {
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    }

    static isIPad() {
        return /iPad/i.test(navigator.userAgent) || 
               (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
    }

    // 优化移动端性能
    optimizeForMobile() {
        if (TouchDragSupport.isMobileDevice()) {
            // 禁用某些动画以提高性能
            document.body.classList.add('mobile-device');
            
            // 添加移动端特定样式
            const mobileStyles = `
                .mobile-device .component-item {
                    transition: transform 0.1s ease !important;
                }
                .mobile-device .network-node {
                    transition: transform 0.1s ease !important;
                }
                .mobile-device .canvas {
                    -webkit-transform-style: flat !important;
                    transform-style: flat !important;
                }
            `;
            
            const styleSheet = document.createElement('style');
            styleSheet.textContent = mobileStyles;
            document.head.appendChild(styleSheet);
        }
    }
}

// 自动初始化移动端支持
document.addEventListener('DOMContentLoaded', () => {
    // 等待主应用初始化
    setTimeout(() => {
        if (window.smartNetApp) {
            const touchSupport = new TouchDragSupport(window.smartNetApp);
            touchSupport.optimizeForMobile();
            window.touchSupport = touchSupport;
        }
    }, 100);
});
