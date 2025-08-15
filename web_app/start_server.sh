#!/bin/bash
# SmartNet Web应用后台启动脚本

# 设置脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$SCRIPT_DIR/logs"
PID_FILE="$SCRIPT_DIR/smartnet_web.pid"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 生成带时间戳的日志文件
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$LOG_DIR/smartnet_web_$TIMESTAMP.log"

# 检查是否已经在运行
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "❌ SmartNet Web应用已在运行 (PID: $PID)"
        echo "   日志查看: tail -f $LOG_DIR/smartnet_web_*.log"
        echo "   停止服务: $SCRIPT_DIR/stop_server.sh"
        exit 1
    else
        echo "清理过期PID文件..."
        rm -f "$PID_FILE"
    fi
fi

echo "🚀 启动SmartNet Web应用..."
echo "📁 项目目录: $PROJECT_DIR"
echo "📝 日志文件: $LOG_FILE"

# 切换到项目目录
cd "$PROJECT_DIR"

# 后台启动Flask应用，重定向所有输出到日志文件
nohup conda run --live-stream --name driftrec python -c "
import sys
import os
import logging
from datetime import datetime
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('$LOG_FILE', encoding='utf-8')]
)

# 重定向标准输出和错误到日志
class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        
    def write(self, message):
        if message.strip():
            self.logger.log(self.level, message.strip())
            
    def flush(self):
        pass

logger = logging.getLogger('smartnet_web')
sys.stdout = LoggerWriter(logger, logging.INFO)
sys.stderr = LoggerWriter(logger, logging.ERROR)

# 添加项目路径
sys.path.insert(0, '$PROJECT_DIR')

try:
    from web_app.app import create_app
    
    logger.info('=' * 60)
    logger.info('🚀 SmartNet Web应用启动')
    logger.info('=' * 60)
    logger.info('📁 项目目录: $PROJECT_DIR')
    logger.info('📝 日志文件: $LOG_FILE')
    logger.info('🌐 访问地址: http://localhost:5000')
    
    # 写入PID文件
    with open('$PID_FILE', 'w') as f:
        f.write(str(os.getpid()))
    
    app = create_app()
    logger.info('✅ Flask应用创建成功')
    logger.info('🎯 可视化拖拽界面已就绪')
    logger.info('🔧 GPU训练后端已配置')
    
    # 启动服务器
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )
    
except Exception as e:
    logger.error('启动失败: %s', str(e))
    logger.exception('详细错误信息:')
    # 清理PID文件
    if os.path.exists('$PID_FILE'):
        os.unlink('$PID_FILE')
    sys.exit(1)
" > "$LOG_FILE" 2>&1 &

# 保存PID到临时文件，等待Python进程写入真实PID
TEMP_PID=$!

# 等待一下让Python进程启动并写入PID
sleep 3

# 检查启动是否成功
if [ -f "$PID_FILE" ]; then
    REAL_PID=$(cat "$PID_FILE")
    if kill -0 "$REAL_PID" 2>/dev/null; then
        echo "✅ SmartNet Web应用启动成功!"
        echo "🆔 进程ID: $REAL_PID"
        echo "🌐 访问地址: http://localhost:5000"
        echo "📊 查看日志: tail -f $LOG_FILE"
        echo "🛑 停止服务: $SCRIPT_DIR/stop_server.sh"
        echo ""
        echo "🎯 Web界面功能:"
        echo "   • 拖拽式神经网络构建"
        echo "   • 实时参数配置"
        echo "   • GPU训练支持"
        echo "   • 模型性能分析"
        echo ""
        echo "📱 快速访问命令:"
        echo "   浏览器打开: open http://localhost:5000"
        echo "   实时日志: tail -f $LOG_FILE"
    else
        echo "❌ 启动失败，请检查日志: $LOG_FILE"
        exit 1
    fi
else
    echo "❌ 启动失败，未找到PID文件，请检查日志: $LOG_FILE"
    exit 1
fi
