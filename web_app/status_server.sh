#!/bin/bash
# SmartNet Web应用状态检查和日志查看脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/smartnet_web.pid"
LOG_DIR="$SCRIPT_DIR/logs"

echo "📊 SmartNet Web应用状态检查"
echo "=" * 50

# 检查运行状态
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "✅ 应用状态: 运行中"
        echo "🆔 进程ID: $PID"
        echo "🌐 访问地址: http://localhost:5000"
        
        # 显示进程信息
        echo ""
        echo "📈 进程信息:"
        ps -p "$PID" -o pid,ppid,cpu,pmem,etime,cmd 2>/dev/null || echo "无法获取进程信息"
        
        # 显示端口信息
        echo ""
        echo "🌍 端口信息:"
        netstat -tlnp 2>/dev/null | grep ":5000 " | head -1 || echo "端口5000未监听"
        
    else
        echo "❌ 应用状态: 已停止"
        echo "⚠️  PID文件存在但进程不存在，清理中..."
        rm -f "$PID_FILE"
    fi
else
    echo "❌ 应用状态: 未运行"
fi

# 显示最新日志文件
echo ""
echo "📝 日志文件:"
if [ -d "$LOG_DIR" ]; then
    LATEST_LOG=$(ls -t "$LOG_DIR"/smartnet_web_*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo "   最新日志: $LATEST_LOG"
        echo "   文件大小: $(du -h "$LATEST_LOG" | cut -f1)"
        echo "   修改时间: $(stat -c %y "$LATEST_LOG" 2>/dev/null || stat -f %Sm "$LATEST_LOG" 2>/dev/null)"
        
        echo ""
        echo "📄 最近10行日志:"
        echo "---"
        tail -n 10 "$LATEST_LOG" 2>/dev/null || echo "无法读取日志文件"
        echo "---"
        
    else
        echo "   无日志文件"
    fi
else
    echo "   日志目录不存在"
fi

echo ""
echo "🔧 管理命令:"
echo "   启动服务: $SCRIPT_DIR/start_server.sh"
echo "   停止服务: $SCRIPT_DIR/stop_server.sh"
echo "   查看状态: $SCRIPT_DIR/status_server.sh"
if [ -n "$LATEST_LOG" ]; then
    echo "   实时日志: tail -f $LATEST_LOG"
fi
echo "   打开浏览器: open http://localhost:5000 (macOS) 或 xdg-open http://localhost:5000 (Linux)"
