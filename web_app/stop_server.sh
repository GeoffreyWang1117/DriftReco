#!/bin/bash
# SmartNet Web应用停止脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/smartnet_web.pid"

echo "🛑 停止SmartNet Web应用..."

if [ ! -f "$PID_FILE" ]; then
    echo "❌ PID文件不存在，应用可能未运行"
    exit 1
fi

PID=$(cat "$PID_FILE")

if ! kill -0 "$PID" 2>/dev/null; then
    echo "❌ 进程不存在 (PID: $PID)"
    rm -f "$PID_FILE"
    exit 1
fi

echo "🔄 正在停止进程 (PID: $PID)..."

# 尝试优雅停止
kill -TERM "$PID"
sleep 3

# 检查是否还在运行
if kill -0 "$PID" 2>/dev/null; then
    echo "⚠️  进程仍在运行，强制终止..."
    kill -KILL "$PID"
    sleep 1
fi

# 再次检查
if ! kill -0 "$PID" 2>/dev/null; then
    echo "✅ SmartNet Web应用已成功停止"
    rm -f "$PID_FILE"
else
    echo "❌ 无法停止进程"
    exit 1
fi
