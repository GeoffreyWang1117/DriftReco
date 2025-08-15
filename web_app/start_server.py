#!/usr/bin/env python3
"""
SmartNet Web应用启动脚本
将所有输出重定向到日志文件，支持后台运行
"""

import os
import sys
import logging
import signal
from datetime import datetime
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from web_app.app import create_app


def setup_logging():
    """设置日志系统"""
    # 创建日志目录
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # 日志文件名包含时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"smartnet_web_{timestamp}.log"
    
    # 配置根日志器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 保留控制台输出用于启动信息
        ]
    )
    
    # 设置Flask和Werkzeug日志级别
    logging.getLogger('werkzeug').setLevel(logging.INFO)
    logging.getLogger('flask').setLevel(logging.INFO)
    
    return log_file


def create_pid_file():
    """创建PID文件用于进程管理"""
    pid_file = Path(__file__).parent / "smartnet_web.pid"
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))
    return pid_file


def signal_handler(signum, frame):
    """信号处理器，用于优雅关闭"""
    logger = logging.getLogger(__name__)
    logger.info(f"收到信号 {signum}，正在关闭服务...")
    
    # 清理PID文件
    pid_file = Path(__file__).parent / "smartnet_web.pid"
    if pid_file.exists():
        pid_file.unlink()
    
    logger.info("SmartNet Web服务已关闭")
    sys.exit(0)


def main():
    """主函数"""
    # 设置日志
    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 创建PID文件
    pid_file = create_pid_file()
    
    logger.info("=" * 60)
    logger.info("🚀 SmartNet Web应用启动")
    logger.info("=" * 60)
    logger.info(f"📁 项目根目录: {project_root}")
    logger.info(f"📝 日志文件: {log_file}")
    logger.info(f"🆔 PID文件: {pid_file}")
    logger.info(f"🌐 Web地址: http://localhost:5000")
    
    try:
        # 创建Flask应用
        app = create_app()
        
        logger.info("✅ Flask应用创建成功")
        logger.info("🎯 可视化拖拽界面已就绪")
        logger.info("🔧 GPU训练后端已配置")
        
        # 启动参数
        host = os.getenv('FLASK_HOST', '0.0.0.0')
        port = int(os.getenv('FLASK_PORT', 5000))
        debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
        
        logger.info(f"🌍 服务器配置: {host}:{port}")
        logger.info(f"🐛 调试模式: {debug}")
        
        print(f"🚀 SmartNet Web应用正在启动...")
        print(f"📝 日志文件: {log_file}")
        print(f"🌐 访问地址: http://localhost:{port}")
        print(f"📊 查看实时日志: tail -f {log_file}")
        print(f"🛑 停止服务: kill -TERM {os.getpid()}")
        
        # 启动Flask服务器
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True,
            use_reloader=False  # 避免重载器干扰日志
        )
        
    except KeyboardInterrupt:
        logger.info("用户中断，正在关闭服务...")
        signal_handler(signal.SIGINT, None)
        
    except Exception as e:
        logger.error(f"启动失败: {e}")
        logger.exception("详细错误信息:")
        sys.exit(1)


if __name__ == "__main__":
    main()
