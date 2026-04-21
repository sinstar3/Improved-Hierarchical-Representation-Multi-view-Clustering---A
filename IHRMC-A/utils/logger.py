"""
日志模块

提供统一的日志记录功能，替代 print 输出
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "IHRMC-A",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 日志文件路径，None 则不保存到文件
        console: 是否输出到控制台
    
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 格式化器
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 文件处理器
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


# 默认日志记录器
_default_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """获取默认日志记录器"""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logger()
    return _default_logger


def set_default_logger(logger: logging.Logger):
    """设置默认日志记录器"""
    global _default_logger
    _default_logger = logger
