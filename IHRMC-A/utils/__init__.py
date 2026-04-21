"""
工具函数模块

包含以下功能：
1. 聚类评估指标计算
2. 标签重排（基于匈牙利算法）
3. 随机种子设置
4. 设备获取
5. 矩阵分析
6. 日志记录
7. 参数验证
8. 性能分析
"""

from .logger import get_logger, setup_logger, set_default_logger
from .profiling import (
    timer_decorator,
    memory_profiler,
    Timer,
    PerformanceMonitor,
)
from .utils import (
    best_map,
    block_diag_ratio,
    clustering_metrics,
    get_device,
    set_seed,
    to_numpy,
    select_anchors,
)


__all__ = [
    # 工具函数
    'clustering_metrics',
    'best_map',
    'block_diag_ratio',
    'set_seed',
    'get_device',
    'to_numpy',
    'select_anchors',
    # 日志
    'get_logger',
    'setup_logger',
    'set_default_logger',
    # 性能分析
    'timer_decorator',
    'memory_profiler',
    'Timer',
    'PerformanceMonitor',
]
