"""
性能分析模块

提供函数执行时间分析和内存使用监控
"""

import functools
import time
import tracemalloc
from typing import Callable, Any, Optional
import logging

logger = logging.getLogger(__name__)


def timer_decorator(func: Callable) -> Callable:
    """
    函数执行时间装饰器
    
    记录函数的执行时间并打印日志
    
    Example:
        @timer_decorator
        def my_function():
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        
        logger.info(f"{func.__name__} 执行时间: {elapsed:.4f} 秒")
        return result
    
    return wrapper


def memory_profiler(func: Callable) -> Callable:
    """
    内存使用监控装饰器
    
    记录函数的内存使用情况
    
    Example:
        @memory_profiler
        def my_function():
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        
        result = func(*args, **kwargs)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        current_mb = current / 1024 / 1024
        peak_mb = peak / 1024 / 1024
        
        logger.info(
            f"{func.__name__} 内存使用: "
            f"当前={current_mb:.2f}MB, 峰值={peak_mb:.2f}MB"
        )
        
        return result
    
    return wrapper


class Timer:
    """
    上下文管理器形式的计时器
    
    Example:
        with Timer("数据处理"):
            process_data()
    """
    
    def __init__(self, name: str = "操作", logger_func: Optional[Callable] = None):
        self.name = name
        self.logger_func = logger_func or logger.info
        self.start_time: Optional[float] = None
        self.elapsed: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        self.logger_func(f"{self.name} 耗时: {self.elapsed:.4f} 秒")
    
    def __str__(self):
        if self.elapsed is not None:
            return f"{self.name}: {self.elapsed:.4f} 秒"
        return f"{self.name}: 未开始"


class PerformanceMonitor:
    """
    性能监控类
    
    用于收集和分析多个操作的性能数据
    
    Example:
        monitor = PerformanceMonitor()
        
        with monitor.track("数据加载"):
            load_data()
        
        with monitor.track("模型训练"):
            train_model()
        
        monitor.report()
    """
    
    def __init__(self):
        self.timings: dict = {}
        self.counts: dict = {}
    
    def track(self, name: str):
        """返回一个上下文管理器用于追踪操作"""
        return _TrackedOperation(self, name)
    
    def add_timing(self, name: str, elapsed: float):
        """添加计时数据"""
        if name not in self.timings:
            self.timings[name] = []
            self.counts[name] = 0
        
        self.timings[name].append(elapsed)
        self.counts[name] += 1
    
    def report(self) -> str:
        """生成性能报告"""
        if not self.timings:
            return "没有性能数据"
        
        lines = ["\n========== 性能报告 =========="]
        
        for name in sorted(self.timings.keys()):
            times = self.timings[name]
            count = self.counts[name]
            total = sum(times)
            avg = total / count
            min_time = min(times)
            max_time = max(times)
            
            lines.append(f"\n{name}:")
            lines.append(f"  调用次数: {count}")
            lines.append(f"  总时间: {total:.4f} 秒")
            lines.append(f"  平均时间: {avg:.4f} 秒")
            lines.append(f"  最小时间: {min_time:.4f} 秒")
            lines.append(f"  最大时间: {max_time:.4f} 秒")
        
        lines.append("==============================\n")
        
        return "\n".join(lines)
    
    def reset(self):
        """重置所有数据"""
        self.timings.clear()
        self.counts.clear()


class _TrackedOperation:
    """被追踪的操作（内部类）"""
    
    def __init__(self, monitor: PerformanceMonitor, name: str):
        self.monitor = monitor
        self.name = name
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        self.monitor.add_timing(self.name, elapsed)
