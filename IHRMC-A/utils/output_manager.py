"""
输出目录管理模块

统一管理所有实验的输出目录结构：
- results/single/          单次实验结果
- results/grid_search/     网格搜索结果
- results/targeted_tuning/ 针对性调优结果
- results/ablation/        消融实验结果
- results/baselines/       基线方法结果
- results/logs/            总体实验日志
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any


# 基础输出目录（使用绝对路径）
BASE_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))

# 子目录定义
OUTPUT_SUBDIRS = {
    'single': 'single',           # 单次实验
    'grid_search': 'grid_search', # 网格搜索
    'targeted': 'targeted_tuning',# 针对性调优
    'ablation': 'ablation',       # 消融实验
    'baseline': 'baselines',      # 基线方法
    'pretrain': 'pretrain',       # 预训练实验
    'logs': 'logs',               # 总体实验日志
}

# 总体日志文件路径
GLOBAL_LOG_FILE = os.path.join(BASE_OUTPUT_DIR, 'logs', 'experiment_log.txt')
GLOBAL_RESULTS_FILE = os.path.join(BASE_OUTPUT_DIR, 'logs', 'experiment_results.json')


def _ensure_log_dir():
    """确保日志目录存在"""
    log_dir = os.path.join(BASE_OUTPUT_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_output_dir(exp_type: str, prefix: Optional[str] = None, 
                   create_dir: bool = True, timestamp: Optional[str] = None, 
                   dataset_name: Optional[str] = None) -> str:
    """
    获取实验输出目录
    
    Args:
        exp_type: 实验类型 ('single', 'grid_search', 'targeted', 'ablation', 'baseline', 'pretrain')
        prefix: 目录前缀（可选）
        create_dir: 是否创建目录
        timestamp: 时间戳（可选，默认使用当前时间）
        dataset_name: 数据集名称（可选，用于在子目录下创建数据集目录）
    
    Returns:
        实验输出目录路径
    
    Examples:
        >>> get_output_dir('single', 'exp', dataset_name='MSRC')
        './results/single/MSRC/exp_20260315_143052'
        
        >>> get_output_dir('grid_search')
        './results/grid_search/hp_search_20260315_143052'
    """
    if exp_type not in OUTPUT_SUBDIRS:
        raise ValueError(f"未知的实验类型: {exp_type}，可选: {list(OUTPUT_SUBDIRS.keys())}")
    
    # 获取子目录
    subdir = OUTPUT_SUBDIRS[exp_type]
    
    # 生成时间戳
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 确定目录前缀
    prefix = prefix or {
        'single': 'exp',
        'grid_search': 'hp_search',
        'targeted': 'targeted',
        'ablation': 'ablation',
        'baseline': 'baseline',
        'pretrain': 'pretrain',
    }.get(exp_type, 'exp')
    
    # 构建目录名
    dir_name = f"{prefix}_{timestamp}"
    
    # 完整路径
    if dataset_name:
        output_dir = os.path.join(BASE_OUTPUT_DIR, subdir, dataset_name, dir_name)
    else:
        output_dir = os.path.join(BASE_OUTPUT_DIR, subdir, dir_name)
    
    if create_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    return output_dir


def init_output_structure():
    """
    初始化输出目录结构
    创建所有必要的子目录
    """
    for subdir in OUTPUT_SUBDIRS.values():
        path = os.path.join(BASE_OUTPUT_DIR, subdir)
        os.makedirs(path, exist_ok=True)
        print(f"创建目录: {path}")
    
    print(f"输出目录结构初始化完成，根目录: {BASE_OUTPUT_DIR}")


def get_global_log_file() -> str:
    """
    获取总体实验日志文件路径
    
    Returns:
        总体实验日志文件路径
    """
    _ensure_log_dir()
    return GLOBAL_LOG_FILE


def get_global_results_file() -> str:
    """
    获取总体实验结果JSON文件路径
    
    Returns:
        总体实验结果JSON文件路径
    """
    _ensure_log_dir()
    return GLOBAL_RESULTS_FILE


def get_save_path(filename: str, output_dir: Optional[str]) -> str:
    """
    获取保存路径
    
    Args:
        filename: 文件名
        output_dir: 输出目录
    
    Returns:
        完整的保存路径
    """
    if output_dir is not None:
        return os.path.join(output_dir, filename)
    return filename


def generate_exp_dir_name(timestamp: str, hyperparams: Dict[str, Any], 
                         metrics: Dict[str, float]) -> str:
    """
    生成实验目录名，格式: 0325-2237-0.90-0.84-0.80-0.1-0.1-0.1-0.0001
    
    Args:
        timestamp: 时间戳，格式为 'YYYYMMDD_HHMMSS'
        hyperparams: 超参数字典
        metrics: 指标字典
    
    Returns:
        格式化的目录名
    
    Examples:
        >>> generate_exp_dir_name('20260325_221730', 
        ...                       {'lambda1': 0.1, 'lambda2': 0.1, 'lambda3': 0.1, 'beta': 0.0001},
        ...                       {'ACC': 0.9381, 'NMI': 0.8909, 'ARI': 0.8642})
        '0325-2217-0.94-0.89-0.86-0.1-0.1-0.1-0.0001'
    """
    # 解析时间戳，转换为 MMDD-HHMM 格式（去掉秒）
    dt = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
    time_part = dt.strftime('%m%d-%H%M')
    
    # 提取指标（保留2位小数）
    acc = metrics.get('ACC', 0.0)
    nmi = metrics.get('NMI', 0.0)
    ari = metrics.get('ARI', 0.0)
    metrics_part = f"{acc:.2f}-{nmi:.2f}-{ari:.2f}"
    
    # 提取核心参数
    lambda1 = hyperparams.get('lambda1', 0.1)
    lambda2 = hyperparams.get('lambda2', 0.1)
    lambda3 = hyperparams.get('lambda3', 0.1)
    beta = hyperparams.get('beta', 0.0001)
    params_part = f"{lambda1}-{lambda2}-{lambda3}-{beta}"
    
    # 组合最终目录名
    dir_name = f"{time_part}-{metrics_part}-{params_part}"
    
    return dir_name


def rename_output_dir(old_dir: str, new_dir_name: str) -> str:
    """
    重命名输出目录
    
    Args:
        old_dir: 原目录路径
        new_dir_name: 新目录名（不含路径）
    
    Returns:
        新目录的完整路径
    """
    # 获取父目录
    parent_dir = os.path.dirname(old_dir)
    new_dir = os.path.join(parent_dir, new_dir_name)
    
    # 如果新目录已存在，添加后缀
    if os.path.exists(new_dir):
        base_name = new_dir_name
        counter = 1
        while os.path.exists(new_dir):
            new_dir = os.path.join(parent_dir, f"{base_name}_{counter}")
            counter += 1
    
    # 重命名目录
    os.rename(old_dir, new_dir)
    print(f"目录已重命名: {old_dir} -> {new_dir}")
    
    return new_dir