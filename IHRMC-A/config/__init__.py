"""
配置管理模块

包含以下功能：
1. Config 类 - 配置管理类，支持配置文件加载和命令行参数覆盖
2. 命令行参数解析 - 解析命令行参数
3. 配置文件加载 - 加载和管理配置文件
4. 数据集配置 - 获取特定数据集的配置
5. 常量定义 - 项目中的所有常量
"""

from .config import Config, load_config, parse_cli_args
from .constants import (
    DEFAULT_PARAMS,
    DEFAULT_M,
    DEFAULT_LR,
    DEFAULT_EPOCHS,
    DEFAULT_MAX_ITERS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_HIDDEN_DIMS,
    DEFAULT_LATENT_DIM,
    DEFAULT_TOL,
    DEFAULT_BETA_MAX,
    EPS,
    SUPPORTED_DATASETS,
    DEFAULT_DATASET,
    DEFAULT_DEVICE,
    DEFAULT_RANDOM_SEED,
)

__all__ = [
    'Config',
    'load_config',
    'parse_cli_args',
    'DEFAULT_PARAMS',
    'DEFAULT_M',
    'DEFAULT_LR',
    'DEFAULT_EPOCHS',
    'DEFAULT_MAX_ITERS',
    'DEFAULT_BATCH_SIZE',
    'DEFAULT_HIDDEN_DIMS',
    'DEFAULT_LATENT_DIM',
    'DEFAULT_TOL',
    'DEFAULT_BETA_MAX',
    'EPS',
    'SUPPORTED_DATASETS',
    'DEFAULT_DATASET',
    'DEFAULT_DEVICE',
    'DEFAULT_RANDOM_SEED',
]
