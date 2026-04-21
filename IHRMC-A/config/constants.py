"""
常量定义模块

包含项目中使用的所有常量，便于统一管理和修改
"""

from typing import List

# ==============================================================================
# 模型默认参数
# ==============================================================================

# 默认核心超参数 [lambda1, lambda2, lambda3, beta]
DEFAULT_PARAMS: List[float] = [1.0, 0.0, 0.006, 1e-4]

# 默认锚点数
DEFAULT_M: int = 28

# 默认学习率
DEFAULT_LR: float = 2 ** (-10)  # ~0.0009765625

# 默认预训练学习率
DEFAULT_PRE_LR: float = 2 ** (-10)

# 默认训练轮数
DEFAULT_EPOCHS: int = 2

# 默认预训练轮数
DEFAULT_PRE_EPOCHS: int = 2

# 默认收敛容差
DEFAULT_TOL: float = 1e-7

# 默认最大迭代次数
DEFAULT_MAX_ITERS: int = 100

# 默认批次大小
DEFAULT_BATCH_SIZE: int = 32

# 默认隐藏层维度
DEFAULT_HIDDEN_DIMS: List[int] = [200, 100]

# 默认潜在表示维度
DEFAULT_LATENT_DIM: int = 100

# 默认ADMM惩罚参数最大值
DEFAULT_BETA_MAX: float = 10.0

# 默认早停最小变化量
DEFAULT_EARLY_STOP_MIN_DELTA: float = 1e-3

# 默认超图k近邻数
DEFAULT_K_NEIGHBORS: int = 5

# 默认L2正则化系数
DEFAULT_L2: float = 1e-3

# 默认SSC正则化参数
DEFAULT_SSC_GAMMA: float = 1e-1

# 默认SSC收敛容差
DEFAULT_SSC_TOL: float = 2e-5

# 默认同伦求解器最大迭代次数
DEFAULT_HOMOTOPY_MAX_ITER: int = 600

# 默认同伦求解器收敛容差
DEFAULT_HOMOTOPY_TOL: float = 1e-3


# ==============================================================================
# 数值常量
# ==============================================================================

# 极小值，用于避免除零
EPS: float = 1e-8

# 非零元素判断阈值
NONZERO_THRESHOLD: float = 1e-6

# SVD正则化项
SVD_REG: float = 1e-8


# ==============================================================================
# 数据集相关
# ==============================================================================

# 支持的数据集列表
SUPPORTED_DATASETS: List[str] = [
    "Simulated",
    "MSRC",
    "100Leaves",
    "BBC",
    "WebKB",
    "BBCSport",
    "Caltech101"
]

# 默认数据集
DEFAULT_DATASET: str = "MSRC"

# 数据集下载URL前缀
DATASET_URL_PREFIX: str = "https://raw.githubusercontent.com/ericyangyu/mvc_datasets/main"


# ==============================================================================
# 可视化相关
# ==============================================================================

# 默认图表DPI
DEFAULT_DPI: int = 150

# 默认图表大小
DEFAULT_FIG_SIZE: tuple = (10, 6)

# t-SNE最大样本数
TSNE_MAX_SAMPLES: int = 1000


# ==============================================================================
# 设备相关
# ==============================================================================

# 默认设备
DEFAULT_DEVICE: str = "cuda"

# 备用设备
FALLBACK_DEVICE: str = "cpu"


# ==============================================================================
# 随机种子
# ==============================================================================

# 默认随机种子
DEFAULT_RANDOM_SEED: int = 42
