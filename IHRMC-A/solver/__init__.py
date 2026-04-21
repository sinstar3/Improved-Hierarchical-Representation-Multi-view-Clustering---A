"""
求解器模块

包含两个主要功能：
1. SSC (Sparse Subspace Clustering) - 稀疏子空间聚类
2. t-SVT (Tensor Singular Value Thresholding) - 张量奇异值阈值
3. 同伦求解器 - 用于稀疏编码
"""

from .solve import SSC_Basic, t_svt, TNN, proj_simplex_batch
from .homotopy_solver import solve_homotopy

__all__ = [
    'SSC_Basic',
    't_svt',
    'TNN',
    'proj_simplex_batch',
    'solve_homotopy'
]
