"""
求解器模块

包含两个主要功能：
1. SSC (Sparse Subspace Clustering) - 稀疏子空间聚类
2. t-SVT (Tensor Singular Value Thresholding) - 张量奇异值阈值
"""

import warnings
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning

from solver.homotopy_solver import solve_homotopy

# 忽略收敛警告
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# ==============================================================================
# SSC (Sparse Subspace Clustering) 模块
# ==============================================================================
def _nonzero_ratio(matrix: np.ndarray) -> float:
    """
    计算矩阵的非零比例
    
    Args:
        matrix: 输入矩阵
    
    Returns:
        非零元素比例
    """
    # 使用相对阈值，基于矩阵的最大值
    threshold = np.max(np.abs(matrix)) * 1e-6 if np.max(np.abs(matrix)) > 0 else 1e-6
    total_elements = matrix.size  # 总元素数
    non_zero_elements = np.sum(np.abs(matrix) > threshold)
    return non_zero_elements / total_elements

def SSC_Basic(
    data: torch.Tensor,
    gamma: float = 1e-1,
    tol: float = 2e-5,
    non_negative: bool = False,
    anchor_indices: Optional[torch.Tensor] = None
    
) -> torch.Tensor:
    """
    基本的稀疏子空间聚类 (SSC) 实现。
    
    对每个数据点，使用其他数据点作为字典，计算稀疏编码，并对结果进行归一化。
    如果提供了锚点索引，则使用锚点作为字典，返回锚点表达矩阵。
    
    Args:
        data: 数据矩阵，形状 (N, d) torch tensor
        gamma: 正则化参数
        tol: 收敛 tolerance
        non_negative: 是否使用非负约束
        anchor_indices: 锚点索引，形状 (m,) torch tensor
    
    Returns:
        C: 归一化后的系数矩阵
            - 如果提供了锚点索引，形状 (m, N) torch tensor
            - 否则，形状 (N, N) torch tensor
    """
    # 准备数据
    with torch.no_grad():
        device = data.device
        N, d = data.shape
        # 转置数据
        data_np = data.detach().cpu().numpy().T  # (d, N)
        # 确保使用双精度浮点数
        data_np = data_np.astype(np.float64)
        
        # 检查是否使用锚点
        if anchor_indices is not None:
            # 使用锚点作为字典
            m = len(anchor_indices)
            # 初始化锚点表达矩阵 (m, N)
            C = np.zeros((m, N), dtype=np.float64)
            
            # 获取锚点数据
            anchor_indices_np = anchor_indices.detach().cpu().numpy()
            anchor_data = data_np[:, anchor_indices_np]  # (d, m)
            
            # 对每个数据点计算稀疏编码
            for i in range(N):
                y = data_np[:, i].flatten()  # (d,)
                
                # 检查当前样本是否是锚点
                is_anchor = np.isin(i, anchor_indices_np)
                
                # 构建字典：如果是锚点，排除自己；否则使用所有锚点
                if is_anchor:
                    # 找到当前锚点在锚点列表中的索引
                    anchor_idx = np.where(anchor_indices_np == i)[0][0]
                    # 构建排除自己的锚点数据
                    mask = np.ones(m, dtype=bool)
                    mask[anchor_idx] = False
                    dict_data = anchor_data[:, mask]
                    # 求解 L1 问题
                    try:
                        coef_partial = solve_homotopy(dict_data, y, gamma, max_iter=600, tol=tol, non_negative=non_negative)
                        # 构建完整的系数向量，将自己的系数设为0
                        coef = np.zeros(m, dtype=np.float64)
                        coef[mask] = coef_partial
                    except Exception as e:
                        # 静默处理错误，直接设置为零向量
                        coef = np.zeros(m, dtype=np.float64)
                else:
                    # 非锚点，使用所有锚点作为字典
                    try:
                        coef = solve_homotopy(anchor_data, y, gamma, max_iter=600, tol=tol, non_negative=non_negative)
                    except Exception as e:
                        # 静默处理错误，直接设置为零向量
                        coef = np.zeros(m, dtype=np.float64)
                
                if np.sum(np.abs(coef) < 1e-10) == len(coef):
                    coef = np.zeros(m, dtype=np.float64)
                
                # 填充系数矩阵
                C[:, i] = coef
            
            # 对系数矩阵进行归一化
            col_norms = np.sqrt(np.sum(C ** 2, axis=0, keepdims=True))
            # 避免除以零
            col_norms[col_norms == 0] = 1
            C_normalized = C / col_norms
            
            # 当锚点数量等于样本数量时，确保矩阵对称（与传统自表达保持一致）
            if m == N:
                C_normalized = (C_normalized + C_normalized.T) / 2
                print(f"[锚点SSC (退化为自表达)] 完成稀疏编码，非零比例={_nonzero_ratio(C_normalized):.4f}")
            else:
                print(f"[锚点SSC] 完成稀疏编码，非零比例={_nonzero_ratio(C_normalized):.4f}")
        else:
            # 传统SSC，使用所有样本作为字典
            # 初始化系数矩阵
            C = np.zeros((N, N), dtype=np.float64)
            
            # 对每个数据点计算稀疏编码
            for i in range(N):
                # 构建字典（排除当前数据点）
                if i == 0:
                    # 第一个样本，字典是剩余所有样本
                    X = data_np[:, 1:].copy()  # (d, N-1)
                else:
                    # 其他样本，字典是前后所有样本
                    X = np.hstack([data_np[:, :i], data_np[:, i+1:]])  # (d, N-1)
                
                y = data_np[:, i].flatten()  # (d,)
                
                # 求解 L1 问题
                try:
                    coef = solve_homotopy(X, y, gamma, max_iter=600, tol=tol, non_negative=non_negative)
                except Exception as e:
                    # 静默处理错误，直接设置为零向量
                    coef = np.zeros(X.shape[1], dtype=np.float64)
                
                if np.sum(np.abs(coef) < 1e-10) == len(coef):
                    coef = np.zeros(len(coef), dtype=np.float64)
                
                # 填充系数矩阵（对角线为0）
                if i == 0:
                    C[i, 1:] = coef
                else:
                    C[i, :i] = coef[:i]
                    C[i, i+1:] = coef[i:]
            
            # 对系数矩阵进行归一化
            col_norms = np.sqrt(np.sum(C ** 2, axis=0, keepdims=True))
            # 避免除以零
            col_norms[col_norms == 0] = 1
            C_normalized = C / col_norms
            
            # 确保矩阵对称
            C_normalized = (C_normalized + C_normalized.T) / 2
            
            # 计算非零比例
            non_zero_ratio = _nonzero_ratio(C_normalized)
            print(f"[基本SSC] 完成稀疏编码，非零比例={non_zero_ratio:.4f}")
        
    # 转换回float32输出
    return torch.tensor(C_normalized, dtype=torch.float32, device=device)

# ==============================================================================
# t-SVT (Tensor Singular Value Thresholding) 模块
# ==============================================================================

def t_svt(
    T: torch.Tensor,
    tau: float,
    mode: int = 3,
    is_weight: bool = False,
    use_mixed_precision: bool = True,
    svd_driver: str = 'gesvd'
) -> Tuple[torch.Tensor, float]:
    """
    张量奇异值阈值 (Tensor Singular Value Thresholding, t-SVT)
    
    基于 t-SVD 的张量低秩逼近算法。通过对张量的每个 frontal 切片进行 FFT 变换后
    在频域进行 SVD 分解，然后对奇异值进行软阈值操作，最后通过逆 FFT 恢复张量。
    
    数学原理：
    min_X ||X||_* + 1/(2*tau) * ||X - T||_F^2
    
    其中 ||X||_* 是张量核范数，定义为所有 frontal 切片奇异值之和。
    
    复杂度分析：
    - FFT/IFFT: O(n1*n2*n3*log(n3))
    - SVD分解: O(n3*min(n1,n2)*(n1+n2))
    - 总时间复杂度: O(n1*n2*n3*log(n3) + n3*min(n1,n2)*(n1+n2))
    
    Args:
        T: 输入三阶张量，形状 (n1, n2, n3)
        tau: 软阈值参数，控制奇异值收缩程度
        mode: 张量模式，控制维度变换方式
              - mode=1: 变换 (N, V, m) -> (m, N, V)
              - mode=3: 变换 (m, N, V) -> (m, V, N)
        is_weight: 是否使用加权阈值
        use_mixed_precision: 是否使用混合精度（float32）加速计算
        svd_driver: SVD驱动选择，'gesvd'（默认，更稳定）或 'gesdd'（更快但内存消耗大）
    
    Returns:
        X: 阈值后的张量，形状与输入相同
        objV: 张量核范数值（所有保留奇异值之和）
    
    Raises:
        ValueError: 如果输入不是三阶张量
    """
    if len(T.shape) != 3:
        raise ValueError("输入张量必须是三阶张量")
    
    device = T.device
    tau = float(tau)
    original_dtype = T.dtype
    
    # 根据 mode 调整维度
    if mode == 1:
        Y = torch.transpose(T, 0, 2).permute(1, 0, 2)
    elif mode == 3:
        Y = torch.transpose(T, 1, 2)
    else:
        Y = T
    
    # FFT - 使用混合精度加速
    if use_mixed_precision and Y.dtype == torch.float64:
        Y = Y.float()
    
    Y_fft = torch.fft.fft(Y, dim=2)
    n1, n2, n3 = Y_fft.shape
    
    # 计算权重
    C = None
    if is_weight:
        C = torch.sqrt(torch.tensor(n3 * n2, device=device, dtype=Y_fft.dtype))
    
    # 正频率切片数量（包括直流和奈奎斯特频率）
    end_value = (n3 + 1) // 2
    
    # 批量提取正频率切片，形状 (end_value, n1, n2)
    batch = Y_fft[:, :, :end_value].permute(2, 0, 1).contiguous()
    
    # 添加小的正则化项（批量）
    eps = 1e-8
    reg = eps * torch.eye(n1, n2, device=device).unsqueeze(0).repeat(end_value, 1, 1)
    batch_reg = batch + reg.to(dtype=batch.dtype)
    
    # 批量 SVD（复数输入）
    try:
        if device == 'cuda':
            U, S, Vh = torch.linalg.svd(batch_reg, full_matrices=False, driver=svd_driver)
        else:
            U, S, Vh = torch.linalg.svd(batch_reg, full_matrices=False)
    except Exception:
        # 如果SVD失败，使用更稳定的方法
        U, S, Vh = torch.linalg.svd(batch_reg, full_matrices=False)
    
    # 批量软阈值
    if is_weight:
        weight = C / (S + 1e-8)
        tau_weight = tau * weight
        S_thresh = torch.clamp(S - tau_weight, min=0)
    else:
        S_thresh = torch.clamp(S - tau, min=0)
    
    # 将 S_thresh 转换为复数对角矩阵（与 U 同类型）
    diag_S = torch.diag_embed(S_thresh).to(dtype=U.dtype)
    batch_reconstructed = U @ diag_S @ Vh
    
    # 将重建结果放回 Y_fft 的正频率位置
    Y_fft[:, :, :end_value] = batch_reconstructed.permute(1, 2, 0)
    
    # 目标函数值（张量核范数）- 正频率部分
    objV = torch.sum(S_thresh).item()
    
    # 处理共轭对称部分：负频率切片由正频率切片的共轭得到
    if n3 > 1:
        # 构建负频率索引（排除直流和奈奎斯特频率）
        # 对于 n3 个点，正频率索引是 0, 1, ..., end_value-1
        # 负频率索引是 n3-1, n3-2, ..., end_value（如果 n3 是奇数）或 end_value（如果 n3 是偶数）
        for i in range(1, end_value):
            neg_idx = n3 - i
            if neg_idx < n3 and neg_idx != i:
                # batch_reconstructed 形状是 (end_value, n1, n2)
                # 需要转置为 (n1, n2) 然后赋值
                Y_fft[:, :, neg_idx] = batch_reconstructed[i].conj()
                objV += torch.sum(S_thresh[i]).item()
    
    # 逆 FFT 并取实部
    Y_ifft = torch.fft.ifft(Y_fft, dim=2).real
    
    # 恢复原始精度
    if use_mixed_precision and original_dtype == torch.float64:
        Y_ifft = Y_ifft.double()
    
    # 恢复原始维度
    if mode == 1:
        X = Y_ifft.permute(1, 0, 2)                 # (N, V, m) → (V, N, m)
        X = torch.transpose(X, 0, 2)                # (V, N, m) → (m, N, V)
    elif mode == 3:
        X = torch.transpose(Y_ifft, 1, 2)
    else:
        X = Y_ifft
    
    return X, objV

def TNN(
    T: torch.Tensor
    ) -> torch.Tensor:
    """
    计算张量核范数 (Tensor Nuclear Norm, TNN)
    
    张量核范数是通过 t-SVD（张量奇异值分解）计算的，它是张量奇异值的和。
    
    复杂度分析：
    - FFT：O(n1*n2*n3*log(n3))
    - SVD分解（n3次）：O(n3*min(n1,n2)*(n1+n2))
    - 奇异值求和：O(n3*min(n1,n2))
    - 总时间复杂度：O(n1*n2*n3*log(n3) + n3*min(n1,n2)*(n1+n2))
    - 空间复杂度：O(n1*n2*n3) 用于存储张量和中间结果

    Args:
        T: 输入三阶张量，形状 (n1, n2, n3)

    Returns:
        norm: 张量核范数
    """
    # 沿第三维进行离散傅里叶变换 (FFT)，得到频域张量
    T_fft = torch.fft.fft(T, dim=2)  # shape (n1, n2, n3)
    
    # 计算所有频率切片的奇异值之和
    total_norm = 0.0
    n3 = T_fft.shape[2]
    
    for k in range(n3):
        # 对每个频率切片进行SVD
        slice_ = T_fft[:, :, k]
        # 计算奇异值
        S = torch.linalg.svdvals(slice_)
        # 奇异值求和并累加到总范数
        total_norm += torch.sum(S).real.item()
    
    return torch.tensor(total_norm, dtype=T.dtype, device=T.device)

def proj_simplex_batch(X: torch.Tensor) -> torch.Tensor:
    """
    批量投影到概率单纯形 (Probability Simplex)
    
    将矩阵的每一列投影到概率单纯形上，即满足：
    - 所有元素非负: x_i >= 0
    - 元素之和为 1: sum(x_i) = 1
    
    算法基于论文 "Efficient Projections onto the l1-Ball for Learning in High Dimensions"，
    使用排序和累积和的方法在 O(m log m) 时间内完成投影。
    
    数学原理：
    对于向量 x，投影到单纯形的问题为：
    min ||x - v||_2^2  s.t.  v >= 0, sum(v) = 1
    
    解析解为：v = max(x - theta, 0)
    其中 theta 是使得 sum(max(x - theta, 0)) = 1 的阈值。
    
    Args:
        X: 输入矩阵，形状 (m, N)，每列需要投影到单纯形
    
    Returns:
        X_proj: 投影后的矩阵，形状 (m, N)，每列在概率单纯形上
    
    Example:
        >>> X = torch.randn(5, 3)
        >>> X_proj = proj_simplex_batch(X)
        >>> print(X_proj.sum(dim=0))  # 每列之和为 1
        >>> print((X_proj >= 0).all())  # 所有元素非负
    """
    m, N = X.shape
    # 对每列降序排序
    sorted_X, _ = torch.sort(X, dim=0, descending=True)  # (m, N)
    # 计算累积和减去1
    cumsum = torch.cumsum(sorted_X, dim=0) - 1  # (m, N)
    # 构造条件矩阵：sorted_X - cumsum / k > 0
    k = torch.arange(1, m+1, device=X.device).unsqueeze(1)  # (m, 1)
    cond = sorted_X - cumsum / k > 0  # (m, N)
    # 找到满足条件的最大行索引（每列独立）
    # 方法：从下往上找第一个True，等价于取最后一个True的位置
    # 注意：rho 是 1-based 索引
    rho = m - (cond.flip(dims=[0]).int().argmax(dim=0))  # (N,)
    # 对于全部为False的列，rho应为0（无有效投影，实际不应出现）
    # 这里简单处理：如果全部False，设置rho=1，此时theta=cumsum[0]，投影结果就是clamp
    rho = torch.where(cond.any(dim=0), rho, torch.ones_like(rho))
    # 转换为 0-based 索引
    rho_0based = rho - 1
    # 提取对应行的cumsum
    theta = torch.gather(cumsum, 0, rho_0based.unsqueeze(0)) / rho  # (1, N)
    theta = theta.squeeze(0)  # (N,)
    # 投影并截断
    X_proj = X - theta.unsqueeze(0)
    X_proj = torch.clamp(X_proj, min=0)
    return X_proj