"""
工具函数模块

包含以下功能：
1. 聚类评估指标计算
2. 标签重排（基于匈牙利算法）
3. 随机种子设置
"""

import random
from typing import Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
)


# ==============================================================================
# 标签处理模块
# ==============================================================================

def best_map(L1, L2):
    """
    基于匈牙利算法重排标签，使 L2 与 L1 匹配最佳，
    并确保重排后的标签索引与真实标签索引一一对应
    
    Args:
        L1: 真实标签，torch张量
        L2: 预测标签，torch张量
    
    Returns:
        重排后的预测标签，torch张量，索引与真实标签一致
    """
    import torch
    
    # 确保输入是张量
    # 转换为张量
    if isinstance(L1, np.ndarray):
        L1 = torch.from_numpy(L1)
    if isinstance(L2, np.ndarray):
        L2 = torch.from_numpy(L2)
    
    if not hasattr(L1, 'device'):
        L1 = torch.tensor(L1)
    if not hasattr(L2, 'device'):
        L2 = torch.tensor(L2, device=L1.device)
    
    device = L1.device
    
    # 确保标签是一维张量
    L1 = L1.flatten()
    L2 = L2.flatten()
    
    # 获取真实标签的唯一类别
    unique_L1, _ = torch.sort(torch.unique(L1))
    n = unique_L1.numel()
    
    # 确保真实标签是连续的0-n-1索引
    # 如果不是，先映射到连续索引
    if not torch.all(unique_L1 == torch.arange(n, device=device)):
        # 创建映射字典：旧标签 -> 新标签
        label_map = {old.item(): new for new, old in enumerate(unique_L1)}
        # 映射L1到连续索引
        L1_mapped = torch.zeros_like(L1)
        for old, new in label_map.items():
            L1_mapped[L1 == old] = new
    else:
        L1_mapped = L1
    
    # 构建成本矩阵（使用负的匹配计数，因为 linear_sum_assignment 寻找最小值）
    cost_matrix = torch.zeros((n, n), dtype=torch.int64, device=device)
    for i in range(n):
        for j in range(n):
            cost_matrix[i, j] = -torch.sum((L1_mapped == i) & (L2 == j))
    
    # 使用匈牙利算法求解最佳匹配（需要numpy）
    row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
    
    # 创建映射字典：预测标签 -> 真实标签索引
    mapping = {col_ind[i]: row_ind[i] for i in range(n)}
    
    # 根据匹配结果重排标签
    if isinstance(L2, torch.Tensor):
        new_L2 = torch.zeros_like(L2)
        for pred_label in torch.unique(L2):
            pred_label_item = pred_label.item()
            if pred_label_item in mapping:
                new_L2[L2 == pred_label] = mapping[pred_label_item]
            else:
                # 处理未匹配的标签（如果有）
                new_L2[L2 == pred_label] = pred_label
    else:
        # 如果 L2 是 numpy 数组
        new_L2 = np.zeros_like(L2)
        for pred_label in np.unique(L2):
            if pred_label in mapping:
                new_L2[L2 == pred_label] = mapping[pred_label]
            else:
                # 处理未匹配的标签（如果有）
                new_L2[L2 == pred_label] = pred_label
    
    return new_L2


# ==============================================================================
# 评估指标模块
# ==============================================================================

def clustering_metrics(y_true, y_pred):
    """
    计算多种聚类评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
    
    Returns:
        dict: 包含监督评估指标的字典
    """
    # 确保标签是numpy数组并展平
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # 计算监督指标
    return {
        'ACC': float(accuracy_score(y_true, y_pred)),
        'NMI': float(normalized_mutual_info_score(y_true, y_pred)),
        'ARI': float(adjusted_rand_score(y_true, y_pred)),
        'Precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'Recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'F-score': float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    }


# ==============================================================================
# 矩阵分析模块
# ==============================================================================

def block_diag_ratio(matrix, y_true):
    """
    计算矩阵的块对角比（同一类内元素和 / 总元素和）
    
    Args:
        matrix: torch.Tensor, shape (N, N)，输入矩阵
        y_true: torch.Tensor, shape (N,)，真实标签
    
    Returns:
        float: 块对角比
    """
    # 使用 PyTorch 操作
    matrix_abs = torch.abs(matrix)
    total = matrix_abs.sum()
    
    if total == 0:
        return 0.0
    
    # 创建标签矩阵，其中相同标签的位置为True
    label_matrix = y_true.unsqueeze(1) == y_true
    # 计算类内元素和
    within = torch.sum(matrix_abs * label_matrix)
    
    return (within / total).item()

def to_numpy(data):
    """
    将张量转换为numpy数组
    
    Args:
        data: torch.Tensor 或 numpy array
    
    Returns:
        numpy array
    """
    if torch.is_tensor(data):
        return data.detach().cpu().numpy() if data.requires_grad else data.cpu().numpy()
    return data

def print_data_info(X, y_true):
    """
    打印多视图数据集的基本信息和统计
    Args:
        X: torch.Tensor, shape (N, D, V)，多视图数据
        y_true: torch.Tensor，真实标签
    """
    # 打印基本信息
    y_true_np = to_numpy(y_true)
    N, D, V = X.shape
    print(f"视图数: {V}, 样本数: {N}, 类别数: {len(np.unique(y_true_np))}")
    print("数据统计：")
    
    # 打印每个视图的统计信息
    for i in range(V):
        X_view = X[:, :, i]
        X_np = to_numpy(X_view)
        print(f"视图 {i}: shape={X_np.shape}, mean={X_np.mean():.3f}, std={X_np.std():.3f}, min={X_np.min():.3f}, max={X_np.max():.3f}")
    
    # 打印标签信息
    print(f"标签: 类别数={len(np.unique(y_true_np))}, 分布={np.bincount(y_true_np)}")

def print_latent_info(H, y_true, prefix=""):
    """
    打印潜在表示 H 的统计信息（类内距离、类间距离等）
    
    Args:
        H: torch.Tensor, shape (N, d)，潜在表示
        y_true: torch.Tensor, shape (N,)，真实标签
        prefix: str, 打印信息的前缀
    """
    # 转换为numpy数组
    H_np = to_numpy(H)
    y = to_numpy(y_true)
    
    unique_labels = np.unique(y)
    intra_dists = []
    inter_dists = []
    
    # 计算类内和类间距离
    for lab in unique_labels:
        mask = (y == lab)
        class_samples = H_np[mask]
        
        # 计算类内距离
        if len(class_samples) > 1:
            intra = np.mean(np.linalg.norm(class_samples[:, None] - class_samples[None, :], axis=2))
            intra_dists.append(intra)
        
        # 计算类间距离
        other_samples = H_np[~mask]
        if len(other_samples) > 0 and len(class_samples) > 0:
            inter = np.mean(np.linalg.norm(class_samples[:, None] - other_samples[None, :], axis=2))
            inter_dists.append(inter)
    
    # 打印统计信息
    if intra_dists:
        avg_intra = np.mean(intra_dists)
        print(f"{prefix}平均类内距离: {avg_intra:.4f}")
    if inter_dists:
        avg_inter = np.mean(inter_dists)
        print(f"{prefix}平均类间距离: {avg_inter:.4f}")
        if intra_dists:
            print(f"{prefix}类间/类内比值: {avg_inter/avg_intra:.4f}")


# ==============================================================================
# 随机种子模块
# ==============================================================================

def set_seed(seed: Optional[int] = 42) -> None:
    """
    设置所有随机种子，确保实验可复现
    
    Args:
        seed: 随机种子值，为 None 或字符串 'None' 时不设置种子
    """
    # 处理字符串 'None' 的情况
    if seed == 'None':
        seed = None
    
    if seed is not None:
        # 确保 seed 是整数
        seed = int(seed)
        # 设置PyTorch种子
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # 设置NumPy和Python随机种子
        np.random.seed(seed)
        random.seed(seed)
        
        # 确保CUDA确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        print(f"随机种子已设置为: {seed}")
    else:
        print("未设置随机种子，实验结果可能不可复现")


def get_device() -> str:
    """
    获取可用的计算设备
    
    Returns:
        str: 设备名称 ('cuda' 或 'cpu')
    """
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


# ==============================================================================
# 锚点选择模块
# ==============================================================================

def select_anchors(
    H: torch.Tensor,
    m: int,
    anchor_selection: str = 'kmeans',
    random_seed: Optional[int] = None,
    device: str = 'cuda',
    mask: Optional[torch.Tensor] = None,
    masks: Optional[list] = None,
    X: Optional[list] = None
) -> torch.Tensor:
    """
    选择锚点，支持完整样本选择和跨视图互补锚点机制
    
    Args:
        H: 输入数据，形状 (N, feature_dim)
        m: 锚点数
        anchor_selection: 锚点选择方法 ('kmeans' 或 'random')
        random_seed: 随机种子
        device: 计算设备
        mask: 单个视图掩码，形状 (N, feature_dim)，1表示存在，0表示缺失
        masks: 掩码列表，每个元素是形状为 (N, D_v) 的张量
        X: 多视图数据列表，用于跨视图互补锚点选择
    
    Returns:
        M: 锚点索引，形状 (m,)
    """
    from sklearn.cluster import KMeans
    
    # 自动推断参数
    N = H.shape[0]
    V = len(X) if X is not None else None
    
    with torch.no_grad():
        # 辅助函数：从指定样本中选择锚点
        def _select_from_samples(sample_indices, data):
            if len(sample_indices) >= m:
                if anchor_selection == 'kmeans':
                    H_subset = data[sample_indices]
                    kmeans = KMeans(n_clusters=m, init='k-means++', random_state=random_seed)
                    kmeans.fit(H_subset.detach().cpu().numpy())
                    centers = torch.tensor(kmeans.cluster_centers_, device=device)
                    distances = torch.cdist(centers, H_subset)
                    M_subset = torch.argmin(distances, dim=1)
                    return sample_indices[M_subset]
                else:
                    if random_seed is not None:
                        g = torch.Generator(device='cpu').manual_seed(random_seed)
                        perm = torch.randperm(len(sample_indices), generator=g)
                    else:
                        perm = torch.randperm(len(sample_indices), device=device)
                    return sample_indices[perm[:m]]
            return sample_indices
        
        # 辅助函数：随机选择补充锚点
        def _random_select(sample_indices, count, seed_offset=0):
            if random_seed is not None:
                g = torch.Generator(device='cpu').manual_seed(random_seed + seed_offset)
                perm = torch.randperm(len(sample_indices), generator=g)
            else:
                perm = torch.randperm(len(sample_indices), device=device)
            return sample_indices[perm[:count]]
        
        if masks is not None and X is not None and V is not None and N is not None:
            # 跨视图互补锚点机制
            print("使用跨视图互补锚点机制")
            
            # 步骤1：识别完整样本
            view_availability = [masks[v].sum(dim=1) > 0 for v in range(V)]
            complete_mask = view_availability[0]
            for v in range(1, V):
                complete_mask = complete_mask & view_availability[v]
            complete_samples = torch.where(complete_mask)[0]
            print(f"完整样本数: {len(complete_samples)}")
            
            # 步骤2：从完整样本中选择基础锚点
            base_anchors = torch.tensor([], device=device)
            if len(complete_samples) > 0:
                base_anchors = _select_from_samples(complete_samples, H)
                if len(complete_samples) >= m:
                    print(f"从完整样本中使用{anchor_selection}选择了 {len(base_anchors)} 个锚点")
                else:
                    print(f"完整样本数不足，使用所有 {len(base_anchors)} 个完整样本作为基础锚点")
            
            # 步骤3：补充额外锚点
            additional_anchor_count = m - len(base_anchors)
            additional_anchors = []
            
            if additional_anchor_count > 0:
                print(f"需要从各视图特有的完整样本中选择 {additional_anchor_count} 个额外锚点")
                per_view_count = additional_anchor_count // V
                remainder = additional_anchor_count % V
                
                for v in range(V):
                    view_complete = view_availability[v]
                    other_views_incomplete = ~complete_mask
                    unique_to_view = view_complete & other_views_incomplete
                    unique_samples = torch.where(unique_to_view)[0]
                    
                    view_anchor_count = per_view_count + (1 if v < remainder else 0)
                    if view_anchor_count <= 0:
                        continue
                    
                    # 选择源样本
                    source_samples = None
                    if len(unique_samples) >= view_anchor_count:
                        source_samples = unique_samples
                    else:
                        all_available = torch.where(view_complete)[0]
                        all_available = all_available[~torch.isin(all_available, base_anchors)]
                        if len(all_available) >= view_anchor_count:
                            source_samples = all_available
                    
                    if source_samples is not None:
                        if anchor_selection == 'kmeans':
                            print(f"视图 {v}: 使用k-means++选择额外锚点")
                            H_subset = X[v][source_samples]
                            kmeans = KMeans(n_clusters=view_anchor_count, init='k-means++', random_state=random_seed)
                            kmeans.fit(H_subset.detach().cpu().numpy())
                            centers = torch.tensor(kmeans.cluster_centers_, device=device)
                            distances = torch.cdist(centers, H_subset)
                            M_subset = torch.argmin(distances, dim=1)
                            view_anchors = source_samples[M_subset]
                        else:
                            view_anchors = _random_select(source_samples, view_anchor_count, seed_offset=v)
                        additional_anchors.extend(view_anchors.tolist())
                        print(f"从视图 {v} 选择了 {len(view_anchors)} 个额外锚点")
            
            # 步骤4：合并和处理最终锚点
            if len(additional_anchors) > 0:
                all_anchors = torch.cat([base_anchors, torch.tensor(additional_anchors, device=device)])
            else:
                all_anchors = base_anchors
            
            all_anchors = torch.unique(all_anchors)
            
            if len(all_anchors) > m:
                all_anchors = _random_select(all_anchors, m, seed_offset=0)
            elif len(all_anchors) < m:
                all_samples = torch.arange(N, device=device)
                remaining_samples = all_samples[~torch.isin(all_samples, all_anchors)]
                if len(remaining_samples) > 0:
                    supplement_count = m - len(all_anchors)
                    supplement_anchors = _random_select(remaining_samples, supplement_count, seed_offset=100)
                    all_anchors = torch.cat([all_anchors, supplement_anchors])
            
            M = torch.sort(all_anchors)[0].long()
            print(f"最终选择了 {len(M)} 个锚点")
            return M
        else:
            # 传统锚点选择方法
            print("没有掩码，使用传统方法选择锚点")
            
            # 确定可用样本
            if mask is not None:
                non_missing_mask = mask.sum(dim=1) > 0
                non_missing_indices = torch.where(non_missing_mask)[0]
                if len(non_missing_indices) < m:
                    print(f"警告：非缺失样本数 ({len(non_missing_indices)}) 少于锚点数 ({m})，使用所有样本")
                    non_missing_indices = torch.arange(H.shape[0], device=device)
                H_non_missing = H[non_missing_indices]
            else:
                print("没有掩码，使用所有样本")
                non_missing_indices = torch.arange(H.shape[0], device=device)
                H_non_missing = H
            
            # 选择锚点
            M = _select_from_samples(non_missing_indices, H_non_missing)
            print(f"使用{anchor_selection}选择锚点")
            
            M = torch.sort(M)[0].long()
            print(f"最终选择了 {len(M)} 个锚点")
            return M