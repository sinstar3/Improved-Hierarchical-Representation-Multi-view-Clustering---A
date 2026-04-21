"""
数据加载模块

包含以下功能：
1. 模拟数据生成 - 生成论文 5.2.1 节描述的模拟多视图数据集
2. MATLAB 数据集加载 - 加载各种多视图聚类数据集
3. 数据集下载 - 自动下载数据集到本地
4. 数据预处理 - 数据标准化和标签处理

支持的数据集：
- MSRC - 微软研究剑桥数据集
- 100Leaves - UCI 100 叶子数据集
- BBCnews - BBC 新闻数据集
- WebKB - WebKB2 数据集
- Simulated - 模拟数据集
"""

import os
import urllib

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

class DataLoader:
    """
    数据加载器类，用于加载和预处理多视图数据集
    """
    
    # 数据集配置 - (加载函数, 参数字典)
    DATASET_CONFIGS = {
        "Simulated": ("_load_simulated_raw", {}),
        "MSRC": ("_load_matlab_raw", {"transpose_views": False}),
        "100leaves": ("_load_matlab_raw", {"transpose_views": False}),
        "BBC": ("_load_matlab_raw", {"transpose_views": False}),
        "WebKB": ("_load_matlab_raw", {"transpose_views": False}),
        "BBCSport": ("_load_matlab_raw", {"transpose_views": True}),
        "Caltech101": ("_load_matlab_raw", {"transpose_views": False})
    }
    
    def __init__(self, data_dir=None, dataset_name=None, normalize_method=None):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录，默认为None（使用默认目录）
            dataset_name: 数据集名称
            normalize_method: 归一化方法 ('tfidf', 'l2', 'standard', 'minmax', None)
        """
        self.data_dir = data_dir if data_dir else self._get_default_data_dir()
        self.dataset_name = dataset_name
        self.normalize_method = normalize_method
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
    
    def _download_file(self, url, dest_path):
        """下载文件到指定路径"""
        if not os.path.exists(dest_path):
            print(f"下载 {url} ...")
            urllib.request.urlretrieve(url, dest_path)
            print("下载完成")
        else:
            print(f"文件已存在: {dest_path}")
    
    def _get_default_data_dir(self):
        """
        获取默认数据目录
        
        Returns:
            默认数据目录路径
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, '..', 'data')
    
    def _preprocess_data(self, X, masks=None):
        """
        对数据进行预处理
        
        Args:
            X: 单个视图张量 (N, D) 或视图列表
            masks: 掩码列表，1表示存在，0表示缺失
        
        Returns:
            预处理后的单个视图张量或视图列表
            预处理后的掩码列表（如果提供）
        """
        if isinstance(X, list):
            # 处理视图列表
            processed_X = []
            processed_masks = []
            for i, view in enumerate(X):
                mask = masks[i] if masks else None
                print(f"处理视图 {i}，形状: {view.shape}")
                processed_view, processed_mask = self._preprocess_single_view(view, i, mask)
                processed_X.append(processed_view)
                processed_masks.append(processed_mask)
            if masks:
                return processed_X, processed_masks
            else:
                return processed_X
        else:
            # 处理单个视图
            mask = masks[0] if masks else None
            processed_view, processed_mask = self._preprocess_single_view(X, 0, mask)
            if masks:
                return processed_view, [processed_mask]
            else:
                return processed_view
    
    def _preprocess_single_view(self, X_view, view_idx, mask=None):
        """
        对单个视图进行预处理
        
        Args:
            X_view: 单个视图张量 (N, D)
            view_idx: 视图索引
            mask: 掩码张量，1表示存在，0表示缺失
        
        Returns:
            预处理后的单个视图张量
            预处理后的掩码张量
        """
        # 处理NaN值
        if torch.isnan(X_view).any():
            print(f"视图 {view_idx} 包含NaN值，正在处理...")
            # 用0替换NaN值
            X_view = torch.nan_to_num(X_view, nan=0.0)
        
        if self.normalize_method is not None and self.normalize_method != "None":
            if self.normalize_method == 'l2':
                print(f"视图 {view_idx} 使用L2归一化")
                if mask is not None:
                    # 只对存在的数据进行L2归一化
                    norm = torch.norm(X_view * mask, dim=1, keepdim=True)
                else:
                    norm = torch.norm(X_view, dim=1, keepdim=True)
                # 避免除以零
                norm = torch.clamp(norm, min=1e-12)
                X_view = X_view / norm
            elif self.normalize_method == 'standard':
                print(f"视图 {view_idx} 使用StandardScaler进行标准化")
                # 直接在张量上进行标准化
                if mask is not None:
                    # 只对存在的数据计算均值和标准差
                    valid_data = X_view * mask
                    # 计算非零元素的数量
                    count = mask.sum(dim=0, keepdim=True)
                    count = torch.clamp(count, min=1)
                    # 计算均值
                    mean = valid_data.sum(dim=0, keepdim=True) / count
                    # 计算标准差
                    std = torch.sqrt(((valid_data - mean) ** 2).sum(dim=0, keepdim=True) / count) + 1e-8
                else:
                    mean = X_view.mean(dim=0, keepdim=True)
                    std = X_view.std(dim=0, keepdim=True) + 1e-8  # 避免除以零
                X_view = (X_view - mean) / std
            elif self.normalize_method == 'minmax':
                print(f"视图 {view_idx} 使用MinMaxScaler进行归一化")
                # 直接在张量上进行最小-最大归一化
                if mask is not None:
                    # 只对存在的数据计算最小值和最大值
                    valid_data = X_view * mask
                    min_val = valid_data.min(dim=0, keepdim=True)[0]
                    max_val = valid_data.max(dim=0, keepdim=True)[0]
                else:
                    min_val = X_view.min(dim=0, keepdim=True)[0]
                    max_val = X_view.max(dim=0, keepdim=True)[0]
                range_val = max_val - min_val + 1e-8  # 避免除以零
                X_view = (X_view - min_val) / range_val
            elif self.normalize_method == 'tfidf':
                print(f"视图 {view_idx} 使用TF-IDF进行处理")
                # 先剔除缺失样本，再进行TF-IDF和SVD，最后补0
                if mask is not None:
                    # 检测完整样本
                    sample_mask = (mask.sum(dim=1) > 0)  # shape: (N,)
                    valid_indices = sample_mask.nonzero(as_tuple=True)[0]
                    invalid_indices = (~sample_mask).nonzero(as_tuple=True)[0]
                    
                    # 提取完整样本
                    valid_view = X_view[valid_indices].cpu().numpy()
                    
                    # TF-IDF处理（仅对完整样本）
                    tfidf = TfidfTransformer()
                    valid_tfidf = tfidf.fit_transform(valid_view)
                    
                    # 获取TF-IDF后的特征维度，动态确定SVD降维维度
                    tfidf_dim = valid_tfidf.shape[1]
                    svd_components = min(500, tfidf_dim)
                    svd = TruncatedSVD(n_components=svd_components, random_state=42)
                    valid_reduced = svd.fit_transform(valid_tfidf)
                    explained_variance = svd.explained_variance_ratio_.sum()
                    print(f"视图 {view_idx} 解释方差比: {explained_variance:.4f}, TF-IDF维度: {tfidf_dim}, SVD降维: {svd_components}")
                    
                    # 创建结果矩阵，先填0（动态维度）
                    N = X_view.shape[0]
                    reduced_dim = valid_reduced.shape[1]
                    reduced = torch.zeros((N, reduced_dim), dtype=torch.float32)
                    # 将处理后的结果放回对应位置
                    reduced[valid_indices] = torch.tensor(valid_reduced, dtype=torch.float32)
                    
                    # 重新生成掩码
                    new_mask = torch.ones((N, reduced_dim), dtype=torch.float32)
                    # 缺失样本位置的掩码设为0
                    new_mask[invalid_indices] = 0.0
                    
                    X_view = reduced
                    mask = new_mask
                else:
                    # 无缺失情况，直接处理
                    current_view = X_view.cpu().numpy()
                    tfidf = TfidfTransformer()
                    current_tfidf = tfidf.fit_transform(current_view)
                    tfidf_dim = current_tfidf.shape[1]
                    svd_components = min(500, tfidf_dim)
                    svd = TruncatedSVD(n_components=svd_components, random_state=42)
                    current_reduced = svd.fit_transform(current_tfidf)
                    explained_variance = svd.explained_variance_ratio_.sum()
                    print(f"视图 {view_idx} 解释方差比: {explained_variance:.4f}, TF-IDF维度: {tfidf_dim}, SVD降维: {svd_components}")
                    X_view = torch.tensor(current_reduced, dtype=torch.float32)
        else:
            print(f"视图 {view_idx} 不进行归一化")
        
        # 重新应用掩码，确保缺失值仍为0
        if mask is not None:
            X_view = X_view * mask
        
        return X_view, mask
    
    def _load_simulated_raw(self, N=200, V=3, latent_dim=2, obs_dim=100, seed=0):
        """
        生成论文 5.2.1 节描述的模拟多视图数据集
        
        Args:
            N: 样本数
            V: 视图数
            latent_dim: 潜在空间维度
            obs_dim: 观察空间维度
            seed: 随机种子
        
        Returns:
            X: 三阶张量 (N, D, V)
            Y: 一阶张量 (N,)
        """
        if seed is not None and seed != 'None':
            torch.manual_seed(int(seed))
        
        # 生成潜在表示
        n_per_class = N // 4
        centers = [torch.randn(latent_dim) * 2 for _ in range(4)]
        latent = torch.vstack([torch.randn(n_per_class, latent_dim) + center for center in centers])
        Y = torch.repeat_interleave(torch.tensor([0, 1, 2, 3]), n_per_class)

        # 生成随机映射矩阵
        A = torch.randn(obs_dim, latent_dim)

        # 定义非线性激活函数
        activation_funcs = [
            torch.sigmoid,  # sigmoid
            torch.tanh,
            torch.relu       # relu
        ]
        
        # 生成多视图数据（生成视图列表）
        X_list = []
        for v in range(V):
            func = activation_funcs[v % len(activation_funcs)]
            view = func(latent @ A.T) ** 2  # 线性变换 + 非线性激活 + 平方
            X_list.append(view)
        
        # 数据预处理（传入视图列表）
        X = self._preprocess_data(X_list)
        
        return X, Y
    
    def _load_matlab_raw(self, transpose_views=False):
        """
        通用的 MATLAB 数据集加载函数
        
        Args:
            transpose_views: 是否转置视图数据
        
        Returns:
            X: 三阶张量 (N, D, V)
            Y: 一阶张量 (N,)
        """
        # 文件名映射
        filename_map = {
            '100Leaves': '100leaves.mat',
            'BBC': 'BBC.mat',
            'MSRC': 'MSRC.mat',
            'WebKB': 'WebKB.mat',
            'BBCSport': 'BBCSport.mat',
            'Caltech101': 'Caltech101.mat'
        }
        filename = filename_map.get(self.dataset_name, f'{self.dataset_name.lower()}.mat')
        
        filepath = os.path.join(self.data_dir, filename)
        url = f"https://raw.githubusercontent.com/ericyangyu/mvc_datasets/main/{filename}"
        
        self._download_file(url, filepath)
        mat = loadmat(filepath)
        
        # 打印MATLAB文件中的键，以了解数据结构
        print(f"MATLAB文件中的键: {list(mat.keys())}")
        
        # 提取视图数据
        if 'X' in mat:
            views = mat['X'][0]
        elif 'fea' in mat:
            # Caltech101 数据集使用 'fea' 作为视图数据键
            print(f"使用键 'fea' 作为视图数据")
            views = mat['fea'][0]
        else:
            # 尝试其他可能的键名
            for key in mat.keys():
                if not key.startswith('__'):
                    print(f"尝试使用键 '{key}' 作为视图数据")
                    views = mat[key][0]
                    break
        
        # 处理标签（自动转换为0-based）
        if 'Y' in mat:
            y = mat['Y'].flatten().astype(int)
        elif 'gt' in mat:
            # Caltech101 数据集使用 'gt' 作为标签键
            print(f"使用键 'gt' 作为标签数据")
            y = mat['gt'].flatten().astype(int)
        else:
            # 尝试其他可能的键名
            for key in mat.keys():
                if not key.startswith('__') and key not in ['X', 'fea']:
                    print(f"尝试使用键 '{key}' 作为标签数据")
                    y = mat[key].flatten().astype(int)
                    break
        
        if y.min() == 1:
            y = y - 1
        Y = torch.tensor(y, dtype=torch.long)
        
        # 处理视图数据
        processed_views = []
        for view in views:
            # 检查是否为稀疏矩阵
            if hasattr(view, 'toarray'):
                # 转换为密集矩阵
                view = view.toarray()
            processed_views.append(torch.tensor(view.astype(np.float32)))
        
        X_list = []
        for i, view in enumerate(processed_views):
            if transpose_views:
                # 转置视图
                view = view.T
            X_list.append(view)
        
        return X_list, Y
    
    def _generate_missing_data(self, X, missing_rate=0.3, seed=None, missing_mode="sample"):
        """
        生成随机视图缺失数据，确保每个样本至少保留一个完整的视图（样本级）或至少一个特征（特征级）。
        采用精确控制缺失样本数的方法，使各视图缺失率接近目标值。

        Args:
            X: 视图列表，每个元素形状 (N, D_v)
            missing_rate: 目标缺失比例（0~1），若超过理论最大值则自动调整为最大值
            seed: 随机种子
            missing_mode: "sample"（样本级）或 "feature"（特征级）

        Returns:
            X_missing: 缺失后的视图列表
            masks: 掩码列表，1表示存在，0表示缺失
        """
        if seed is not None and seed != 'None':
            torch.manual_seed(int(seed))

        V = len(X)
        N = X[0].shape[0]
        X_missing = []
        masks = []

        if missing_mode == "sample":
            # ------------------- 样本级缺失 -------------------
            max_missing = (V - 1) / V
            if missing_rate > max_missing:
                print(f"警告：目标缺失率 {missing_rate:.3f} 超过理论最大值 {max_missing:.3f}（V={V}），已自动调整为 {max_missing:.3f}。")
                missing_rate = max_missing

            # 精确控制各视图缺失率
            target_missing_count = int(N * missing_rate)  # 每个视图期望缺失的样本数
            # 为防止所有视图同时缺失，需要确保每个样本至少保留一个视图，因此实际能允许的最大缺失率略低于理论值
            # 这里我们通过循环调整直到达到目标缺失率附近
            while True:
                # 为每个视图生成缺失样本索引
                missing_indices_per_view = []
                for v in range(V):
                    # 随机选择 target_missing_count 个样本作为缺失
                    perm = torch.randperm(N, device=X[0].device)
                    missing = perm[:target_missing_count]
                    missing_indices_per_view.append(missing)

                # 构建掩码矩阵 (N, V)，1表示存在，0表示缺失
                mask = torch.ones((N, V), device=X[0].device)
                for v in range(V):
                    mask[missing_indices_per_view[v], v] = 0

                # 确保每个样本至少有一个视图存在
                all_missing = (mask.sum(dim=1) == 0)
                if all_missing.any():
                    # 为全缺失样本随机恢复一个视图
                    n_missing = all_missing.sum().item()
                    recover_view = torch.randint(0, V, (n_missing,), device=X[0].device)
                    mask[all_missing, recover_view] = 1

                # 计算实际缺失率
                actual_missing_rates = [1 - mask[:, v].mean().item() for v in range(V)]
                avg_actual = sum(actual_missing_rates) / V
                # 如果实际平均缺失率与目标相差小于0.01，则接受；否则微调 target_missing_count
                if abs(avg_actual - missing_rate) < 0.01:
                    break
                else:
                    # 调整 target_missing_count
                    if avg_actual < missing_rate:
                        target_missing_count = min(N-1, target_missing_count + 1)
                    else:
                        target_missing_count = max(0, target_missing_count - 1)
                    # 避免死循环（最多循环100次）
                    if target_missing_count < 0 or target_missing_count >= N:
                        break

            # 将掩码扩展到每个视图的特征维度
            masks = [mask[:, v:v+1].expand(-1, X[v].shape[1]) for v in range(V)]
            # 应用掩码
            X_missing = [X[v] * masks[v] for v in range(V)]

        else:  # feature 模式
            # ------------------- 特征级缺失 -------------------
            for v in range(V):
                D_v = X[v].shape[1]
                # 生成每个特征的掩码 (N, D_v)
                mask = torch.bernoulli(torch.full((N, D_v), 1 - missing_rate, device=X[v].device))
                # 确保每个样本至少有一个特征存在（避免全零行）
                row_sum = mask.sum(dim=1)
                all_zero = (row_sum == 0)
                if all_zero.any():
                    n_zero = all_zero.sum().item()
                    recover_col = torch.randint(0, D_v, (n_zero,), device=X[v].device)
                    mask[all_zero, recover_col] = 1
                masks.append(mask)
                X_missing.append(X[v] * mask)

        # 打印实际缺失率
        for v in range(V):
            actual = 1 - masks[v].sum().item() / masks[v].numel()
            print(f"视图 {v} 实际缺失率: {actual:.4f} (模式: {missing_mode})")

        return X_missing, masks
    
    def load_dataset(self, missing_rate=0.0, seed=None, missing_mode="sample"):
        """
        统一的数据集加载方法
        
        Args:
            missing_rate: 缺失比例，范围 [0, 1)
            seed: 随机种子，用于可重复性
            missing_mode: 缺失模式，"sample" (样本级) 或 "feature" (特征级)
        
        Returns:
            X: 三阶张量 (N, D, V)
            Y: 一阶张量 (N,)
            masks: 掩码列表，1表示存在，0表示缺失（如果 missing_rate > 0）
        """
        # 使用类初始化时设置的数据集名称
        current_dataset = self.dataset_name
        
        # 确保数据集名称存在
        if current_dataset is None:
            print("未指定数据集名称，使用默认数据集 MSRC")
            current_dataset = 'MSRC'
        
        # 所有数据集通过通用加载函数处理
        if current_dataset in self.DATASET_CONFIGS:
            loader_name, params = self.DATASET_CONFIGS[current_dataset]
            loader = getattr(self, loader_name)
            X, Y = loader(**params)
        else:
            print(f"未知数据集: {current_dataset}，使用默认数据集 MSRC")
            loader_name, params = self.DATASET_CONFIGS['MSRC']
            loader = getattr(self, loader_name)
            X, Y = loader(**params)
        
        # 生成缺失数据
        masks = None
        if missing_rate > 0:
            print(f"生成缺失数据，缺失率: {missing_rate}，模式: {missing_mode}")
            X, masks = self._generate_missing_data(X, missing_rate, seed, missing_mode)
        
        # 对数据进行预处理
        if masks is not None:
            X , masks = self._preprocess_data(X, masks)
        else:
            X = self._preprocess_data(X)
        
        if masks is not None:
            return X, Y, masks
        else:
            return X, Y
