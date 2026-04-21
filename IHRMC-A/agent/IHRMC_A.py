"""
IHRMC-A 模型实现

改进的分层表示多视图聚类模型
"""

import math
import time
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from sklearn.cluster import SpectralClustering
from net.network import Autoencoder, SingleLayerAE, pretrain_single_layer_ae, update_single_ae
from solver.solve import SSC_Basic, TNN, proj_simplex_batch, t_svt
from utils.utils import best_map, block_diag_ratio, select_anchors
class IHRMC_A:
    """
    改进的分层表示多视图聚类 (Improved Hierarchical Representation Multi-view Clustering - A)
    
    论文：Improved Hierarchical Representation for Multi-view Clustering with Anchor-based Representation and Hypergraph Regularization
    实现基于 ADMM 优化算法。
    
    核心功能：
    1. 多视图特征学习：通过自编码器学习每个视图的低维表示
    2. 锚点表达学习：使用锚点系数矩阵学习样本之间的关系
    3. 一致性约束：强制不同视图的表示系数矩阵趋于统一
    4. 张量低秩约束：捕捉视图间的高阶相关性
    5. 超图正则化：利用超图结构增强聚类性能
    6. 谱聚类：基于学习到的相似矩阵进行聚类
    
    优化过程：
    - 使用 ADMM (Alternating Direction Method of Multipliers) 算法
    - 交替更新自编码器参数、锚点系数矩阵、统一相似矩阵和张量变量
    """
    def __init__(
        self,
        n_clusters: int,
        params: Optional[List[float]] = None,
        m: int = 28,  # 锚点数
        max_iters: int = 100,  # ADMM最大迭代次数
        T: int = 20,  # 自适应权重预热轮数
        iter_t: int = 20,  # 每个 ADMM 迭代中自编码器的训练轮数（预训练和主训练共用）
        batch_size: int = 32,  # 批次大小
        random_seed: Optional[int] = None,  # 随机种子
        lr: float = 2**(-10),  # 学习率（预训练和主训练共用）
        epochs: int = 2,  # 训练轮数（预训练和主训练共用）
        device: str = 'cuda',
        hidden_dims: List[int] = [200, 100],  # 自编码器隐藏层维度
        latent_dim: int = 100,  # 自编码器潜在表示维度
        beta_max: float = 10,  # 惩罚参数最大值
        tol: float = 1e-7,  # 收敛容差
        use_anchor: bool = True,  # 是否使用锚点表达
        use_adaptive_weight: bool = True,  # 是否使用自适应权重
        lightweight_mode: bool = False,  # 轻量级模式：True时不计算损失和可视化
        use_pre_cache: bool = True,  # 是否使用预训练缓存
        cv_init_method: int = 0,  # Cv初始化方式
        early_stop: bool = True,  # 是否使用早停机制
        early_stop_min_delta: float = 1e-3,  # 早停最小改善值
        anchor_selection: str = 'kmeans',  # 锚点选择方法: 'kmeans' 或 'random'
        ):  
        """
        初始化IHRMC-A模型
        
        Args:
            n_clusters: 聚类数
            params: 参数列表 [lambda1, lambda2, lambda3, beta]
                    其中 lambda1 是网络权重L2正则化系数，lambda2 是自表达损失权重，lambda3 是张量核范数权重
            m: 锚点数
            beta_max: 惩罚参数最大值
            lr: 自编码器学习率
            epochs: 每个 ADMM 迭代中自编码器的训练轮数
            tol: 收敛容差（基于 ||W - C||_inf）
            max_iters: ADMM最大迭代次数
            T: 自适应权重预热轮数
            iter_t: 每个 ADMM 迭代中自编码器的训练轮数（预训练和主训练共用）
            batch_size: 批次大小
            device: 计算设备 ('cpu' 或 'cuda')
            hidden_dims: 自编码器隐藏层维度列表
            latent_dim: 自编码器潜在表示维度
            use_pre_cache: 是否使用预训练缓存
            random_seed: 随机种子（用于缓存键），为 None 时使用默认值 42
            cv_init_method: Cv初始化方式，0: 基本SSC(原始X), 1: 零矩阵, 2: 随机矩阵
            early_stop_min_delta: 早停最小改善值，即损失改善的阈值
            anchor_selection: 锚点选择方法: 'kmeans' 或 'random'
            use_anchor: 是否使用锚点表达
            lightweight_mode: 轻量级模式：True时不计算损失和可视化，加速训练
            use_adaptive_weight: 是否使用自适应权重
        """
        # 默认核心超参数 [lambda1, lambda2, lambda3, beta]
        default_params = [0.001, 0.1, 1.0, 0.1]
        
        # 更新核心超参数
        if params is None:
            params = default_params
        elif len(params) < 4:
            # 如果列表长度不足，使用默认值填充
            params = params + default_params[len(params):]
        
        # 存储超参数
        self.n_clusters = int(n_clusters)
        self.lambda1 = float(params[0])  # 网络权重L2正则化系数
        self.lambda2 = float(params[1])  # 自表达损失权重
        self.lambda3 = float(params[2])  # 张量核范数权重
        self.beta = float(params[3]) if len(params) > 3 else 0.1
        self.m = int(m)
        self.max_iters = int(max_iters)
        self.T = int(T)
        self.iter_t = int(iter_t)
        self.batch_size = int(batch_size)
        if random_seed == 'None' or random_seed is None:
            self.random_seed = None
        else:
            self.random_seed = int(random_seed)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.device = device
        self.hidden_dims = [int(d) for d in hidden_dims]
        self.latent_dim = int(latent_dim)
        self.beta_max = float(beta_max)
        self.tol = float(tol)
        self.use_anchor = bool(use_anchor)
        self.use_adaptive_weight = bool(use_adaptive_weight)
        self.lightweight_mode = bool(lightweight_mode)
        self.use_pre_cache = bool(use_pre_cache)
        self.cv_init_method = int(cv_init_method)
        self.early_stop = bool(early_stop) 
        self.early_stop_min_delta = float(early_stop_min_delta)
        self.anchor_selection = anchor_selection
        # 混合精度训练
        self.scaler = GradScaler(enabled=self.device.startswith('cuda'))
        # 初始化内存跟踪
        self.max_memory_allocated = 0
        # 以下变量将在 fit 方法中初始化
        self.aes: Optional[List[Autoencoder]] = None          # 每个视图的自编码器列表
        self.opts: Optional[List[optim.Optimizer]] = None     # 对应的优化器列表
        self.C: Optional[List[torch.Tensor]] = None           # 每个视图的锚点表示系数矩阵列表，形状 [(m,N), ...]
        # 锚点表示系数矩阵张量，形状 (m,N,V)
        self.S = None              # 统一相似矩阵，形状 (m,N)
        self.W = None              # 张量变量，形状 (m,N,V)
        self.P = None              # 拉格朗日乘子，形状 (m,N,V)
        # 锚点相关变量
        self.anchor_idx = None # 共享锚点索引，形状 (m,)
        # 缺失视图相关变量
        self.masks = None          # 掩码列表，1表示存在，0表示缺失
        
        self.X = None
        self.Y = None
        self.V: Optional[int] = None  # 视图数
        self.N: Optional[int] = None  # 样本数
        self.start_time: float = 0.0  # 训练开始时间
        # 初始化训练历史记录
        self.history = {
            # 基本信息（每10次迭代）
            '迭代次数': [],  # 迭代次数
            '运行时间': [],  # 运行时间
            '计算容差': [],  # 计算容差

            # 每次迭代的损失指标（由_compute_loss计算）
            '原始目标值': [],  # 原始联合优化目标函数值
            '重建损失': [],  # 重建损失
            '表达损失': [],  # 表达损失（锚点/自表达）
            '正则化损失': [],  # 网络权重L2正则化损失
            '一致性损失': [],  # 一致性损失（∑‖C_v - S‖_F）
            '张量核范数': [],  # 张量核范数

            # 聚类性能指标（每10次迭代）
            'ACC': [],  # 聚类准确率
            'NMI': [],  # 标准化互信息
            'ARI': [],  # 调整兰德指数

            # 矩阵特性指标（每10次迭代）
            'S矩阵块对角比': [],  # S矩阵块对角比

            # 潜在表示指标（每10次迭代）
            'H类间/类内距离比': []  # 类间/类内距离比
        }

    def fit(self, X: List[torch.Tensor], Y: Optional[torch.Tensor] = None, masks: Optional[List[torch.Tensor]] = None) -> Tuple[np.ndarray, Dict[str, List[float]]]:
        """
        训练 IHRMC-A 模型，得到聚类标签。

        Args:
            X: 视图列表，每个元素是形状为 (N, D_v) 的张量，其中 N 是样本数，D_v 是第 v 个视图的特征维度，V 是视图数。
            Y: 可选，一阶张量，形状 (N,)，真实标签，用于评估和可视化
            masks: 可选，掩码列表，每个元素是形状为 (N, D_v) 的张量，1表示存在，0表示缺失
        
        Returns:
            labels: 聚类标签数组，形状 (N,)
            history: 训练历史记录，包含每次迭代的损失和计算容差信息，以及每10次迭代的详细信息
        """
        
         # 记录初始内存使用
        if self.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
            self.max_memory_allocated = torch.cuda.max_memory_allocated() / 1024**2  # 转换为MB
        else:
            # 记录CPU内存使用
            import psutil
            import os
            process = psutil.Process(os.getpid())
            self.max_memory_allocated = process.memory_info().rss / 1024**2  # 转换为MB
        # 1. 初始化参数
        self._init_params(X, Y, masks)
        
        # IHRMC-A模式：在ADMM主循环之前进行一次预训练
        if self.use_anchor:
            self._pretrain_ae()
        
        # 2. ADMM 主循环
        for t in range(self.max_iters):  # 最大迭代次数
            # 记录开始时间
            if t % self.iter_t == 0:
                start_time = time.time()
                update_ae_start = time.time()
            
            # 2.1 更新自编码器参数
            # 每20次迭代更新一次自编码器
            if t % self.iter_t == 0:
                if not self.use_anchor :
                    # HRMC模式：预训练自编码器
                    self._pretrain_ae()
                    self._update_ae()
                else:
                    self._update_ae()
                
                
            
            if t % self.iter_t == 0:
                update_ae_time = time.time() - update_ae_start
                update_c_start = time.time()
            
            # 2.2 更新锚点表达系数矩阵 C
            self._update_C()
            
            if t % self.iter_t == 0:
                update_c_time = time.time() - update_c_start
                update_s_start = time.time()
            
            # 2.3 更新统一相似矩阵 S 和自适应权重
            self._update_S(t)
            
            if t % self.iter_t == 0:
                update_s_time = time.time() - update_s_start
                update_w_start = time.time()
            
            # 2.4 更新张量 W
            self._update_W()
            
            if t % self.iter_t == 0:
                update_w_time = time.time() - update_w_start
                update_p_start = time.time()
            
            # 2.5 更新拉格朗日乘子 P                
            self._update_P()
            
            if t % self.iter_t == 0:
                update_p_time = time.time() - update_p_start
                calculate_loss_start = time.time()
            
            # 2.6 计算损失
            if not self.lightweight_mode:
                compute_full = (t % 10 == 0)
                self._compute_loss(full=compute_full)
            
            if t % self.iter_t == 0:
                calculate_loss_time = time.time() - calculate_loss_start
                total_time = time.time() - start_time
                
                # 打印第t次迭代的耗时
                print(f"第{t}次迭代耗时：")
                print(f"  更新自编码器: {update_ae_time:.4f} 秒")
                print(f"  更新锚点表达系数: {update_c_time:.4f} 秒")
                print(f"  更新S矩阵和权重: {update_s_time:.4f} 秒")
                print(f"  更新W矩阵: {update_w_time:.4f} 秒")
                print(f"  更新P矩阵: {update_p_time:.4f} 秒")
                print(f"  计算损失: {calculate_loss_time:.4f} 秒")
                print(f"  总耗时: {total_time:.4f} 秒")
            
            # 2.7 更新惩罚参数 beta
            self.beta = min(self.beta * 1.1, self.beta_max)
            
            # 2.8 检查收敛
            convergence = self._check_convergence(t)
            
            # 2.9 记录历史和日志
            if not self.lightweight_mode and (t % 10 == 0 or convergence):
                self._record_info(t)
            
            # 2.10 更新内存使用情况
            if self.device.startswith('cuda'):
                current_max = torch.cuda.max_memory_allocated() / 1024**2  # 转换为MB
                if current_max > self.max_memory_allocated:
                    self.max_memory_allocated = current_max
            else:
                # 记录CPU内存使用
                process = psutil.Process(os.getpid())
                current_max = process.memory_info().rss / 1024**2  # 转换为MB
                if current_max > self.max_memory_allocated:
                    self.max_memory_allocated = current_max
            
            # 2.11 如果收敛，退出循环
            if convergence:
                break

        # 3. 执行谱聚类
        labels = self._spectral_cluster()
        
        # 计算运行时间
        end_time = time.time()
        run_time = end_time - self.start_time
        
        # 最终更新内存使用情况
        if self.device.startswith('cuda'):
            current_max = torch.cuda.max_memory_allocated() / 1024**2  # 转换为MB
            if current_max > self.max_memory_allocated:
                self.max_memory_allocated = current_max
            print(f"IHRMC-A 训练完成，总运行时间: {run_time:.2f} 秒，最大显存使用: {self.max_memory_allocated:.2f} MB")
        else:
            # 记录CPU内存使用
            process = psutil.Process(os.getpid())
            current_max = process.memory_info().rss / 1024**2  # 转换为MB
            if current_max > self.max_memory_allocated:
                self.max_memory_allocated = current_max
            print(f"IHRMC-A 训练完成，总运行时间: {run_time:.2f} 秒，最大内存使用: {self.max_memory_allocated:.2f} MB")
        
        # 将运行时间和内存使用添加到历史记录
        self.history['run_time'] = run_time
        self.history['max_memory_allocated'] = self.max_memory_allocated
        
        return labels, self.history
    
    def _init_params(self, X: List[torch.Tensor], Y: Optional[torch.Tensor]=None, masks: Optional[List[torch.Tensor]]=None):
        """
        初始化IHRMC-A模型参数
        """
        # 1. 初始化网络结构和参数
        self.start_time = time.time()
        self.X = [x.to(self.device).float() for x in X]
        # 保存标签用于后续分析
        if Y is not None:
            self.Y = Y
        self.N = self.X[0].shape[0]  # 样本数
        self.V = len(self.X)  # 视图数
        # 初始权重为1/V
        self.omegas = [1/(2*self.V) for _ in range(self.V)]
        
        # 保存掩码
        if masks is not None:
            self.masks = [mask.to(self.device).float() for mask in masks]
        with torch.no_grad():
            # 根据use_anchor参数设置m的值
            if not self.use_anchor:
                # 使用原始HRMC架构，m=N
                self.m = self.N
                # 设置锚点索引为全样本索引
                self.anchor_idx = torch.arange(self.N, device=self.device)
                print(f"使用原始HRMC架构，m=N={self.N}")
            else:
                # 2. 选择锚点索引
                # 从所有视图中随机选择一个视图来初始化锚点
                random_view_idx = np.random.choice(range(self.V))
                print(f"随机选择视图 {random_view_idx} 来初始化锚点")
                if masks is not None:
                    # 如果提供了掩码，使用掩码初始化锚点
                    self.anchor_idx = select_anchors(
                        H=self.X[random_view_idx],
                        m=self.m,
                        anchor_selection=self.anchor_selection,
                        random_seed=self.random_seed,
                        device=self.device,
                        mask=self.masks[random_view_idx],
                        masks=self.masks,
                        X=self.X
                    )
                else:
                    # 如果没有提供掩码，使用所有样本初始化锚点
                    self.anchor_idx = select_anchors(
                        H=self.X[random_view_idx],
                        m=self.m,
                        anchor_selection=self.anchor_selection,
                        random_seed=self.random_seed,
                        device=self.device
                    )
        
        # 3.初始化锚点系数矩阵
        self.C = torch.zeros(self.m, self.N, self.V, device=self.device, dtype=torch.float32)
        for v in range(self.V):
            Xv = self.X[v]
            
            # 处理无效的初始化方法
            if self.cv_init_method not in [0, 1, 2]:
                print(f"cv_init_method={self.cv_init_method} 无效，使用默认方式0（基本SSC）")
                init_method = 0
            else:
                init_method = self.cv_init_method
            
            if init_method == 0:
                # 方式0: 使用基本SSC
                if self.use_anchor:
                    Cv = SSC_Basic(Xv, anchor_indices=self.anchor_idx)
                    A = Xv[self.anchor_idx, :]
                    recon_error = torch.norm(Xv - (A.T @ Cv).T, p='fro').item()
                    print(f"[原始X] 视图 {v} 基本SSC 初始化锚点表达误差 = {recon_error:.4f}")
                else:
                    Cv = SSC_Basic(Xv)
                    recon_error = torch.norm(Xv - Cv @ Xv, p='fro').item()
                    print(f"[原始X] 视图 {v} 基本SSC 初始化自表达误差 = {recon_error:.4f}")
                Cv = Cv.to(self.device)
            elif init_method == 1:
                # 方式1: 零矩阵
                Cv = torch.zeros(self.m, self.N, device=self.device, dtype=torch.float32)
                print(f"[零矩阵] 视图 {v} Cv 初始化完成")
            else:  # init_method == 2
                # 方式2: 随机矩阵
                if self.random_seed is not None:
                    g = torch.Generator(device='cpu').manual_seed(self.random_seed)
                    Cv = torch.randn(self.m, self.N, dtype=torch.float32, generator=g).to(self.device)
                else:
                    Cv = torch.randn(self.m, self.N, device=self.device, dtype=torch.float32)
                # 对角线置零（针对自表达模式）
                if not self.use_anchor:
                    Cv = Cv - torch.diag(torch.diag(Cv))
                print(f"[随机矩阵] 视图 {v} Cv 初始化完成")
            
            self.C[:, :, v] = Cv  # 直接使用 (m, N) 形状
        # 初始化统一相似矩阵 S、张量 W 和 拉格朗日乘子 P
        self.S = torch.zeros(self.m, self.N, device=self.device, dtype=torch.float32)    #(m, N)
        self.W = torch.zeros(self.m, self.N, self.V, device=self.device, dtype=torch.float32)  #(m, N, V)
        self.P = torch.zeros(self.m, self.N, self.V, device=self.device, dtype=torch.float32)  #(m, N, V)
    
    def _pretrain_ae(self):
        """
        预训练自编码器
        当use_anchor=False时使用HRMC的内层循环方式
        当use_anchor=True时使用IHRMC-A的锚点表达方式
        """
        # 对每个视图进行逐层预训练
        self.aes = []
        self.opts = []
        
        for v in range(self.V):
            Xv = self.X[v]
            Cv = self.C[:, :, v]
            if self.masks is not None:
                mask = self.masks[v]
            else:
                mask = None
            input_dim = Xv.shape[1]
            
            # 逐层预训练
            current_input = Xv
            prev_dim = input_dim
            
            # 第一层：input -> hidden_dims[0]
            model1 = SingleLayerAE(prev_dim, self.hidden_dims[0]).to(self.device)
            torch.nn.init.eye_(model1.encoder.weight)
            torch.nn.init.zeros_(model1.encoder.bias)
            torch.nn.init.eye_(model1.decoder.weight)
            torch.nn.init.zeros_(model1.decoder.bias)
            optimizer1 = optim.SGD(model1.parameters(), lr=self.lr)
            
            pretrain_single_layer_ae(model1, optimizer1, current_input, Cv,
                                    self.lambda1, self.lambda2, self.epochs, 
                                    self.batch_size,self.early_stop, self.early_stop_min_delta,
                                    self.anchor_idx, mask)
            
            with torch.no_grad():
                _, H1 = model1(current_input)
            current_input = H1
            prev_dim = self.hidden_dims[0]
            
            # 第二层：hidden_dims[0] -> hidden_dims[1]
            model2 = SingleLayerAE(prev_dim, self.hidden_dims[1]).to(self.device)
            torch.nn.init.eye_(model2.encoder.weight)
            torch.nn.init.zeros_(model2.encoder.bias)
            torch.nn.init.eye_(model2.decoder.weight)
            torch.nn.init.zeros_(model2.decoder.bias)
            optimizer2 = optim.SGD(model2.parameters(), lr=self.lr)
            
            # 第二层输入是编码后的特征，不存在缺失值，不需要掩码
            pretrain_single_layer_ae(model2, optimizer2, current_input, Cv,
                                    self.lambda1, self.lambda2, self.epochs, 
                                    self.batch_size, self.early_stop, self.early_stop_min_delta,
                                    self.anchor_idx, None)
            
            # 构建完整自编码器
            ae = Autoencoder(input_dim, hidden_dims=self.hidden_dims, latent_dim=self.latent_dim).to(self.device)
            
            # 初始化编码器
            ae.encoder[0].weight.data = model1.encoder.weight.data
            ae.encoder[0].bias.data = model1.encoder.bias.data
            ae.encoder[2].weight.data = model2.encoder.weight.data
            ae.encoder[2].bias.data = model2.encoder.bias.data
            
            # 初始化解码器
            ae.decoder[0].weight.data = model2.encoder.weight.data.T
            ae.decoder[0].bias.data = torch.zeros(self.hidden_dims[0], device=self.device, dtype=torch.float32)
            ae.decoder[2].weight.data = model1.encoder.weight.data.T
            ae.decoder[2].bias.data = torch.zeros(input_dim, device=self.device, dtype=torch.float32)
            
            self.aes.append(ae)
            self.opts.append(optim.SGD(ae.parameters(), lr=self.lr))
            
            # 删除中间变量释放内存
            del model1, model2, optimizer1, optimizer2, H1

    def _update_ae(self):
        """
        更新自编码器参数
        """
        # 串行更新自编码器
        for v in range(self.V):
            ae = self.aes[v]
            opt = self.opts[v]
            Xv = self.X[v]
            Cv = self.C[:, :, v].detach()
            if self.masks is not None:
                mask = self.masks[v]
            else:
                mask = None
            
            updated_state_dict = update_single_ae(
                ae, opt, Xv, Cv,  self.lambda1, self.lambda2,
                self.epochs, self.batch_size, self.early_stop, self.early_stop_min_delta, 
                self.anchor_idx, mask, self.scaler
            )
            
            self.aes[v].load_state_dict(updated_state_dict)

    def _update_C(self):
        with torch.no_grad():
            if not self.use_anchor:
                I = torch.eye(self.N, device=self.device)
                for v in range(self.V):
                    ae = self.aes[v]
                    Xv = self.X[v]
                    
                    # 全批次获取潜在表示
                    _, latent = ae(Xv)
                    H = latent  # (N, d_latent)
                    
                    # 计算矩阵 A = H @ H^T
                    A = H @ H.T  # (N,N)
                    # 左侧: lambda2 * A + (omegas[v] + beta) * I
                    LHS = self.lambda2 * A + (2 * self.omegas[v] + self.beta) * I
                    # 右侧: omegas[v] * S + lambda2 * A + beta * W[:,:,v] + P[:,:,v]
                    rhs = (2 * self.omegas[v] * self.S +
                           self.lambda2 * A +
                           self.beta * self.W[:, :, v] +
                           self.P[:, :, v])
                    # 求解线性系统 LHS @ C_new = rhs
                    # 使用 Cholesky 分解优化（对于正定矩阵）
                    try:
                        L = torch.linalg.cholesky(LHS)
                        C_new = torch.cholesky_solve(rhs, L)
                    except torch.linalg.LinAlgError:
                        # 如果 Cholesky 分解失败，回退到一般求解
                        C_new = torch.linalg.solve(LHS, rhs)
                    # 对角线置零
                    C_new = C_new - torch.diag(torch.diag(C_new))
                    self.C[:, :, v] = C_new
            else:
                for v in range(self.V):
                    ae = self.aes[v]
                    Xv = self.X[v]
                        
                    # 全批次获取潜在表示
                    _, latent = ae(Xv)
                    H = latent  # (N, d_latent)
                        
                    # 获取锚点的潜在表示
                    M = H[self.anchor_idx, :]  # (m, d_latent)
                        
                    # 计算矩阵 A = M @ M^T
                    A = M @ M.T  # (m, m)
                    # 左侧: lambda2 * A + (2*omegas[v] + beta) * I
                    I_m = torch.eye(self.m, device=self.device)
                    LHS = self.lambda2 * A + ( 2 * self.omegas[v] + self.beta) * I_m
                    # 右侧: 2 * omegas[v] * S + lambda2 * M @ H.T + beta * W[:,:,v] + P[:,:,v]
                    rhs = ( 2 * self.omegas[v] * self.S +
                            self.lambda2 * M @ H.T +
                            self.beta * self.W[:, :, v] +
                            self.P[:, :, v])
                    # 求解线性系统 LHS @ C_new = rhs
                    # 使用 Cholesky 分解优化（对于正定矩阵）
                    try:
                        L = torch.linalg.cholesky(LHS)
                        C_new = torch.cholesky_solve(rhs, L)
                    except torch.linalg.LinAlgError:
                        # 如果 Cholesky 分解失败，回退到一般求解
                        C_new = torch.linalg.solve(LHS, rhs)
                    # 对角线置零（针对每一行，而非整个矩阵）
                    # C_new 形状为 (m, N)，只需要将每一行的对角线元素（当列索引等于行索引时）置零
                    min_dim = min(C_new.shape[0], C_new.shape[1])
                    for i in range(min_dim):
                        C_new[i, i] = 0
                    # 投影到概率单纯形
                    C_proj_v = proj_simplex_batch(C_new)
                    self.C[:, :, v] = C_proj_v
    
    def _update_S(self, t: int):
        """
        更新统一相似矩阵 S 和自适应权重
        基于锚点表达系数矩阵 C 计算
        """
        with torch.no_grad():
            if self.use_adaptive_weight and t > self.T:
                # 计算自适应权重 ω_v = 1/(||C_v - S_prev||_F)
                for v in range(self.V):
                    Cv = self.C[:, :, v]
                    diff = Cv - self.S
                    norm = torch.norm(diff, p='fro')
                    # 防止除零
                    self.omegas[v] = 1.0 / (2.0 * (norm + 1e-8))
                # 使用加权平均计算 S: S = (∑ω_v * C_v) / (∑ω_v)
                weighted_sum = torch.zeros_like(self.C[:, :, 0])
                omega_sum = 0.0
                for v in range(self.V):
                    weighted_sum += self.omegas[v] * self.C[:, :, v]
                    omega_sum += self.omegas[v]
                self.S = weighted_sum / omega_sum
                
            elif self.use_adaptive_weight:
                # 使用固定权重为1/V
                self.omegas = [1.0 / self.V for _ in range(self.V)]
                # 计算所有视图的平均值
                self.S = torch.mean(self.C, dim=2)  # (m, N)
            else:
                # 使用固定权重为1/2
                self.omegas = [0.5 for _ in range(self.V)]
                # 计算所有视图的平均值
                self.S = torch.mean(self.C, dim=2)  # (m, N)
        
    def _update_W(self):
        """
        更新张量变量 W
        使用锚点表达系数张量进行低秩约束
        """
        with torch.no_grad():
            self.W, _ = t_svt(
                self.C - self.P / self.beta,
                self.lambda3 / self.beta,
                mode=3)
    
    def _update_P(self):
        """
        更新拉格朗日乘子 P
        """
        with torch.no_grad():
            self.P.add_(self.beta * (self.W - self.C))
    
    def _check_convergence(self, t: int):
        """
        检查模型是否收敛
        
        Args:
            t: 当前迭代次数
            
        Returns:
            是否已收敛
        """
        with torch.no_grad():
            # 计算 ||C - W||_inf，使用向量化操作
            diff = self.C - self.W
            max_tolerance = torch.max(torch.abs(diff)).item()
            
            # 记录最大容差
            self.history['计算容差'].append(max_tolerance)
        
        if max_tolerance < self.tol:
            print(f"Converged at iteration {t}, tolerance={max_tolerance:.2e}")
            return True

        return False
    
    def _spectral_cluster(self):
        """
        执行聚类
        
        Returns:
            聚类标签
        """
        if not self.use_anchor:
            # 使用HRMC的方法
            self.A = self.S.clone()
            self.A = (self.A + self.A.T) * 0.5
            # 对角线置零
            self.A.fill_diagonal_(0)
        else:
            # 使用IHRMC-A的方法
            self.A = self.S.T @ self.S  # (N, N)
           
        # 构建特征矩阵
        features = self.A.detach()
        
        # 使用谱聚类
        sc = SpectralClustering(n_clusters=self.n_clusters, affinity='precomputed', random_state=self.random_seed)
       
        labels = sc.fit_predict(features.cpu().numpy())

        # 使用匈牙利算法重排预测标签，使其与真实标签匹配（仅当有真实标签时）
        if hasattr(self, 'Y') and self.Y is not None:
            # 确保 labels 与 self.Y 在同一个设备上
            if isinstance(labels, np.ndarray):
                labels = torch.tensor(labels, device=self.Y.device)
            labels = best_map(self.Y, labels)
        else:
            # 如果没有真实标签，也将numpy数组转换为张量
            if isinstance(labels, np.ndarray):
                labels = torch.tensor(labels, device=self.device)
        
        return labels
    
    def _compute_loss(self, full=False):
        """
        计算原始联合优化目标函数值（不含ADMM惩罚项），并记录到历史中。
        该函数应在每次迭代的变量更新完毕后调用。
        
        Args:
            full: 是否计算完整损失（包括重建损失和正则化），默认为False只计算轻量级损失
        """
        with torch.no_grad():
            if full:
                # 完整损失计算（每10次迭代执行一次）
                total_recon = 0.0
                total_expr = 0.0
                total_reg = 0.0
                total_cons = 0.0

                for v in range(self.V):
                    Xv = self.X[v]
                    # 检查是否有掩码
                    if self.masks is not None:
                        mask = self.masks[v]
                    else:
                        # 没有掩码时，使用全 1 的掩码
                        mask = torch.ones_like(Xv, device=self.device)
                    recon, latent = self.aes[v](Xv)

                    # 1. 重建损失 - 只计算非缺失样本
                    diff = Xv - recon
                    # 应用掩码，只计算非缺失特征的损失
                    masked_diff = diff * mask
                    # 计算非缺失特征的数量
                    non_missing_count = mask.sum()
                    if non_missing_count > 0:
                        recon_loss = 0.5 * torch.sum(masked_diff * masked_diff) / non_missing_count
                    else:
                        recon_loss = 0.0
                    total_recon += recon_loss.item()

                    # 2. 表达损失 - 只计算非缺失样本
                    if self.lambda2 > 0:
                        Cv = self.C[:, :, v]
                        if self.use_anchor:
                            A = latent[self.anchor_idx, :]
                            reconstructed = (A.T @ Cv).T
                            diff = latent - reconstructed
                            # 应用掩码，只计算非缺失样本的损失
                            # 对于样本级缺失，使用行掩码
                            sample_mask = mask.sum(dim=1, keepdim=True) > 0
                            masked_diff = diff * sample_mask
                            # 计算非缺失样本的数量
                            non_missing_samples = sample_mask.sum()
                            if non_missing_samples > 0:
                                expr_loss = self.lambda2 * 0.5 * torch.sum(masked_diff * masked_diff) / non_missing_samples
                            else:
                                expr_loss = 0.0
                        else:
                            diff = latent - Cv @ latent
                            # 应用掩码，只计算非缺失样本的损失
                            sample_mask = mask.sum(dim=1, keepdim=True) > 0
                            masked_diff = diff * sample_mask
                            # 计算非缺失样本的数量
                            non_missing_samples = sample_mask.sum()
                            if non_missing_samples > 0:
                                expr_loss = self.lambda2 * 0.5 * torch.sum(masked_diff * masked_diff) / non_missing_samples
                            else:
                                expr_loss = 0.0
                        total_expr += expr_loss.item()

                    # 3. 网络权重正则化（L2）
                    if self.lambda1 > 0:
                        reg_loss = 0.0
                        for p in self.aes[v].parameters():
                            reg_loss += torch.sum(p * p)
                        total_reg += (self.lambda1 * 0.5 * reg_loss).item()

                    # 4. 一致性损失
                    Cv = self.C[:, :, v]
                    diff = Cv - self.S
                    cons_loss = torch.sum(diff * diff)
                    total_cons += (self.omegas[v] * cons_loss).item()

                # 计算平均值
                avg_recon = total_recon / self.V
                avg_expr = total_expr / self.V
                avg_reg = total_reg / self.V
                avg_cons = total_cons / self.V
            else:
                # 轻量级损失计算（每次迭代执行）
                # 只计算一致性损失和张量核范数，避免前向传播
                total_cons = 0.0
                
                for v in range(self.V):
                    Cv = self.C[:, :, v]
                    diff = Cv - self.S
                    cons_loss = torch.sum(diff * diff)
                    total_cons += (self.omegas[v] * cons_loss).item()
                
                avg_cons = total_cons / self.V if self.V > 0 else 0.0
                
                # 使用上一次的值
                avg_recon = self.history['重建损失'][-1] if self.history['重建损失'] else 0.0
                avg_expr = self.history['表达损失'][-1] if self.history['表达损失'] else 0.0
                avg_reg = self.history['正则化损失'][-1] if self.history['正则化损失'] else 0.0


            # 张量核范数
            if self.lambda3 > 0:
                tensor_norm = self.lambda3  * 0.5 * TNN(self.C).item()
            else:
                tensor_norm = 0.0

            # 原始目标函数
            original_obj = avg_recon + avg_expr + avg_reg + avg_cons + tensor_norm

            # 记录到历史
            self.history['原始目标值'].append(original_obj)
            self.history['重建损失'].append(avg_recon)
            self.history['表达损失'].append(avg_expr)
            self.history['正则化损失'].append(avg_reg)
            self.history['一致性损失'].append(avg_cons)
            self.history['张量核范数'].append(tensor_norm)

    def _record_info(self, t):
        """
        记录额外的分析指标（聚类性能、块对角比等），每10次迭代调用一次。
        注意：此函数不修改已有损失历史，只追加新指标。
        """
        # 1. 从历史中获取当前迭代的损失值
        if len(self.history['原始目标值']) > 0:
            total_loss = self.history['原始目标值'][-1]
            recon_loss = self.history['重建损失'][-1]
            expr_loss = self.history['表达损失'][-1]
            reg_loss = self.history['正则化损失'][-1]
            cons_loss = self.history['一致性损失'][-1]
            tensor_norm = self.history['张量核范数'][-1]
            tol = self.history['计算容差'][-1] if self.history['计算容差'] else 0.0
            print(f"Iter {t:3d}, 原始目标值={total_loss:.4e}, "
                f"重建损失={recon_loss:.4e}, 表达损失={expr_loss:.4e}, 正则化损失={reg_loss:.4e}, "
                f"一致性损失={cons_loss:.4e}, 张量核范数={tensor_norm:.4e}, "
                f"计算容差={tol}")

        # 2. 记录 S 矩阵的块对角比
        with torch.no_grad():
            if self.S.shape[0] != self.N:
                # S 是 m×n 锚点-样本矩阵，构建样本间相似矩阵
                similarity = self.S.T @ self.S
            else:
                similarity = self.S
            if hasattr(self, 'Y') and self.Y is not None:
                y_true = self.Y.to(self.device)
                y_int = y_true.clone()
                if y_int.min() == 1:
                    y_int -= 1
                block_diag = block_diag_ratio(similarity, y_int)
                self.history['S矩阵块对角比'].append(block_diag)
            else:
                self.history['S矩阵块对角比'].append(0)

        # 3. 若提供真实标签，计算聚类指标
        if hasattr(self, 'Y') and self.Y is not None:
            from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
            labels = self._spectral_cluster()                     # 返回聚类标签
            y_true_np = self.Y.cpu().numpy()
            self.history['ACC'].append(accuracy_score(y_true_np, labels))
            self.history['NMI'].append(normalized_mutual_info_score(y_true_np, labels))
            self.history['ARI'].append(adjusted_rand_score(y_true_np, labels))

            # 4. 计算潜在表示 H 的类内/类间距离比
            # 优化：使用向量化计算，避免Python循环
            with torch.no_grad():
                _, H = self.aes[0](self.X[0])  # 取第一个视图的潜在表示
                y = y_int.cpu().numpy()
                unique_labels = np.unique(y)
                
                # 预计算所有样本的L2范数
                H_np = H.cpu().numpy()
                n_samples = H_np.shape[0]
                
                # 使用向量化计算距离矩阵
                intra_dists = []
                inter_dists = []
                
                for lab in unique_labels:
                    mask = (y == lab)
                    class_indices = np.where(mask)[0]
                    other_indices = np.where(~mask)[0]
                    
                    if len(class_indices) > 1:
                        # 类内距离：使用向量化计算
                        class_H = H_np[class_indices]
                        # 计算成对距离
                        diff = class_H[:, None, :] - class_H[None, :, :]
                        dists = np.linalg.norm(diff, axis=2)
                        # 取上三角（排除对角线）
                        upper_tri_indices = np.triu_indices(len(class_indices), k=1)
                        intra_dists.extend(dists[upper_tri_indices].tolist())
                    
                    if len(class_indices) > 0 and len(other_indices) > 0:
                        # 类间距离
                        class_H = H_np[class_indices]
                        other_H = H_np[other_indices]
                        # 计算所有类间距离
                        diff = class_H[:, None, :] - other_H[None, :, :]
                        dists = np.linalg.norm(diff, axis=2)
                        inter_dists.extend(dists.flatten().tolist())
                
                if intra_dists and inter_dists:
                    ratio = np.mean(inter_dists) / (np.mean(intra_dists) + 1e-8)
                    self.history['H类间/类内距离比'].append(ratio)
                else:
                    self.history['H类间/类内距离比'].append(0.0)
        else:
            # 无标签时填充零
            self.history['ACC'].append(0)
            self.history['NMI'].append(0)
            self.history['ARI'].append(0)
            self.history['H类间/类内距离比'].append(0)

        # 5. 记录当前迭代次数和运行时间
        self.history['迭代次数'].append(t)
        self.history['运行时间'].append(time.time() - self.start_time)
