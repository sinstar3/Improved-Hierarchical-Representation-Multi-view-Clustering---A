"""
网络模块

包含以下功能：
1. 单隐藏层自编码器 (SingleLayerAE)
2. 完整自编码器 (Autoencoder)

主要用于 HRMC 模型中的特征提取和表示学习。
"""

from typing import List, Tuple, Optional, Dict, Any, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler


# ==============================================================================
# 单隐藏层自编码器
# ==============================================================================

class SingleLayerAE(nn.Module):
    """
    单隐藏层自编码器
    
    用于逐层贪婪预训练的基础构建块，每个实例包含一个编码器和一个解码器。
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        初始化单隐藏层自编码器
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 重建结果和潜在表示
        """
        h = self.activation(self.encoder(x))
        x_recon = self.decoder(h)
        return x_recon, h


# ==============================================================================
# 完整自编码器
# ==============================================================================

class Autoencoder(nn.Module):
    """
    用于每个视图的自编码器，学习样本的低维非线性表示。
    
    结构：输入层 -> 隐藏层 -> 编码层 -> 解码隐藏层 -> 输出层
    激活函数使用 Tanh
    
    功能：
    1. 特征提取：将高维输入数据压缩到低维潜在空间
    2. 重建：从低维表示重建原始输入
    
    应用：
    - 为每个视图学习独立的低维表示
    - 作为HRMC模型的特征提取器
    - 支持多隐藏层结构，适应不同复杂度的数据
    """
    def __init__(self, input_dim: int, hidden_dims: List[int] = [200, 100], latent_dim: int = 100):
        """
        初始化自编码器
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            latent_dim: 潜在表示维度
        """
        super().__init__()
        
        # 编码器：input -> hidden1 -> hidden2 -> ... -> latent
        encoder_layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hdim))
            encoder_layers.append(nn.Tanh())
            prev_dim = hdim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 解码器：latent -> hidden2 -> hidden1 -> input
        decoder_layers = []
        reversed_dims = hidden_dims[::-1][1:]
        prev_dec_dim = hidden_dims[-1]
        for hdim in reversed_dims:
            decoder_layers.append(nn.Linear(prev_dec_dim, hdim))
            decoder_layers.append(nn.Tanh())
            prev_dec_dim = hdim
        decoder_layers.append(nn.Linear(prev_dec_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 重建结果和潜在表示
        """
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent


# ==============================================================================
# 自编码器训练函数
# ==============================================================================
def pretrain_single_layer_ae(
    model: nn.Module,
    optimizer: optim.Optimizer,
    Xv: torch.Tensor,
    Cv: torch.Tensor,
    lambda1: float,
    lambda2: float,
    epochs: int,
    batch_size: int,
    early_stop: bool,
    early_stop_min_delta: float,
    M: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    预训练单层自编码器
    
    Args:
        model: 单层自编码器模型
        optimizer: 优化器
        Xv: 输入数据
        Cv: 系数矩阵 (m, N)
        epochs: 训练轮数
        lambda1: 网络权重L2正则化系数
        lambda2: 自表达损失权重
        early_stop: 是否使用早停
        early_stop_min_delta: 早停最小变化量
        M: 锚点索引 (m,)
        batch_size: mini-batch大小
        mask: 掩码，1表示存在，0表示缺失
    
    Returns:
        encoder_weights: 编码器权重
        encoder_bias: 编码器偏置
        decoder_weights: 解码器权重
        decoder_bias: 解码器偏置
        H: 编码后的表示
    """
    N = Xv.shape[0]
    # 初始化损失历史
    loss_history = []
    update_count = 0

    # 优化：提前缓存参数列表，避免每次循环都调用 model.parameters()
    param_list = list(model.parameters())
    
    for _ in range(epochs):
        # 随机打乱样本顺序
        indices = torch.randperm(N)
        # 每个epoch开始时计算一次所有样本的潜在表示（epoch开始时网络参数固定）
        with torch.no_grad():
            _, full_latent = model(Xv)
            if M.shape[0] != N:
                # 构建锚点矩阵 A 从锚点索引 M
                anchor_matrix = full_latent[M, :]
                # 向量化：一次性计算所有样本的 reconstructed_latent = (anchor_matrix.T @ Cv).T
                reconstructed_latents = (anchor_matrix.T @ Cv).T  # (N, hidden_dim)
            else:
                # 向量化：一次性计算所有样本的 reconstructed_latent = Cv.T @ full_latent
                reconstructed_latents = Cv @ full_latent  # (N, hidden_dim)
        # 使用mini-batch训练
        for i in range(0, N, batch_size):
            update_count += 1
            batch_indices = indices[i:i+batch_size]
            x_batch = Xv[batch_indices]  # mini-batch样本
            
            # 获取对应的掩码批次
            if mask is not None:
                mask_batch = mask[batch_indices]
            
            optimizer.zero_grad()
            recon, _ = model(x_batch)
            
            # 计算重建损失
            if mask is not None:
                # 应用掩码，只计算非缺失特征的损失
                diff = x_batch - recon
                masked_diff = diff * mask_batch
                # 计算非缺失特征的数量
                non_missing_count = mask_batch.sum()
                if non_missing_count > 0:
                    # 归一化损失，使其与无掩码时的量级相当
                    recon_loss = 0.5 * torch.sum(masked_diff ** 2) / non_missing_count * mask_batch.numel()
                else:
                    recon_loss = 0.0
            else:
                recon_loss = 0.5 * torch.sum((x_batch - recon) ** 2)
            
            # 计算锚点表达损失：使用预先计算好的 reconstructed_latents
            current_latent = full_latent[batch_indices]  # (batch_size, hidden_dim)
            reconstructed_latent = reconstructed_latents[batch_indices]  # (batch_size, hidden_dim)
            self_expr_loss = 0.5 * torch.sum((current_latent - reconstructed_latent) ** 2)
            self_expr_loss *= lambda2
            
            # 计算正则化损失
            reg_loss = sum(p.norm()**2 for p in param_list) * lambda1 * 0.5
            loss = recon_loss + self_expr_loss + reg_loss
            loss.backward()
            optimizer.step()
            
            loss_history.append(loss.item())
            
            # 早停检查
            if early_stop and len(loss_history) > 1:
                loss_diff = abs(loss_history[-1] - loss_history[-2])
                if loss_diff < early_stop_min_delta:
                    #print(f"convergent at the {update_count}-th update ({2} epoch)")
                    break
        
        if early_stop and len(loss_history) > 1 and abs(loss_history[-1] - loss_history[-2]) < early_stop_min_delta:
            break
    
    # 获取编码后的表示
    with torch.no_grad():
        _, H = model(Xv)
    
    return (model.encoder.weight.data.clone(), model.encoder.bias.data.clone(),
            model.decoder.weight.data.clone(), model.decoder.bias.data.clone(), H)

def update_single_ae(
    model: nn.Module,
    optimizer: optim.Optimizer,
    Xv: torch.Tensor,
    Cv: torch.Tensor,
    lambda1: float,
    lambda2: float,
    epochs: int,
    batch_size: int,
    early_stop: bool,
    early_stop_min_delta: float,
    M: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    scaler: Optional[GradScaler] = None
) -> Dict[str, torch.Tensor]:
    """
    更新单个自编码器
    
    Args:
        model: 自编码器
        optimizer: 优化器
        Xv: 输入数据
        Cv: 系数矩阵 (m, N)
        epochs: 训练轮数
        lambda1: 网络权重L2正则化系数
        lambda2: 自表达损失权重
        early_stop: 是否使用早停
        early_stop_min_delta: 早停最小变化量
        M: 锚点索引 (m,)
        batch_size: mini-batch大小
        mask: 掩码，1表示存在，0表示缺失
        scaler: GradScaler实例，用于混合精度训练
    
    Returns:
        更新后的自编码器状态字典
    """
    N = Xv.shape[0]
    # 初始化损失历史
    loss_history = []
    update_count = 0

    # 提前缓存参数列表，避免每次循环都调用 model.parameters()
    param_list = list(model.parameters())
    
    # 设置随机数生成器（只设置一次）
    for _ in range(epochs):
        # 随机打乱样本顺序
        indices = torch.randperm(N)
        # 每个epoch开始时计算一次所有样本的潜在表示（epoch开始时网络参数固定）
        with torch.no_grad():
            _, full_latent = model(Xv)
            if M.shape[0] != N:
                # 构建锚点矩阵 A 从锚点索引 M
                anchor_matrix = full_latent[M, :]
                # 向量化：一次性计算所有样本的 reconstructed_latent = (anchor_matrix.T @ Cv).T
                reconstructed_latents = (anchor_matrix.T @ Cv).T  # (N, hidden_dim)
            else:
                # 向量化：一次性计算所有样本的 reconstructed_latent = Cv.T @ full_latent
                reconstructed_latents = Cv @ full_latent  # (N, hidden_dim)
            

        # 使用mini-batch训练
        for i in range(0, N, batch_size):
            update_count += 1
            batch_indices = indices[i:i+batch_size]
            x_batch = Xv[batch_indices]  # mini-batch样本
            
            # 获取对应的掩码批次
            if mask is not None:
                mask_batch = mask[batch_indices]
            
            optimizer.zero_grad()
            
            # 使用混合精度训练
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    recon, _ = model(x_batch)
                    
                    # 计算重建损失
                    if mask is not None:
                        # 应用掩码，只计算非缺失特征的损失
                        diff = x_batch - recon
                        masked_diff = diff * mask_batch
                        # 计算非缺失特征的数量
                        non_missing_count = mask_batch.sum()
                        if non_missing_count > 0:
                            # 归一化损失，使其与无掩码时的量级相当
                            recon_loss = 0.5 * torch.sum(masked_diff ** 2) / non_missing_count * mask_batch.numel()
                        else:
                            recon_loss = 0.0
                    else:
                        recon_loss = 0.5 * torch.sum((x_batch - recon) ** 2)
                    
                    # 计算表达损失
                    current_latent = full_latent[batch_indices]  # (batch_size, hidden_dim)
                    reconstructed_latent = reconstructed_latents[batch_indices]  # (batch_size, hidden_dim)
                    self_expr_loss = 0.5 * torch.sum((current_latent - reconstructed_latent) ** 2)
                    self_expr_loss *= lambda2
                    
                    # 计算正则化损失
                    reg_loss = sum(p.norm()**2 for p in param_list) * lambda1 * 0.5
                    
                    loss = recon_loss + self_expr_loss + reg_loss
                
                # 缩放梯度并更新参数
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 常规精度训练
                recon, _ = model(x_batch)
                
                # 计算重建损失
                if mask is not None:
                    # 应用掩码，只计算非缺失特征的损失
                    diff = x_batch - recon
                    masked_diff = diff * mask_batch
                    # 计算非缺失特征的数量
                    non_missing_count = mask_batch.sum()
                    if non_missing_count > 0:
                        # 归一化损失，使其与无掩码时的量级相当
                        recon_loss = 0.5 * torch.sum(masked_diff ** 2) / non_missing_count * mask_batch.numel()
                    else:
                        recon_loss = 0.0
                else:
                    recon_loss = 0.5 * torch.sum((x_batch - recon) ** 2)
                
                # 计算表达损失
                current_latent = full_latent[batch_indices]  # (batch_size, hidden_dim)
                reconstructed_latent = reconstructed_latents[batch_indices]  # (batch_size, hidden_dim)
                self_expr_loss = 0.5 * torch.sum((current_latent - reconstructed_latent) ** 2)
                self_expr_loss *= lambda2
                
                # 计算正则化损失
                reg_loss = sum(p.norm()**2 for p in param_list) * lambda1 * 0.5
                
                loss = recon_loss + self_expr_loss + reg_loss
                loss.backward()
                optimizer.step()
            
            loss_history.append(loss.item())
            
            # 早停检查
            if early_stop and len(loss_history) > 1:
                loss_diff = abs(loss_history[-1] - loss_history[-2])
                if loss_diff < early_stop_min_delta:
                    #print(f"convergent at the {update_count}-th update ({2} epoch)")
                    break
        
        if early_stop and len(loss_history) > 1 and abs(loss_history[-1] - loss_history[-2]) < early_stop_min_delta:
            break
    
    return model.state_dict()
