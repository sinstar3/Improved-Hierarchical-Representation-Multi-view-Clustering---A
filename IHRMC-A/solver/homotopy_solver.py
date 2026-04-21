"""
同伦求解器模块

用于求解 L1 正则化问题
"""

import numpy as np


def solve_homotopy(A, b, lambda_, max_iter=600, tol=1e-3, non_negative=False):
    """
    求解 L1 正则化问题：min_x lambda * ||x||_1 + 1/2 * ||Ax - b||_2^2
    
    Args:
        A: 数据矩阵，形状 (d, n)
        b: 目标向量，形状 (d,)
        lambda_: 正则化参数
        max_iter: 最大迭代次数
        tol: 收敛阈值
        non_negative: 是否使用非负约束
    
    Returns:
        x: 稀疏编码结果，形状 (n,)
    """
    m, n = A.shape
    
    # 初始化变量
    x = np.zeros(n)
    z_x = np.zeros(n)
    gamma_x = []  # 活动集
    
    # 计算初始梯度
    Primal_constrk = -A.T @ b
    
    # 找到初始活动集
    if non_negative:
        idx = np.argmin(Primal_constrk)
        c = max(-Primal_constrk[idx], 0)
    else:
        idx = np.argmax(np.abs(Primal_constrk))
        c = np.abs(Primal_constrk[idx])
    
    epsilon = c
    gamma_xk = [idx]
    
    # 计算初始目标函数值
    f = epsilon * np.linalg.norm(x, 1) + 0.5 * np.linalg.norm(b - A @ x) ** 2
    z_x[gamma_xk] = -np.sign(Primal_constrk[gamma_xk])
    z_xk = z_x.copy()
    
    # 计算 AtgxAgx 和 iAtgxAgx
    A_active = A[:, gamma_xk]
    AtgxAgx = A_active.T @ A_active
    if np.linalg.matrix_rank(AtgxAgx) < len(gamma_xk):
        AtgxAgx += 1e-3 * np.eye(len(gamma_xk))
    iAtgxAgx = np.linalg.inv(AtgxAgx)
    
    # 主循环
    iter_count = 0
    old_delta = 0
    count_delta_stop = 0
    out_x = []
    
    while iter_count < max_iter:
        iter_count += 1
        
        # 更新变量
        gamma_x = gamma_xk.copy()
        z_x = z_xk.copy()
        x_k = x.copy()
        
        # 计算搜索方向
        del_x = iAtgxAgx @ z_x[gamma_x]
        del_x_vec = np.zeros(n)
        del_x_vec[gamma_x] = del_x
        
        # 计算 dk
        Asupported = A[:, gamma_x]
        Agdelx = Asupported @ del_x
        dk = A.T @ Agdelx
        
        # 处理机器精度误差
        pk_temp = Primal_constrk.copy()
        gammaL_temp = np.where(np.abs(np.abs(Primal_constrk) - epsilon) < min(epsilon, 2 * np.finfo(float).eps))[0]
        pk_temp[gammaL_temp] = np.sign(Primal_constrk[gammaL_temp]) * epsilon
        
        xk_temp = x_k.copy()
        xk_temp[np.abs(x_k) < 2 * np.finfo(float).eps] = 0
        
        # 计算步长
        def update_primal(out_x, gamma_x, z_x, xk_temp, del_x_vec, pk_temp, dk, epsilon, n, non_negative):
            gamma_lc = np.setdiff1d(np.arange(n), np.union1d(gamma_x, out_x))
            
            if non_negative:
                delta1 = np.inf
            else:
                # 检查分母是否为零
                denominator = 1 + dk[gamma_lc]
                non_zero_mask = denominator != 0
                if np.any(non_zero_mask):
                    delta1_constr = np.full_like(gamma_lc, np.inf, dtype=np.float64)
                    delta1_constr[non_zero_mask] = (epsilon - pk_temp[gamma_lc[non_zero_mask]]) / denominator[non_zero_mask]
                    delta1_pos_ind = np.where(delta1_constr > 0)[0]
                    if len(delta1_pos_ind) > 0:
                        delta1 = np.min(delta1_constr[delta1_pos_ind])
                    else:
                        delta1 = np.inf
                else:
                    delta1 = np.inf
            
            # 避免除零错误
            denominator = 1 - dk[gamma_lc]
            # 检查分母是否为零
            non_zero_mask = denominator != 0
            if np.any(non_zero_mask):
                delta2_constr = np.full_like(gamma_lc, np.inf, dtype=np.float64)
                delta2_constr[non_zero_mask] = (epsilon + pk_temp[gamma_lc[non_zero_mask]]) / denominator[non_zero_mask]
                delta2_pos_ind = np.where(delta2_constr > 0)[0]
                if len(delta2_pos_ind) > 0:
                    delta2 = np.min(delta2_constr[delta2_pos_ind])
                else:
                    delta2 = np.inf
            else:
                delta2 = np.inf
            
            if delta1 > delta2:
                delta = delta2
                i_delta = gamma_lc[delta2_pos_ind[np.argmin(delta2_constr[delta2_pos_ind])]]
            else:
                delta = delta1
                if not non_negative:
                    i_delta = gamma_lc[delta1_pos_ind[np.argmin(delta1_constr[delta1_pos_ind])]]
                else:
                    i_delta = -1
            
            # 计算 delta3
            if len(gamma_x) > 0:
                # 检查分母是否为零，添加小 epsilon 避免除零
                denominator = del_x_vec[gamma_x]
                # 添加小 epsilon 避免除零
                # 当分母接近零时，使用一个固定的小值
                epsilon = 1e-10
                denominator = np.where(np.abs(denominator) < epsilon, epsilon, denominator)
                # 保持符号
                denominator = np.sign(del_x_vec[gamma_x]) * np.abs(denominator)
                # 避免分母为零
                denominator = np.where(denominator == 0, epsilon, denominator)
                delta3_constr = -xk_temp[gamma_x] / denominator
                delta3_pos_index = np.where(delta3_constr > 0)[0]
                if len(delta3_pos_index) > 0:
                    delta3 = np.min(delta3_constr[delta3_pos_index])
                    out_x_index = gamma_x[delta3_pos_index[np.argmin(delta3_constr[delta3_pos_index])]]
                else:
                    delta3 = np.inf
                    out_x_index = -1
            else:
                delta3 = np.inf
                out_x_index = -1
            
            out_x = []
            if delta3 > 0 and delta3 <= delta:
                delta = delta3
                out_x = [out_x_index]
            
            # 处理退化情况
            xk_1 = xk_temp + delta * del_x_vec
            if out_x:
                xk_1[out_x] = 0
            
            wrong_sign = []
            if len(gamma_x) > 0:
                wrong_sign = np.where(np.sign(xk_1[gamma_x]) * z_x[gamma_x] == -1)[0]
                if non_negative:
                    neg_idx = np.where(xk_1[gamma_x] < 0)[0]
                    wrong_sign = np.union1d(wrong_sign, neg_idx)
            
            if len(wrong_sign) > 0:
                delta = 0
                val_wrong_x = np.abs(del_x_vec[np.array(gamma_x)[wrong_sign]])
                ind_wrong_x = np.argsort(val_wrong_x)[::-1]
                out_x = [gamma_x[wrong_sign[ind_wrong_x[0]]]]
            
            # 检查是否有多个元素想进入活动集
            if len(gamma_lc) > 0:
                i_delta_temp = gamma_lc[np.where(np.abs(pk_temp[gamma_lc] + delta * dk[gamma_lc]) - (epsilon - delta) >= 10 * np.finfo(float).eps)[0]]
                if len(i_delta_temp) >= 1:
                    if not np.isin(i_delta, i_delta_temp):
                        # 检查除数是否为零
                        non_zero_mask = dk[i_delta_temp] != 0
                        if np.any(non_zero_mask):
                            i_delta_temp_non_zero = i_delta_temp[non_zero_mask]
                            v_temp = -pk_temp[i_delta_temp_non_zero] / dk[i_delta_temp_non_zero]
                            if len(v_temp) > 0:
                                i_temp = np.argmax(v_temp)
                                i_delta = i_delta_temp_non_zero[i_temp]
                                delta = 0
                                out_x = []
            
            return i_delta, delta, out_x
        
        i_delta, delta, out_x = update_primal(out_x, gamma_x, z_x, xk_temp, del_x_vec, pk_temp, dk, epsilon, n, non_negative)
        
        # 检查是否卡住
        if old_delta < 4 * np.finfo(float).eps and delta < 4 * np.finfo(float).eps:
            count_delta_stop += 1
            if count_delta_stop >= 500:
                break
        else:
            count_delta_stop = 0
        old_delta = delta
        
        # 更新参数
        xk_1 = x_k + delta * del_x_vec
        Primal_constrk = Primal_constrk + delta * dk
        epsilon_old = epsilon
        epsilon = epsilon - delta
        
        # 检查是否达到目标 lambda
        if epsilon <= lambda_:
            # 计算正确的步长
            alpha = epsilon_old - lambda_
            xk_1 = x_k + alpha * del_x_vec
            # 更新 x
            x = xk_1
            break
        
        # 计算停止准则
        keep_going = True
        if delta != 0:
            # 目标函数值停止准则
            prev_f = f
            Asupported = A[:, gamma_x]
            f = lambda_ * np.linalg.norm(xk_1, 1) + 0.5 * np.linalg.norm(b - Asupported @ xk_1[gamma_x]) ** 2
            keep_going = (abs((prev_f - f) / prev_f) > tol)
        
        # 检查是否迭代卡住
        if keep_going and np.linalg.norm(xk_1 - x_k) < 100 * np.finfo(float).eps:
            keep_going = False
        
        if not keep_going:
            break
        
        # 更新活动集
        if out_x:
            # 从活动集中移除元素
            len_gamma = len(gamma_x)
            outx_index = gamma_x.index(out_x[0])
            gamma_x[outx_index], gamma_x[-1] = gamma_x[-1], gamma_x[outx_index]
            gamma_x = gamma_x[:-1]
            gamma_xk = gamma_x.copy()
            
            # 更新 AtgxAgx 和 iAtgxAgx
            rowi = outx_index
            colj = outx_index
            AtgxAgx_ij = AtgxAgx.copy()
            # 交换行
            AtgxAgx_ij[[rowi, len_gamma-1], :] = AtgxAgx_ij[[len_gamma-1, rowi], :]
            # 交换列
            AtgxAgx_ij[:, [colj, len_gamma-1]] = AtgxAgx_ij[:, [len_gamma-1, colj]]
            
            iAtgxAgx_ij = iAtgxAgx.copy()
            # 交换行
            iAtgxAgx_ij[[colj, len_gamma-1], :] = iAtgxAgx_ij[[len_gamma-1, colj], :]
            # 交换列
            iAtgxAgx_ij[:, [rowi, len_gamma-1]] = iAtgxAgx_ij[:, [len_gamma-1, rowi]]
            
            AtgxAgx = AtgxAgx_ij[:-1, :-1]
            
            # 更新逆矩阵
            n_inv = AtgxAgx_ij.shape[0]
            Q11 = iAtgxAgx_ij[:-1, :-1]
            Q12 = iAtgxAgx_ij[:-1, -1]
            Q21 = iAtgxAgx_ij[-1, :-1]
            Q22 = iAtgxAgx_ij[-1, -1]
            Q12Q21_Q22 = Q12 * (Q21 / Q22)
            iAtgxAgx = Q11 - Q12Q21_Q22
            
            # 设置移除元素的值为0
            xk_1[out_x[0]] = 0
        else:
            # 添加元素到活动集
            gamma_xk = gamma_x.copy()
            if i_delta != -1:
                gamma_xk.append(i_delta)
            new_x = i_delta
            
            # 更新 AtgxAgx 和 iAtgxAgx
            if len(gamma_x) > 0:
                # 确保 new_x 是整数，并且 A[:, new_x] 是二维矩阵
                new_x_idx = int(new_x)
                A_new_x = A[:, new_x_idx]
                # 确保 A_new_x 是二维矩阵
                if len(A_new_x.shape) == 1:
                    A_new_x = A_new_x.reshape(-1, 1)
                AtgxAnx = A[:, gamma_x].T @ A_new_x
                AtgxAgx_mod = np.vstack([
                    np.hstack([AtgxAgx, AtgxAnx]),
                    np.hstack([AtgxAnx.T, A_new_x.T @ A_new_x])
                ])
                AtgxAgx = AtgxAgx_mod
                
                # 更新逆矩阵
                n_inv = AtgxAgx.shape[0]
                iA11 = iAtgxAgx
                iA11A12 = iA11 @ AtgxAgx[:-1, -1]
                A21iA11 = AtgxAgx[-1, :-1] @ iA11
                S = AtgxAgx[-1, -1] - AtgxAgx[-1, :-1] @ iA11A12
                
                if S == 0:
                    Q11_right = iA11A12 * (A21iA11 * 0)
                    iAtgxAgx = np.zeros((n_inv, n_inv))
                    iAtgxAgx[:-1, :-1] = iA11 + Q11_right
                    iAtgxAgx[:-1, -1] = -iA11A12 * S
                    iAtgxAgx[-1, :-1] = -A21iA11 * S
                    iAtgxAgx[-1, -1] = S
                else:
                    Q11_right = iA11A12 * (A21iA11 / S)
                    iAtgxAgx = np.zeros((n_inv, n_inv))
                    iAtgxAgx[:-1, :-1] = iA11 + Q11_right
                    iAtgxAgx[:-1, -1] = -iA11A12 / S
                    iAtgxAgx[-1, :-1] = -A21iA11 / S
                    iAtgxAgx[-1, -1] = 1 / S
            
            # 设置新元素的值为0
            if i_delta != -1:
                xk_1[i_delta] = 0
        
        # 更新 z_xk
        z_xk = np.zeros(n)
        if gamma_xk:
            z_xk[gamma_xk] = -np.sign(Primal_constrk[gamma_xk])
        Primal_constrk[gamma_x] = np.sign(Primal_constrk[gamma_x]) * epsilon
        
        # 更新 x 和 grad
        x = xk_1
        Primal_constrk = -A.T @ (b - A @ x)
    
    return x