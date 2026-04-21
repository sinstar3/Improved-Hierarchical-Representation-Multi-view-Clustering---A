#!/usr/bin/env python3
"""
参数敏感性分析实验

分析各超参数对算法性能的影响
数据集：MSRC
参数范围：
- lambda1: 0.0001 - 0.1 (分段线性增长)
- lambda2: 0.001 - 1 (分段线性增长)
- lambda3: 0.001 - 1 (分段线性增长)
- beta: 0.0001 - 2 (分段线性增长)
- m: 从c(聚类数)到n(样本数)，等距c个值

使用方法：
1. 修改下方的 PARAM_TO_ANALYZE 变量选择要分析的参数
2. 运行：python sensitivity_test.py
"""

import os
import numpy as np
import pandas as pd
import yaml
import torch

from agent.IHRMC_A import IHRMC_A
from data.dataloader import DataLoader
from utils.utils import clustering_metrics, set_seed

def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'default_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def generate_param_values_piecewise(start, end):
    """
    生成分段线性的参数值（在对数空间中按固定步长增长）

    步长规则：
    - 0.0001 - 0.001: step = 0.0001
    - 0.001 - 0.01: step = 0.001
    - 0.01 - 0.1: step = 0.005
    - 0.1 - 1: step = 0.05
    - 1 - 10: step = 0.5
    - 10+: step = 1
    """
    values = []
    v = start

    while v <= end:
        values.append(v)

        # 根据当前值的范围确定步长
        if v < 0.001:
            step = 0.0001
        elif v < 0.01:
            step = 0.001
        elif v < 0.1:
            step = 0.005
        elif v < 1:
            step = 0.05
        elif v < 10:
            step = 0.5
        else:
            step = 1.0

        v = v + step
        # 避免浮点精度问题
        v = round(v, 10)

    return np.array(values)

def run_single_experiment(config, X, Y, missing_mask, params, m, runs=1):
    """运行单次实验"""
    results = []

    n_clusters = len(torch.unique(Y))

    for run in range(1, runs + 1):
        model = IHRMC_A(
            n_clusters=n_clusters,
            params=params,
            m=m,
            max_iters=config['model']['max_iters'],
            T=config['model']['T'],
            iter_t=config['model']['iter_t'],
            batch_size=config['model']['batch_size'],
            random_seed=42,
            lr=config['model']['lr'],
            epochs=config['model']['epochs'],
            device='cuda' if torch.cuda.is_available() else 'cpu',
            hidden_dims=config['model']['hidden_dims'],
            latent_dim=config['model']['latent_dim'],
            beta_max=10,
            tol=config['model']['tol'],
            use_anchor=True,
            use_adaptive_weight=True,
            lightweight_mode=config['model']['lightweight_mode'],
            use_pre_cache=config['model']['use_pre_cache'],
            cv_init_method=1,
        )

        if missing_mask is not None:
            labels, _ = model.fit(X, Y, missing_mask)
        else:
            labels, _ = model.fit(X, Y)

        metrics = clustering_metrics(Y, labels)

        result = {
            'lambda1': params[0],
            'lambda2': params[1],
            'lambda3': params[2],
            'beta': params[3],
            'm': m,
            'ACC': metrics['ACC'],
            'NMI': metrics['NMI'],
            'ARI': metrics['ARI'],
            'F1': metrics['F-score'],
        }
        results.append(result)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results

def analyze_parameter(param_name, config, dataset_name, VMR=0, runs=1):
    """分析单个参数的变化"""
    print(f"\n" + "=" * 60)
    print(f"分析参数: {param_name}")
    print("=" * 60)

    # 加载数据（只加载一次）
    normalize_method = config['dataset']['normalize_method']
    data_loader = DataLoader(
        dataset_name=dataset_name,
        normalize_method=normalize_method
    )

    if VMR > 0:
        X, Y, missing_mask = data_loader.load_dataset(VMR, 42)
    else:
        X, Y = data_loader.load_dataset()
        missing_mask = None

    n_clusters = len(torch.unique(Y))
    n_samples = X[0].shape[0]

    # 基准参数
    base_params = [0.001, 0.07, 0.1, 0.0001]

    # 定义各参数的范围
    param_ranges = {
        'lambda1': (0.0001, 0.1),
        'lambda2': (0.001, 1.0),
        'lambda3': (0.001, 1.0),
        'beta': (0.0001, 2.0),
    }

    if param_name == 'm':
        # 从c到n以c为间隔生成值
        param_values = np.arange(n_clusters, n_samples + 1, n_clusters).astype(int)
    else:
        start, end = param_ranges[param_name]
        param_values = generate_param_values_piecewise(start, end)

    print(f"参数范围: {param_values[0]:.6f} 到 {param_values[-1]:.6f}")
    print(f"参数数量: {len(param_values)}")
    print(f"参数值: {param_values}")

    results = []
    for i, value in enumerate(param_values):
        print(f"  [{i+1}/{len(param_values)}] {param_name} = {value:.6f}")

        if param_name == 'lambda1':
            params = [value, base_params[1], base_params[2], base_params[3]]
            m = 70
        elif param_name == 'lambda2':
            params = [base_params[0], value, base_params[2], base_params[3]]
            m = 70
        elif param_name == 'lambda3':
            params = [base_params[0], base_params[1], value, base_params[3]]
            m = 70
        elif param_name == 'beta':
            params = [base_params[0], base_params[1], base_params[2], value]
            m = 70
        elif param_name == 'm':
            params = base_params
            m = int(value)

        res = run_single_experiment(config, X, Y, missing_mask, params, m, runs)

        for r in res:
            r['参数'] = param_name
            r['参数值'] = value
        results.extend(res)

    # 保存结果
    output_dir = os.path.join(os.path.dirname(__file__), 'results', 'sensitivity')
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"sensitivity_{dataset_name}_vmr{VMR}_{param_name}.xlsx")
    df = pd.DataFrame(results)

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='详细结果', index=False)

    print(f"\n结果已保存到: {output_file}")

    return output_file

if __name__ == "__main__":
    """主函数"""
    # ========== 配置部分 ==========
    # 要分析的参数: lambda1, lambda2, lambda3, beta, m
    PARAM_TO_ANALYZE = 'm'  # <-- 修改这里选择要分析的参数
    DATASET_NAME = 'MSRC'          # 数据集名称
    VMR = 0.1                        # 视图缺失率
    RUNS = 1                       # 每个参数的运行次数
    # ============================

    config = load_config()

    # 设置随机种子
    seed = 42
    set_seed(seed=seed)

    print("=" * 60)
    print(f"参数敏感性分析实验 - {DATASET_NAME} 数据集")
    print(f"VMR: {VMR}")
    print("=" * 60)

    # 分析指定参数
    analyze_parameter(PARAM_TO_ANALYZE, config, DATASET_NAME, VMR, RUNS)