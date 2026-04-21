"""
消融实验

验证各核心改进模块对算法性能的独立贡献与协同作用

实验设置：
1. HRMC：未加入任何改进模块的原始算法，作为性能基线
2. HRMC+锚点机制：仅在原始HRMC中引入锚点优化模块
3. HRMC+掩码机制：仅在原始HRMC中加入视图掩码模块
4. IHRMC-A：同时集成锚点机制与掩码机制的最终改进模型

数据集：BBC，VMR=0.7
"""

import os
import pandas as pd
import yaml
import torch

from agent.IHRMC_A import IHRMC_A
from data.dataloader import DataLoader
from utils.utils import clustering_metrics

def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'default_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def run_experiment(config, algorithm_name, params, use_anchor, use_adaptive_weight, use_mask, beta_max, iter_t,  VMR, cv_init_method, runs=10):
    """运行指定算法多次并记录结果"""
    results = []
    
    # 加载数据
    dataset_name = config['dataset']['name']
    normalize_method = config['dataset']['normalize_method']
    
    data_loader = DataLoader(
        dataset_name=dataset_name,
        normalize_method=normalize_method
    )
    
    # 加载数据
    if VMR > 0:
        X, Y, missing_mask = data_loader.load_dataset(VMR, 42)
    else:
        X, Y = data_loader.load_dataset()
    
    n_clusters = len(torch.unique(Y))
    
    for run in range(1, runs + 1):
        print(f"\n[{algorithm_name}] VMR={VMR} 运行 {run}/{runs}...")
        
        model = IHRMC_A(
            n_clusters=n_clusters,
            params=params,
            m=config['model']['m'],
            max_iters=config['model']['max_iters'],
            T=config['model']['T'],
            iter_t=iter_t,
            batch_size=config['model']['batch_size'],
            random_seed=None,
            lr=config['model']['lr'],
            epochs=config['model']['epochs'],
            device='cuda' if torch.cuda.is_available() else 'cpu',
            hidden_dims=config['model']['hidden_dims'],
            latent_dim=config['model']['latent_dim'],
            beta_max=beta_max,
            tol=config['model']['tol'],
            use_anchor=use_anchor,
            use_adaptive_weight=use_adaptive_weight,
            lightweight_mode=config['model']['lightweight_mode'],
            use_pre_cache=config['model']['use_pre_cache'],
            cv_init_method=cv_init_method,
        )
        
        # 训练模型
        if VMR > 0 and use_mask == True:
            labels, history = model.fit(X, Y, missing_mask)
        else:
            labels, history = model.fit(X, Y)
        
        # 计算指标
        metrics = clustering_metrics(Y, labels)
        
        # 记录结果
        result = {
            '算法': algorithm_name,
            '运行次数': run,
            '数据集': dataset_name,
            'VMR': VMR,
            'ACC': metrics['ACC'],
            'NMI': metrics['NMI'],
            'ARI': metrics['ARI'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'F1': metrics['F-score'],
            '运行时间(s)': history.get('run_time', 0),
            '最大内存(MB)': history.get('max_memory_allocated', 0),
        }
        results.append(result)
        
        # 清理内存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results

def main():
    """主函数"""
    # 加载配置
    config = load_config()
    
    # 设置数据集和VMR
    config['dataset']['name'] = '100leaves'
    VMR = 0.3
    
    print("=" * 60)
    print(f"消融实验 - 100leaves 数据集，VMR={VMR}")
    print("=" * 60)
    
    # 算法参数
    # HRMC: use_anchor=False, use_adaptive_weight=False
    # HRMC+锚点: use_anchor=True, use_adaptive_weight=False
    # HRMC+掩码: use_anchor=False, use_adaptive_weight=False (使用掩码处理缺失)
    # IHRMC-A: use_anchor=True, use_adaptive_weight=True
    
    hrmc_params = [0.001,0.07,1.0,0.001]
    ihrmc_params = [0.001,0.07,1.0,0.001]
    
    all_results = []
    
    # 1. HRMC 基线
    print("\n" + "=" * 60)
    print("1. HRMC (基线)")
    print("=" * 60)
    hrmc_results = run_experiment(
        config, 'HRMC', hrmc_params,
        use_anchor=False, use_adaptive_weight=False, use_mask = False,
        beta_max=10e10, iter_t=20, VMR=VMR, cv_init_method=0
    )
    all_results.extend(hrmc_results)
    
    # 2. HRMC + 锚点机制
    print("\n" + "=" * 60)
    print("2. HRMC + 锚点机制")
    print("=" * 60)
    hrmc_anchor_results = run_experiment(
        config, 'HRMC+锚点', hrmc_params,
        use_anchor=True, use_adaptive_weight=True, use_mask = False,
        iter_t=10, beta_max=10, VMR=VMR, cv_init_method=0
    )
    all_results.extend(hrmc_anchor_results)
    
    # 3. HRMC + 掩码机制
    print("\n" + "=" * 60)
    print("3. HRMC + 掩码机制")
    print("=" * 60)
    hrmc_mask_results = run_experiment(
        config, 'HRMC+掩码', hrmc_params,
        use_anchor=False, use_adaptive_weight=False, use_mask = True,
        iter_t=20, beta_max=10e10, VMR=VMR, cv_init_method=1
    )
    all_results.extend(hrmc_mask_results)
    
    # 4. IHRMC-A (完整模型)
    print("\n" + "=" * 60)
    print("4. IHRMC-A (完整模型)")
    print("=" * 60)
    ihrmc_results = run_experiment(
        config, 'IHRMC-A', ihrmc_params,
        use_anchor=True, use_adaptive_weight=True, use_mask = True,
        iter_t=10, beta_max=10, VMR=VMR, cv_init_method=0
    )
    all_results.extend(ihrmc_results)
    
    # 转换为 DataFrame
    df = pd.DataFrame(all_results)
    
    # 确保目录存在
    output_dir = os.path.join(os.path.dirname(__file__), 'results', 'ablation')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存到 Excel
    output_file = os.path.join(output_dir, f"ablation_bbc_vmr{VMR}.xlsx")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='详细结果', index=False)
    
    print(f"\n结果已保存到: {output_file}")
    return output_file

if __name__ == "__main__":
    main()