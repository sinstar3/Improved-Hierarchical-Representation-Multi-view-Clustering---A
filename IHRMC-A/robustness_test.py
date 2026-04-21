
"""
验证算法对视图缺失的鲁棒性

运行步骤：
1. 设置VMR从0.1到0.7，步长0.1
2. 在每个VMR下运行HRMC和IHRMC-A算法
3. 每个算法运行10次，记录结果到Excel
4. 分析两种算法在不同缺失率下的性能变化趋势
"""

import os
import pandas as pd
import yaml
import torch

from agent.IHRMC_A import IHRMC_A
from data.dataloader import DataLoader
from utils import clustering_metrics

def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'default_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def run_algorithm(config, algorithm_name, params, use_anchor, use_adaptive_weight, use_mask, beta_max, iter_t, VMR, cv_init_method, m=None, runs=1):
    """运行指定算法多次并记录结果"""
    results = []
    # 加载数据
    dataset_name = config['dataset']['name']
    normalize_method = config['dataset']['normalize_method']
    # 使用 DataLoader 加载数据
    data_loader = DataLoader(
        dataset_name=dataset_name,
        normalize_method=normalize_method
    )
    if VMR > 0:
        X, Y, missing_mask = data_loader.load_dataset(VMR, 42)
    else:
        X, Y = data_loader.load_dataset()  # VMR=0，完整视图
    n_clusters = len(torch.unique(Y))
    m = m if m is not None else config['model']['m']
    for run in range(1, runs + 1):
        print(f"\n[{algorithm_name}] VMR={VMR} 运行 {run}/{runs}...")
        model = IHRMC_A(
            n_clusters=n_clusters,
            params=params,
            m=m,
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
            'VMR': VMR,
            '运行次数': run,
            '数据集': dataset_name,
            '锚点数': m,
            'ACC': metrics['ACC'],
            'NMI': metrics['NMI'],
            'ARI': metrics['ARI'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'F1': metrics['F-score'],
            '运行时间(s)': history.get('run_time', 0),
            '最大内存使用(MB)': history.get('max_memory_allocated', 0),
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
    
    # 设置数据集为BBC
    config['dataset']['name'] = 'BBC'
    
    # 算法参数（针对BBC数据集）
    hrmc_params = [0.001, 0.05, 1.0, 0.1]  # HRMC
    ihrmc_params = [0.001, 0.1, 1.0, 0.1]   # IHRMC-A
    
    # 视图缺失率范围
    vmr_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    all_results = []
    
    # 对每个VMR值运行实验
    for VMR in vmr_values:
        print(f"\n=== 处理 VMR={VMR} ===")
        
        # 运行 HRMC 算法
        print("\n--- 运行 HRMC 算法 ---")
        hrmc_results = run_algorithm(
            config, 'HRMC', hrmc_params, 
            use_anchor=False, use_adaptive_weight=False, use_mask = False, 
            beta_max=10e10, iter_t=20, VMR=VMR, cv_init_method=0
        )
        all_results.extend(hrmc_results)
        '''
        # 运行 IHRMC-A 算法
        print("\n--- 运行 IHRMC-A 算法 ---")
        if VMR < 0.45:
            m = 70
        else:
            m = 300
        ihrmc_results = run_algorithm(
            config, 'IHRMC-A', ihrmc_params, 
            use_anchor=True, use_adaptive_weight=True, use_mask = True, 
            beta_max=10, VMR=VMR, cv_init_method=1, m=m
        )
        all_results.extend(ihrmc_results)
        '''
    # 转换为 DataFrame
    df = pd.DataFrame(all_results)
    
    # 确保 results/robustness 目录存在
    output_dir = os.path.join(os.path.dirname(__file__), 'results', 'robustness')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存到 Excel
    output_file = os.path.join(output_dir, f"IHRMC_A_bbc_vmr_analysis.xlsx")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='详细结果', index=False)
    
    print(f"\n结果已保存到: {output_file}")
    return output_file

if __name__ == "__main__":
    main()