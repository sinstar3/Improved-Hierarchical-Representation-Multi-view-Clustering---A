"""
IHRMC-A 主脚本

运行 IHRMC-A 实验，支持：
1. 单次实验运行
2. 多数据集支持
3. 配置管理
4. 设备自动检测
5. 结果汇总
"""

import os
# 避免Windows上的KMeans内存泄漏
os.environ['OMP_NUM_THREADS'] = '1'

from datetime import datetime

import numpy as np
import torch

from agent import IHRMC_A
from config import load_config, parse_cli_args
from data import DataLoader
from draw import Visualizer
from utils import clustering_metrics, get_device, set_seed
from utils.output_manager import get_output_dir, generate_exp_dir_name, rename_output_dir


def run_experiment(config):
    """
    运行单次实验
    
    Args:
        config: 配置对象
    
    Returns:
        metrics: 评估指标字典
        output_dir: 输出目录
    """
    # 设置随机种子
    seed = config.get('model.random_seed', None)
    set_seed(seed=seed)
    
    # 获取数据集名称
    dataset_name = config.get('dataset.name', 'MSRC')
    normalize_method = config.get('dataset.normalize_method', None)
    
    # 创建输出目录
    output_dir = get_output_dir(
        'single', 
        prefix='exp', 
        create_dir=True, 
        dataset_name=dataset_name
    )
    print(f"实验输出目录: {output_dir}")
    print(f"将运行的数据集: {dataset_name}")
    
    # 加载数据
    print(f"\n加载数据集: {dataset_name}")
    data_loader = DataLoader(
        dataset_name=dataset_name,
        normalize_method=normalize_method
    )
    # 获取缺失数据参数
    missing_rate = config.get('dataset.missing_rate', 0.0)
    missing_mode = config.get('dataset.missing_mode', 'sample')
    use_mask = config.get('model.use_mask', True)
    # 使用模型的随机种子作为缺失数据生成的种子
    missing_seed = seed
    
    # 获取设备和模型参数
    device = config.get('training.device', None) or get_device()
    model_params = config.get_model_params()
    model_params['device'] = device
    
    print(f"使用设备: {device}")
    print(f"模型参数: {model_params}")
    
    if missing_rate > 0:
        X, Y, masks = data_loader.load_dataset(missing_rate=missing_rate, seed=missing_seed, missing_mode=missing_mode)
        print(f"生成缺失数据，缺失率: {missing_rate}，模式: {missing_mode}")
    else:
        X, Y = data_loader.load_dataset()
    
    n_clusters = len(torch.unique(Y))
    print(f"数据集加载完成，样本数: {X[0].shape[0]}, 视图数: {len(X)}, 聚类数: {n_clusters}")
    
    # 提取核心超参数 [lambda1, lambda2, lambda3, beta]
    params = [
        model_params.get('lambda1', 0.001),  # 网络权重L2正则化系数
        model_params.get('lambda2', 0.1),    # 自表达损失权重
        model_params.get('lambda3', 1.0),    # 张量核范数权重
        model_params.get('beta', 0.1)        # ADMM 惩罚参数初始值
    ]
    
    # 创建IHRMC-A模型
    model = IHRMC_A(
        n_clusters=n_clusters,
        params=params,
        m=model_params.get('m', 28),  # 锚点数
        max_iters=model_params.get('max_iters', 100),  # ADMM最大迭代次数
        T=model_params.get('T', 20),  # 自适应权重预热轮数
        iter_t=model_params.get('iter_t', 20),  # 每个 ADMM 迭代中自编码器的训练轮数（预训练和主训练共用）
        batch_size=model_params.get('batch_size', 32),  # 批次大小
        random_seed=model_params.get('random_seed', None),  # 随机种子
        lr=model_params.get('lr', 1e-10),  # 学习率（预训练和主训练共用）
        epochs=model_params.get('epochs', 2),  # 训练轮数（预训练和主训练共用）
        device=device,
        hidden_dims=model_params.get('hidden_dims', [200, 100]),  # 自编码器隐藏层维度
        latent_dim=model_params.get('latent_dim', 100),  # 自编码器潜在表示维度
        beta_max=model_params.get('beta_max', 1e10),  # 惩罚参数最大值
        tol=model_params.get('tol', 1e-7),  # 收敛容差
        use_anchor=model_params.get('use_anchor', True),  # 是否使用锚点表达
        use_adaptive_weight=model_params.get('use_adaptive_weight', True),  # 是否使用自适应权重
        lightweight_mode=model_params.get('lightweight_mode', False),  # 轻量级模式
        use_pre_cache=model_params.get('use_pre_cache', True),  # 是否使用预训练缓存
        cv_init_method=model_params.get('cv_init_method', 0),  # Cv初始化方式
        early_stop=model_params.get('early_stop', True),  # 是否使用早停机制
        early_stop_min_delta=model_params.get('early_stop_min_delta', 1e-3),  # 早停最小改善值
        anchor_selection=model_params.get('anchor_selection', 'kmeans')  # 锚点选择方法
    )
    
    # 运行模型
    print(f"\n========== 开始训练 IHRMC-A 模型 ==========")
    if missing_rate > 0 and use_mask == True:
        pred_labels, history = model.fit(X, Y, masks)
    else:
        pred_labels, history = model.fit(X, Y)
    
    # 计算评估指标
    y_true_np = Y.cpu().numpy()
    metrics = clustering_metrics(y_true_np, pred_labels.cpu().numpy())
    
    # 打印结果
    print(f"\n========== 实验结果 ==========")
    print(f"数据集: {dataset_name}")
    print(f"ACC: {metrics['ACC']:.4f}")
    print(f"NMI: {metrics['NMI']:.4f}")
    print(f"ARI: {metrics['ARI']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F-score: {metrics['F-score']:.4f}")
    print(f"运行时间: {history.get('run_time', 0):.2f} 秒")
    
    # 保存结果
    save_results(metrics, model_params, history, output_dir, dataset_name, missing_rate, missing_mode)
    
    # 可视化（非轻量级模式）
    if not model_params.get('lightweight_mode', False):
        visualize_results(model, X, Y, pred_labels, history, output_dir, dataset_name, device)
    
    # 重命名输出目录为包含实验结果的格式
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    new_dir_name = generate_exp_dir_name(
        timestamp=timestamp,
        hyperparams={
            'lambda1': model_params.get('lambda1', 0.1),
            'lambda2': model_params.get('lambda2', 0.0),
            'lambda3': model_params.get('lambda3', 0.006),
            'beta': model_params.get('beta', 1e-4),
        },
        metrics=metrics
    )
    output_dir = rename_output_dir(output_dir, new_dir_name)
    
    return metrics, output_dir

def save_results(metrics, hyperparams, history, output_dir, dataset_name, missing_rate=0.0, missing_mode="sample"):
    """
    保存实验结果
    
    Args:
        metrics: 评估指标
        hyperparams: 超参数
        history: 训练历史
        output_dir: 输出目录
        dataset_name: 数据集名称
        missing_rate: 缺失数据比例
        missing_mode: 缺失模式
    """
    import json
    
    def convert_to_native_types(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native_types(item) for item in obj]
        elif isinstance(obj, (np.ndarray, np.generic)):
            return obj.item() if np.isscalar(obj) else obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        else:
            return obj
    
    result = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'run_time': history.get('run_time', 0),
        'max_memory_allocated': history.get('max_memory_allocated', 0),
        'missing_rate': missing_rate,
        'missing_mode': missing_mode,
        'hyperparams': convert_to_native_types(hyperparams),
        'metrics': convert_to_native_types(metrics)
    }
    
    save_path = os.path.join(output_dir, 'experiment_results.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"实验结果已保存到: {save_path}")
    
    # 检查是否为轻量级模式，轻量级模式下不生成训练日志
    lightweight_mode = hyperparams.get('lightweight_mode', False)
    if not lightweight_mode:
        # 生成训练日志 Excel 文件
        try:
            import pandas as pd
            
            # 准备训练日志数据
            log_data = []
            
            # 确定最大的迭代次数（基于计算容差的长度）
            max_iters = len(history.get('计算容差', []))
            
            if max_iters > 0:
                for i in range(max_iters):
                    iter_num = i
                    
                    # 对于只在每10次迭代记录的数据，找到最近的记录
                    def get_value(key, default=0):
                        data_list = history.get(key, [])
                        if len(data_list) == 0:
                            return default
                        # 如果是每10次迭代记录的数据（如ACC、NMI等）
                        if key in ['ACC', 'NMI', 'ARI', 'S矩阵块对角比', 'H类间/类内距离比', '运行时间']:
                            # 找到最近的记录索引
                            record_iter = i // 10
                            if record_iter < len(data_list):
                                return data_list[record_iter]
                            elif len(data_list) > 0:
                                return data_list[-1]  # 使用最后一个值
                            else:
                                return default
                        else:
                            # 对于每次迭代都记录的数据
                            if i < len(data_list):
                                return data_list[i]
                            else:
                                return default
                    
                    log_entry = {
                        '迭代次数': iter_num,
                        '运行时间': get_value('运行时间', 0),
                        '计算容差': get_value('计算容差', 0),
                        '原始目标值': get_value('原始目标值', 0),
                        '重建损失': get_value('重建损失', 0),
                        '表达损失': get_value('表达损失', 0),
                        '正则化损失': get_value('正则化损失', 0),
                        '一致性损失': get_value('一致性损失', 0),
                        '张量核范数': get_value('张量核范数', 0),
                        'ACC': get_value('ACC', 0),
                        'NMI': get_value('NMI', 0),
                        'ARI': get_value('ARI', 0),
                        'S矩阵块对角比': get_value('S矩阵块对角比', 0),
                        'H类间/类内距离比': get_value('H类间/类内距离比', 0)
                    }
                    log_data.append(log_entry)
            
            if log_data:
                # 定义列的顺序，按照旧版本的格式
                column_order = [
                    '迭代次数',
                    '运行时间',
                    '计算容差',
                    '原始目标值',
                    '重建损失',
                    '表达损失',
                    '正则化损失',
                    '超图正则化损失',
                    '一致性损失',
                    '张量核范数',
                    'ACC',
                    'NMI',
                    'ARI',
                    'S矩阵块对角比',
                    'H类间/类内距离比'
                ]
                
                # 创建 DataFrame 并按照指定顺序排列列
                df = pd.DataFrame(log_data)
                
                # 确保所有列都存在
                for col in column_order:
                    if col not in df.columns:
                        df[col] = 0.0
                
                df = df[column_order]
                
                # 保存为 Excel 文件
                excel_path = os.path.join(output_dir, 'training_log.xlsx')
                df.to_excel(excel_path, index=False)
                
                print(f"训练日志已保存到: {excel_path}")
            else:
                print("警告: 训练日志数据为空，跳过 Excel 日志生成")
        except ImportError:
            print("警告: 缺少 pandas 或 openpyxl，跳过 Excel 日志生成")
        except Exception as e:
            print(f"生成 Excel 日志失败: {e}")
    else:
        print("轻量级模式下，跳过训练日志生成")

def visualize_results(model, X, Y, pred_labels, history, output_dir, dataset_name, device):
    """
    可视化实验结果
    
    Args:
        model: IHRMC-A模型
        X: 数据
        Y: 真实标签
        pred_labels: 预测标签
        history: 训练历史
        output_dir: 输出目录
        dataset_name: 数据集名称
        device: 计算设备
    """
    visualizer = Visualizer()
    
    # 可视化训练历史
    save_path = os.path.join(output_dir, f'{dataset_name}_training_history.png')
    visualizer.plot_training_history(history, dataset_name, save_path=save_path)
    print(f"训练历史图已保存到: {save_path}")
    
    # 可视化S矩阵
    with torch.no_grad():
        S_np = model.S.cpu().numpy()
    save_path = os.path.join(output_dir, f'{dataset_name}_S_matrix.png')
    visualizer.plot_s_matrix(S_np, Y, dataset_name, save_path=save_path)
    print(f"S矩阵图已保存到: {save_path}")
    
    # 可视化混淆矩阵
    save_path = os.path.join(output_dir, f'{dataset_name}_confusion_matrix.png')
    visualizer.plot_confusion_matrix(Y, pred_labels, dataset_name, save_path=save_path)
    print(f"混淆矩阵图已保存到: {save_path}")
    
    # 可视化聚类结果对比
    with torch.no_grad():
        _, latent = model.aes[0](X[0].to(device))
    X_latent = latent.cpu().numpy()
    save_path = os.path.join(output_dir, f'{dataset_name}_cluster_comparison.png')
    visualizer.plot_cluster_comparison(X_latent, Y, pred_labels, dataset_name, save_path=save_path)
    print(f"聚类对比图已保存到: {save_path}")
    
    # 可视化指标热力图
    metrics = clustering_metrics(Y.cpu().numpy(), pred_labels.cpu().numpy())
    save_path = os.path.join(output_dir, f'{dataset_name}_metrics_heatmap.png')
    visualizer.plot_metrics_heatmap(metrics, dataset_name, save_path=save_path)
    print(f"指标热力图已保存到: {save_path}")

def main():
    """
    主函数，运行IHRMC-A实验
    """
    try:
        # 解析命令行参数并加载配置
        cli_args = parse_cli_args()
        config = load_config(cli_args=cli_args)
        
        # 运行实验
        metrics, output_dir = run_experiment(config)
        
        print(f"\n实验结果保存在: {output_dir}")
        
        return metrics
        
    except KeyboardInterrupt:
        print("\n用户中断实验")
        return None
    except Exception as e:
        print(f"\n实验运行出错: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    main()
