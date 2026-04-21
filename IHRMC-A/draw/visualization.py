"""
可视化模块

包含以下功能：
1. 矩阵可视化（C矩阵、S矩阵）
2. 聚类结果可视化（t-SNE、散点图、标签分布）
3. 实验结果对比（混淆矩阵、实验对比、块对角比对比）
4. 超参数分析（敏感性分析、热力图、搜索总结）
5. 训练过程可视化（训练历史）
6. 指标可视化（指标热力图）
"""

from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils.output_manager import get_save_path

matplotlib.use('Agg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class Visualizer:
    """
    可视化工具类，用于生成各种实验相关的图表
    """
    
    def __init__(self, 
                 colors: Optional[Dict[str, str]] = None,
                 dpi: int = 150,
                 font_family: str = None,
                 default_fig_size: tuple = (10, 6),
                 max_samples: int = 1000):
        """
        初始化可视化工具
        
        Args:
            colors: 颜色配置字典
            dpi: 图表分辨率
            font_family: 字体家族
            default_fig_size: 默认图表大小
            max_samples: t-SNE 最大样本数
        """
        # 颜色配置
        self.colors = colors or {
            'total_loss': '#2E86AB',
            'recon_loss': '#A23B72',
            'self_expr_loss': '#F18F01',
            'reg_loss': '#C73E1D',
            'true_labels': '#2E86AB',
            'pred_labels': '#A23B72',
            'c_matrix': '#2E86AB',
            's_matrix': '#A23B72',
            'heatmap': 'YlGnBu',
            'confusion_matrix': 'Blues',
            'scatter': 'tab10'
        }
        
        # 通用配置
        self.dpi = dpi
        self.default_fig_size = default_fig_size
        self.max_samples = max_samples
        
        # 字体配置
        if font_family is None:
            # 查找系统中可用的中文字体
            available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
            # 优先使用系统中可用的中文字体
            chinese_fonts = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Arial Unicode MS', 'sans-serif']
            selected_font = None
            
            for font in chinese_fonts:
                if font in available_fonts:
                    selected_font = font
                    break
            
            if selected_font:
                self.font_family = selected_font
            else:
                self.font_family = 'sans-serif'
        else:
            self.font_family = font_family
        
        # 设置字体
        plt.rcParams['font.sans-serif'] = [self.font_family]
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
    
    def _add_class_boundaries(self, y_true) -> None:
        """
        添加类别边界线
        
        Args:
            y_true: numpy array or torch tensor, shape (N,)，真实标签
        """
        # 转换为numpy数组
        if hasattr(y_true, 'cpu'):
            y_true = y_true.cpu().numpy()
        
        boundaries = np.cumsum(np.bincount(y_true))[:-1]
        for b in boundaries:
            plt.axhline(b - 0.5, color='cyan', linewidth=1.2, linestyle='-', alpha=0.7)
            plt.axvline(b - 0.5, color='cyan', linewidth=1.2, linestyle='-', alpha=0.7)
    
    def _save_and_cleanup(self, fig: plt.Figure, save_path: str, dpi: int = None) -> None:
        """
        保存图表并清理资源
        
        Args:
            fig: 图表对象
            save_path: 保存路径
            dpi: 分辨率，默认使用初始化时的设置
        """
        plt.savefig(save_path, dpi=dpi or self.dpi, bbox_inches='tight')
        plt.close(fig)
        del fig
    
    def _plot_scatter_with_legend(self, ax: plt.Axes, x: np.ndarray, y: np.ndarray, labels: np.ndarray, 
                                 title: str, cmap: str = None, s: int = 40) -> None:
        """
        绘制散点图并添加图例
        
        Args:
            ax: 坐标轴对象
            x: x坐标
            y: y坐标
            labels: 标签
            title: 标题
            cmap: 颜色映射，默认使用初始化时的设置
            s: 点大小
        """
        scatter = ax.scatter(x, y, c=labels, cmap=cmap or self.colors['scatter'], 
                           s=s, edgecolor='k', linewidth=0.5, alpha=0.8)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        handles, _ = scatter.legend_elements()
        ax.legend(handles, [f'Class {i}' for i in sorted(set(labels))], 
                  title="Classes", loc='best', fontsize=9, title_fontsize=10)
    
    def _plot_bar_with_labels(self, ax: plt.Axes, x: np.ndarray, y: np.ndarray, 
                            title: str, color: str = None) -> None:
        """
        绘制柱状图并添加标签
        
        Args:
            ax: 坐标轴对象
            x: x坐标
            y: y值
            title: 标题
            color: 颜色，默认使用初始化时的设置
        """
        bars = ax.bar(x, y, color=color or self.colors['true_labels'], 
                     alpha=0.8, edgecolor='black', linewidth=0.8)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Class', fontsize=10, labelpad=8)
        ax.set_ylabel('Count', fontsize=10, labelpad=8)
        ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', labelsize=9)
        
        max_val = max(y) if len(y) > 0 else 1
        for i, (cls, cnt) in enumerate(zip(x, y)):
            ax.text(cls, cnt + max_val * 0.02, str(cnt), 
                   ha='center', fontsize=9, fontweight='semibold')
    
    def plot_c_matrix(
        self, C,
        y_true,
        dataset_name: str,
        save_path: Optional[str] = None
        ) -> None:
        """
        绘制系数矩阵 C 的块对角结构（按真实标签排序）
        
        Args:
            C: numpy array or torch tensor，系数矩阵
                - 形状 (N, N)：传统自表达
                - 形状 (m, N)：锚点表达
            y_true: torch tensor, shape (N,)，真实标签（用于排序和添加边界）
            dataset_name: str，数据集名称（用于标题和默认文件名）
            save_path: str，可选，指定保存路径，若为 None 则自动生成文件名
        """
        # 转换为numpy数组
        if hasattr(C, 'cpu'):
            C = C.cpu().numpy()
        y_true = y_true.cpu().numpy()
        
        # 检查是否为锚点表达（形状为 (m, N)）
        if C.shape[0] != C.shape[1]:
            # 对于锚点表达，绘制亲和矩阵 C.T @ C
            C = C.T @ C
            title = f'{dataset_name} - Affinity matrix (C.T @ C) sorted by true labels'
        else:
            # 对于传统自表达，直接绘制 C
            title = f'{dataset_name} - C matrix (view 0) sorted by true labels'
        
        order = np.argsort(y_true)
        C_sorted = C[order][:, order]
        
        N = C.shape[0]
        fig_size = min(10, max(6, N / 30 + 3))
        
        fig = plt.figure(figsize=(fig_size, fig_size))
        
        im = plt.imshow(C_sorted, cmap='hot', interpolation='nearest', 
                       aspect='equal')
        
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
        
        plt.title(title, 
                 fontsize=12, fontweight='bold', pad=12)
        
        self._add_class_boundaries(y_true)
        
        plt.tick_params(axis='both', labelsize=9)
        
        if save_path is None:
            save_path = f'{dataset_name}_C_matrix.png'
        
        self._save_and_cleanup(fig, save_path)
        del im, cbar
    
    def plot_s_matrix(
        self, S,
        y_true,
        dataset_name: str,
        save_path: Optional[str] = None
        ) -> None:
        """
        绘制统一相似矩阵 S 的块对角结构（按真实标签排序）
        
        Args:
            S: numpy array or torch tensor, shape (N, N)，统一相似矩阵
            y_true: torch tensor, shape (N,)，真实标签（用于排序和添加边界）
            dataset_name: str，数据集名称（用于标题和默认文件名）
            save_path: str，可选，指定保存路径，若为 None 则自动生成文件名
        """
        # 转换为numpy数组
        if hasattr(S, 'cpu'):
            S = S.cpu().numpy()
        y_true = y_true.cpu().numpy()
        
        # 确保S是(N, N)形状
        if S.shape[0] != S.shape[1]:
            # 如果S是(m, N)形状，构建相似矩阵
            S = S.T @ S
        
        order = np.argsort(y_true)
        S_sorted = S[order][:, order]
        
        N = S.shape[0]
        fig_size = min(10, max(6, N / 30 + 3))
        
        fig = plt.figure(figsize=(fig_size, fig_size))
        
        im = plt.imshow(S_sorted, cmap='hot', interpolation='nearest',
                       aspect='equal')
        
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
        
        plt.title(f'{dataset_name} - Unified similarity matrix S sorted by true labels',
                 fontsize=12, fontweight='bold', pad=12)
        
        self._add_class_boundaries(y_true)
        
        plt.tick_params(axis='both', labelsize=9)
        
        if save_path is None:
            save_path = f'{dataset_name}_S_matrix.png'
        
        self._save_and_cleanup(fig, save_path)
        del im, cbar
    
    def plot_cluster_comparison(
        self,
        latent,
        y_true,
        y_pred,
        dataset_name: str,
        save_path: Optional[str] = None
        ) -> None:
        """
        绘制聚类结果对比图，包括：
        1. 真实标签的散点图
        2. 预测标签的散点图
        
        注意：聚类标签与真实标签之间没有直接的对应关系，不进行"正确/错误"标注。
        聚类质量通过 ACC、NMI、ARI 等指标衡量。
        
        Args:
            latent: torch tensor, shape (N, d)，潜在表示
            y_true: torch tensor, shape (N,)，真实标签
            y_pred: torch tensor, shape (N,)，预测标签
            dataset_name: str，数据集名称
            save_path: str，可选，保存路径
        """
        from sklearn.manifold import TSNE
        
        # 转换为numpy数组
        if hasattr(latent, 'cpu'):
            latent = latent.cpu().numpy()
        if hasattr(y_true, 'cpu'):
            y_true = y_true.cpu().numpy()
        if hasattr(y_pred, 'cpu'):
            y_pred = y_pred.cpu().numpy()
        
        N = latent.shape[0]
        if N > self.max_samples:
            indices = np.random.choice(N, size=self.max_samples, replace=False)
            print(f"大型数据集采样: {N} -> {self.max_samples} 样本")
            latent, y_true, y_pred = latent[indices], y_true[indices], y_pred[indices]
       # 使用t-SNE降维，明确设置初始化和学习率参数以消除警告
        tsne = TSNE(n_components=2, random_state=42, perplexity=20, init='random', learning_rate=200.0)
        X_tsne = tsne.fit_transform(latent)
        
        fig = plt.figure(figsize=(12, 5))
        gs = fig.add_gridspec(1, 2, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        
        self._plot_scatter_with_legend(ax1, X_tsne[:, 0], X_tsne[:, 1], y_true, 
                                     f'{dataset_name}\nTrue Labels')
        
        self._plot_scatter_with_legend(ax2, X_tsne[:, 0], X_tsne[:, 1], y_pred, 
                                     f'{dataset_name}\nPredicted Labels')
        
        fig.suptitle(f'Clustering Visualization', fontsize=14, fontweight='bold', y=1.02)
        
        if save_path is None:
            save_path = f'{dataset_name}_cluster_comparison.png'
        
        self._save_and_cleanup(fig, save_path)
        del ax1, ax2, tsne, X_tsne
    
    def plot_label_distribution(
        self, y_true,
        y_pred,
        dataset_name: str,
        save_path: Optional[str] = None
        ) -> None:
        """
        绘制真实标签和预测标签的分布
        
        Args:
            y_true: numpy array or torch tensor, shape (N,)，真实标签
            y_pred: numpy array or torch tensor, shape (N,)，预测标签
            dataset_name: str，数据集名称（用于标题和默认文件名）
            save_path: str，可选，指定保存路径，若为 None 则自动生成文件名
        """
        # 转换为numpy数组
        if hasattr(y_true, 'cpu'):
            y_true = y_true.cpu().numpy()
        if hasattr(y_pred, 'cpu'):
            y_pred = y_pred.cpu().numpy()
        
        # 设置更合理的图表大小，适合显示标签分布
        fig_width = 10
        fig_height = 8
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = fig.add_gridspec(1, 2, wspace=0.3)
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
        
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        
        # 计算两个分布的最大计数值
        max_count = max(max(counts_true), max(counts_pred))
        
        # 计算更合理的 y 轴上限，向上取整到最近的10的倍数，确保图表有足够的空间
        y_max = ((max_count // 10) + 1) * 10
        
        # 绘制真实标签分布
        self._plot_bar_with_labels(axes[0], unique_true, counts_true, 
                                 f'{dataset_name}\nTrue Label Distribution',
                                 color=self.colors['true_labels'])
        axes[0].set_ylim(0, y_max)  # 设置 y 轴范围
        
        # 绘制预测标签分布
        self._plot_bar_with_labels(axes[1], unique_pred, counts_pred, 
                                 f'{dataset_name}\nPredicted Label Distribution',
                                 color=self.colors['pred_labels'])
        axes[1].set_ylim(0, y_max)  # 设置相同的 y 轴范围
        
        if save_path is None:
            save_path = f'{dataset_name}_label_distribution.png'
        
        self._save_and_cleanup(fig, save_path)
        del axes
    
    def plot_confusion_matrix(
        self, y_true,
        y_pred,
        dataset_name: str,
        save_path: Optional[str] = None
        ) -> None:
        """
        绘制混淆矩阵
        
        Args:
            y_true: numpy array or torch tensor, shape (N,)，真实标签
            y_pred: numpy array or torch tensor, shape (N,)，预测标签
            dataset_name: str，数据集名称（用于标题和默认文件名）
            save_path: str，可选，指定保存路径，若为 None 则自动生成文件名
        """
        # 转换为numpy数组
        if hasattr(y_true, 'cpu'):
            y_true = y_true.cpu().numpy()
        if hasattr(y_pred, 'cpu'):
            y_pred = y_pred.cpu().numpy()
        
        cm = confusion_matrix(y_true, y_pred)
        classes = sorted(set(y_true))
        num_classes = len(classes)
        
        # 增加图片大小，确保标签有足够空间显示
        fig_width = min(max(12, 5 + num_classes * 0.8), 20)
        fig_height = min(max(10, 4 + num_classes * 0.6), 16)
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # 减少注释大小，增加行宽，确保矩阵清晰
        heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap=self.colors['confusion_matrix'], 
                              xticklabels=classes, 
                              yticklabels=classes,
                              cbar_kws={'label': 'Count', 'shrink': 0.8},
                              annot_kws={'size': 8, 'weight': 'bold'},
                              linewidths=0.5, linecolor='gray',
                              square=True)
        
        # 使用中文标签
        plt.xlabel('预测标签', fontsize=12, fontweight='medium', labelpad=8)
        plt.ylabel('真实标签', fontsize=12, fontweight='medium', labelpad=8)
        plt.title(f'{dataset_name} - 混淆矩阵', fontsize=14, fontweight='bold', pad=15)
        
        # 调整标签旋转角度和字体大小，确保标签清晰显示
        if num_classes > 10:
            heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90, ha='center', fontsize=7)
            heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=7)
        elif num_classes > 5:
            heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=8)
        else:
            heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=9)
            heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=9)
        
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label('Count', fontsize=10, fontweight='medium')
        
        if save_path is None:
            save_path = f'{dataset_name}_confusion_matrix.png'
        
        self._save_and_cleanup(fig, save_path)
        del heatmap, cbar
    
    def plot_experiment_comparison(
        self, experiment_results: List[Dict[str, Any]],
        save_path: Optional[str] = None
        ) -> None:
        """
        绘制多个实验结果的对比图
        
        Args:
            experiment_results: list of dicts，每个字典包含单次实验的结果
                               格式: [{'name': 'exp1', 'ACC': 0.8, 'NMI': 0.7, ...}, ...]
            save_path: str，可选，保存路径
        """
        metrics = ['ACC', 'NMI', 'ARI', 'F-score']
        exp_names = [exp['name'] for exp in experiment_results]
        
        num_exps = len(experiment_results)
        width = max(20, 8 + num_exps * 1.5)
        height = max(14, 10 + num_exps * 0.3)
        
        fig, axes = plt.subplots(2, 2, figsize=(width, height))
        axes = axes.flatten()
        
        colors = plt.cm.Set3(np.linspace(0, 1, num_exps))
        
        for idx, metric in enumerate(metrics):
            values = [exp[metric] for exp in experiment_results]
            ax = axes[idx]
            bars = ax.bar(exp_names, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)
            ax.set_title(f'{metric} Comparison', fontsize=16, fontweight='bold', pad=15)
            ax.set_ylabel(metric, fontsize=14, labelpad=10)
            ax.set_xlabel('Experiment', fontsize=14, labelpad=10)
            
            if len(exp_names) > 10:
                rotation, label_size = 90, 9
            elif len(exp_names) > 5:
                rotation, label_size = 75, 10
            else:
                rotation, label_size = 45, 11
            ax.tick_params(axis='x', rotation=rotation, labelsize=label_size)
            ax.tick_params(axis='y', labelsize=12)
            ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
            ax.set_axisbelow(True)
            
            max_val = max(values) if values else 1
            for i, v in enumerate(values):
                ax.text(i, v + max_val * 0.015, f'{v:.4f}', 
                       ha='center', fontsize=9, fontweight='semibold')
        
        if save_path is None:
            save_path = 'experiment_comparison.png'
        
        self._save_and_cleanup(fig, save_path, dpi=300)
        del axes
    
    def plot_block_diag_comparison(
        self, experiment_results: List[Dict[str, Any]],
        save_path: Optional[str] = None
        ) -> None:
        """
        绘制多个实验的块对角比对比图
        
        Args:
            experiment_results: list of dicts，每个字典包含单次实验的结果
                               格式: [{'name': 'exp1', 'C_view0': 0.3, 'S': 0.35, ...}, ...]
            save_path: str，可选，保存路径
        """
        exp_names = [exp['name'] for exp in experiment_results]
        x = np.arange(len(exp_names))
        width = 0.35
        
        num_exps = len(experiment_results)
        fig_width = max(20, 8 + num_exps * 1.8)
        fig_height = max(8, 6 + num_exps * 0.2)
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        c_values = [exp['C_view0'] for exp in experiment_results]
        s_values = [exp['S'] for exp in experiment_results]
        
        color_c = self.colors['c_matrix']
        color_s = self.colors['s_matrix']
        
        rects1 = ax.bar(x - width/2, c_values, width, label='C_view0', 
                      alpha=0.85, color=color_c, edgecolor='black', linewidth=0.8)
        rects2 = ax.bar(x + width/2, s_values, width, label='S', 
                      alpha=0.85, color=color_s, edgecolor='black', linewidth=0.8)
        
        ax.set_xlabel('Experiment', fontsize=15, fontweight='medium', labelpad=12)
        ax.set_ylabel('Block Diagonal Ratio', fontsize=15, fontweight='medium', labelpad=12)
        ax.set_title('Block Diagonal Ratio Comparison', fontsize=18, fontweight='bold', pad=20)
        ax.set_xticks(x)
        
        if num_exps > 10:
            rotation, label_size = 90, 9
        elif num_exps > 5:
            rotation, label_size = 75, 10
        else:
            rotation, label_size = 45, 11
        ax.set_xticklabels(exp_names, rotation=rotation, fontsize=label_size, ha='right')
        ax.tick_params(axis='y', labelsize=13)
        ax.legend(fontsize=13, loc='best', framealpha=0.9)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
        ax.set_axisbelow(True)
        
        all_values = c_values + s_values
        max_val = max(all_values) if all_values else 1
        
        for rect in rects1:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height + max_val * 0.012,
                    f'{height:.4f}', ha='center', va='bottom', 
                    fontsize=10, fontweight='semibold')
        for rect in rects2:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height + max_val * 0.012,
                    f'{height:.4f}', ha='center', va='bottom', 
                    fontsize=10, fontweight='semibold')
        
        if save_path is None:
            save_path = 'block_diag_comparison.png'
        
        self._save_and_cleanup(fig, save_path, dpi=300)
        del ax
    
    def plot_hyperparameter_sensitivity(
        self, all_results: List[Dict[str, Any]],
        param_name: str,
        metric: str = 'ACC',
        save_path: Optional[str] = None,
        output_dir: Optional[str] = None
        ) -> None:
        """
        绘制单个超参数的敏感性分析图
        
        Args:
            all_results: hyperparameter_search() 返回的结果列表
            param_name: 要分析的超参数名称
            metric: 要绘制的指标 ('ACC', 'NMI', 'ARI', 'F-score')
            save_path: 可选，保存路径
            output_dir: 可选，输出目录
        """
        param_values = []
        metric_values = []
        
        for result in all_results:
            if param_name in result['params']:
                param_values.append(result['params'][param_name])
                metric_values.append(result['metrics'][metric])
        
        if len(param_values) == 0:
            print(f"未找到超参数 {param_name} 的数据")
            return
        
        plt.figure(figsize=self.default_fig_size)
        
        if all(isinstance(v, (int, float)) for v in param_values):
            sorted_indices = np.argsort(param_values)
            param_values_sorted = [param_values[i] for i in sorted_indices]
            metric_values_sorted = [metric_values[i] for i in sorted_indices]
            
            plt.plot(param_values_sorted, metric_values_sorted, 'o-', linewidth=2, markersize=8)
            plt.xscale('log' if min(param_values_sorted) > 0 and max(param_values_sorted) / min(param_values_sorted) > 100 else 'linear')
        else:
            unique_params = sorted(list(set(param_values)))
            avg_metrics = []
            std_metrics = []
            for p in unique_params:
                vals = [m for pm, m in zip(param_values, metric_values) if pm == p]
                avg_metrics.append(np.mean(vals))
                std_metrics.append(np.std(vals))
            
            x_pos = np.arange(len(unique_params))
            plt.bar(x_pos, avg_metrics, yerr=std_metrics, capsize=5, alpha=0.7)
            plt.xticks(x_pos, [str(p) for p in unique_params], rotation=45)
        
        plt.xlabel(param_name, fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.title(f'{metric} vs {param_name}', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        if save_path is None:
            save_path = get_save_path(f'{param_name}_sensitivity.png', output_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_hyperparameter_heatmap(
        self, all_results: List[Dict[str, Any]],
        param_x: str, param_y: str,
        metric: str = 'ACC',
        save_path: Optional[str] = None, 
        output_dir: Optional[str] = None
        ) -> None:
        """
        绘制两个超参数的热力图（需要参数组合有重复值）
        
        Args:
            all_results: hyperparameter_search() 返回的结果列表
            param_x: x轴超参数
            param_y: y轴超参数
            metric: 要绘制的指标
            save_path: 可选，保存路径
            output_dir: 可选，输出目录
        """
        x_vals = []
        y_vals = []
        metric_vals = []
        
        for result in all_results:
            if param_x in result['params'] and param_y in result['params']:
                x_vals.append(result['params'][param_x])
                y_vals.append(result['params'][param_y])
                metric_vals.append(result['metrics'][metric])
        
        if len(x_vals) == 0:
            print(f"未找到超参数组合 {param_x} 和 {param_y} 的数据")
            return
        
        unique_x = sorted(list(set(x_vals)))
        unique_y = sorted(list(set(y_vals)))
        
        if len(unique_x) < 2 or len(unique_y) < 2:
            print("需要至少2个不同的值来绘制热力图")
            return
        
        heatmap_data = np.zeros((len(unique_y), len(unique_x)))
        
        for x, y, m in zip(x_vals, y_vals, metric_vals):
            x_idx = unique_x.index(x)
            y_idx = unique_y.index(y)
            heatmap_data[y_idx, x_idx] = m
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap=self.colors['heatmap'],
                    xticklabels=[str(x) for x in unique_x],
                    yticklabels=[str(y) for y in unique_y])
        plt.xlabel(param_x, fontsize=12)
        plt.ylabel(param_y, fontsize=12)
        plt.title(f'{metric} Heatmap: {param_x} vs {param_y}', fontsize=14, fontweight='bold')
        
        if save_path is None:
            save_path = get_save_path(f'{param_x}_vs_{param_y}_heatmap.png', output_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_hyperparameter_search_summary(
        self, all_results: List[Dict[str, Any]], 
        save_path: Optional[str] = None, 
        output_dir: Optional[str] = None
        ) -> None:
        """
        绘制超参数搜索结果的综合对比图（所有实验的指标对比）
        
        Args:
            all_results: hyperparameter_search() 返回的结果列表
            save_path: 可选，保存路径
            output_dir: 可选，输出目录
        """
        metrics = ['ACC', 'NMI', 'ARI', 'F-score']
        exp_names = [f'Iter {r["search_idx"]+1}' for r in all_results]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_results)))
        
        for idx, metric in enumerate(metrics):
            values = [r['metrics'][metric] for r in all_results]
            bars = axes[idx].bar(exp_names, values, color=colors, alpha=0.7)
            axes[idx].set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel(metric, fontsize=10)
            axes[idx].tick_params(axis='x', rotation=45, labelsize=8)
            axes[idx].grid(axis='y', alpha=0.3)
            
            for i, v in enumerate(values):
                axes[idx].text(i, v + max(values)*0.01, f'{v:.4f}', 
                              ha='center', fontsize=7)
        
        if save_path is None:
            save_path = get_save_path('hyperparameter_search_summary.png', output_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_history(
        self, history: Dict[str, List[float]],
        dataset_name: str,
        save_path: Optional[str] = None
        ) -> None:
        """
        绘制训练过程的损失曲线和误差曲线

        Args:
            history: 训练历史字典，包含 loss 和 error 信息
            dataset_name: 数据集名称
            save_path: 可选，保存路径
        """
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(2, 1, hspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        lines = []
        labels = []

        if '原始目标值' in history:
            line, = ax1.plot(history['原始目标值'], label='原始目标值',
                            linewidth=1.5, color=self.colors['total_loss'], alpha=0.9)
            lines.append(line)
            labels.append('原始目标值')
        if '重建损失' in history:
            line, = ax1.plot(history['重建损失'], label='重建损失',
                            linewidth=1.5, color=self.colors['recon_loss'], alpha=0.9)
            lines.append(line)
            labels.append('重建损失')
        if '表达损失' in history:
            line, = ax1.plot(history['表达损失'], label='表达损失',
                            linewidth=1.5, color=self.colors['self_expr_loss'], alpha=0.9)
            lines.append(line)
            labels.append('表达损失')
        if '正则化损失' in history:
            line, = ax1.plot(history['正则化损失'], label='正则化损失',
                            linewidth=1.5, color=self.colors['reg_loss'], alpha=0.9)
            lines.append(line)
            labels.append('正则化损失')
        if '超图正则化损失' in history:
            line, = ax1.plot(history['超图正则化损失'], label='超图正则化损失',
                            linewidth=1.5, color='#2ECC71', alpha=0.9)
            lines.append(line)
            labels.append('超图正则化损失')
        if '一致性损失' in history:
            line, = ax1.plot(history['一致性损失'], label='一致性损失',
                            linewidth=1.5, color='#9B59B6', alpha=0.9)
            lines.append(line)
            labels.append('一致性损失')
        if '张量核范数' in history:
            line, = ax1.plot(history['张量核范数'], label='张量核范数',
                            linewidth=1.5, color='#E74C3C', alpha=0.9)
            lines.append(line)
            labels.append('张量核范数')

        ax1.set_title(f'{dataset_name} - 训练损失曲线',
                     fontsize=14, fontweight='bold', pad=12)
        ax1.set_xlabel('迭代次数', fontsize=12, labelpad=8)
        ax1.set_ylabel('损失', fontsize=12, labelpad=8)
        ax1.set_yscale('log')
        ax1.grid(alpha=0.2, linestyle='--', linewidth=0.8)
        ax1.set_axisbelow(True)
        ax1.legend(lines, labels, fontsize=10, loc='best', framealpha=0.9)
        ax1.tick_params(axis='both', labelsize=10)

        ax2 = fig.add_subplot(gs[1, 0])
        if '计算容差' in history:
            ax2.plot(history['计算容差'], label='收敛容差',
                    linewidth=1.5, color=self.colors['reg_loss'], alpha=0.9)

        ax2.set_title(f'{dataset_name} - 收敛容差曲线',
                     fontsize=14, fontweight='bold', pad=12)
        ax2.set_xlabel('迭代次数', fontsize=12, labelpad=8)
        ax2.set_ylabel('容差', fontsize=12, labelpad=8)
        ax2.set_yscale('log')
        ax2.grid(alpha=0.2, linestyle='--', linewidth=0.8)
        ax2.set_axisbelow(True)
        ax2.legend(fontsize=10, loc='best', framealpha=0.9)
        ax2.tick_params(axis='both', labelsize=10)

        if save_path is None:
            save_path = f'{dataset_name}_training_history.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        del fig, ax1, ax2
    
    def plot_metrics_heatmap(
        self, metrics_dict: Dict[str, float],
        dataset_name: str,
        save_path: Optional[str] = None
        ) -> None:
        """
        绘制评估指标热力图
        
        Args:
            metrics_dict: 包含评估指标的字典
            dataset_name: 数据集名称
            save_path: 可选，保存路径
        """
        valid_metrics = {k: v for k, v in metrics_dict.items() if v is not None}
        
        if not valid_metrics:
            return
        
        metrics = list(valid_metrics.keys())
        values = list(valid_metrics.values())
        num_metrics = len(metrics)
        
        data = np.array(values).reshape(1, -1)
        
        fig_width = min(max(10, 6 + num_metrics * 1.0), 14)
        fig_height = min(max(4, 3 + num_metrics * 0.15), 6)
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        heatmap = sns.heatmap(data, annot=True, fmt='.4f', cmap='Greens',
                             xticklabels=metrics, yticklabels=[dataset_name],
                             cbar_kws={'label': 'Value', 'shrink': 0.8},
                             annot_kws={'size': 10, 'weight': 'bold'},
                             linewidths=0.8, linecolor='gray',
                             square=True)
        
        plt.title(f'{dataset_name} - Clustering Metrics', 
                 fontsize=14, fontweight='bold', pad=15)
        
        heatmap.set_xticklabels(heatmap.get_xticklabels(), 
                               rotation=30, ha='right', fontsize=10, fontweight='medium')
        heatmap.set_yticklabels(heatmap.get_yticklabels(), 
                               rotation=0, fontsize=11, fontweight='medium')
        
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Value', fontsize=11, fontweight='medium')
        
        if save_path is None:
            save_path = f'{dataset_name}_metrics_heatmap.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        del fig, heatmap
