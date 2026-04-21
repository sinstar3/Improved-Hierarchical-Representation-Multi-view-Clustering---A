"""
配置管理模块

包含以下功能：
1. Config 类 - 配置管理类，支持配置文件加载和命令行参数覆盖
2. 命令行参数解析 - 解析命令行参数
3. 配置文件加载 - 加载和管理配置文件
4. 数据集配置 - 获取特定数据集的配置
"""

import argparse
import os
from typing import Any, Dict, Optional

import yaml

class Config:
    def __init__(self, config_file=None, cli_args=None):
        """
        初始化配置管理类
        
        Args:
            config_file: 配置文件路径，默认为None
            cli_args: 命令行参数，默认为None
        """
        self.config = {}
        
        # 加载配置文件
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        
        # 应用命令行参数覆盖
        if cli_args:
            self._apply_cli_args(cli_args)
    
    def get(self, key, default=None):
        """
        获取配置值
        
        Args:
            key: 配置键，支持点号分隔的嵌套键
            default: 默认值
        
        Returns:
            配置值或默认值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if k not in value:
                return default
            value = value[k]
        
        return value
    
    def update(self, new_config):
        """
        更新配置
        
        Args:
            new_config: 新配置字典
        """
        self.config.update(new_config)
    
    def save(self, config_file):
        """
        保存配置到文件
        
        Args:
            config_file: 保存路径
        """
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
    
    def get_model_params(self):
        """
        获取模型参数
        
        Returns:
            模型参数字典
        """
        model_params = self.get('model', {})
        # 过滤掉None值
        return {k: v for k, v in model_params.items() if v is not None}
    
    def get_section_params(self, section):
        """
        获取指定 section 的参数
        
        Args:
            section: 配置节名称
        
        Returns:
            参数字典
        """
        return self.get(section, {})
    
    def get_training_params(self):
        """
        获取训练参数
        
        Returns:
            训练参数字典
        """
        return self.get_section_params('training')
    
    def get_hyperparameter_search_params(self):
        """
        获取超参数搜索参数
        
        Returns:
            超参数搜索参数字典
        """
        return self.get_section_params('hyperparameter_search')
    
    def get_output_params(self):
        """
        获取输出参数
        
        Returns:
            输出参数字典
        """
        return self.get_section_params('output')
    
    def _apply_cli_args(self, cli_args):
        """
        应用命令行参数覆盖配置
        
        Args:
            cli_args: 命令行参数命名空间
        """
        # 处理模型参数
        model_params = ['lambda1', 'lambda2', 'lambda3', 'lr', 'inner_epochs',
                       'beta', 'beta_max', 'tol', 'latent_dim',
                       'use_pretrain_cache', 'random_seed', 'm', 'use_anchor']
        
        for param in model_params:
            if hasattr(cli_args, param) and getattr(cli_args, param) is not None:
                if 'model' not in self.config:
                    self.config['model'] = {}
                self.config['model'][param] = getattr(cli_args, param)
        
        # 特殊处理 use_anchor 参数
        if hasattr(cli_args, 'use_anchor') and cli_args.use_anchor is not None:
            if 'model' not in self.config:
                self.config['model'] = {}
            use_anchor_str = cli_args.use_anchor.lower()
            self.config['model']['use_anchor'] = use_anchor_str in ['true', '1', 'yes']
        
        # 特殊处理 lightweight_mode 参数
        if hasattr(cli_args, 'lightweight_mode') and cli_args.lightweight_mode is not None:
            if 'model' not in self.config:
                self.config['model'] = {}
            lightweight_mode_str = cli_args.lightweight_mode.lower()
            self.config['model']['lightweight_mode'] = lightweight_mode_str in ['true', '1', 'yes']
        
        # 处理特殊参数：hidden_dims
        if hasattr(cli_args, 'hidden_dims') and cli_args.hidden_dims is not None:
            if 'model' not in self.config:
                self.config['model'] = {}
            try:
                hidden_dims = [int(d) for d in cli_args.hidden_dims.split(',')]
                self.config['model']['hidden_dims'] = hidden_dims
            except ValueError:
                print(f"警告: 无法解析 hidden_dims 参数: {cli_args.hidden_dims}")
        
        # 处理数据集参数
        if hasattr(cli_args, 'dataset') and cli_args.dataset is not None:
            if 'dataset' not in self.config:
                self.config['dataset'] = {}
            self.config['dataset']['name'] = cli_args.dataset
        
        # 处理设备参数
        if hasattr(cli_args, 'device') and cli_args.device is not None:
            if 'training' not in self.config:
                self.config['training'] = {}
            self.config['training']['device'] = cli_args.device


def parse_cli_args():
    """
    解析命令行参数
    
    Returns:
        命令行参数命名空间
    """
    parser = argparse.ArgumentParser(description='IHRMC-A Configuration')
    
    # 模型参数
    model_group = parser.add_argument_group('模型参数')
    model_group.add_argument('--lambda1', type=float, help='自表达损失权重')
    model_group.add_argument('--lambda2', type=float, help='一致性约束权重')
    model_group.add_argument('--lambda3', type=float, help='张量核范数权重')
    model_group.add_argument('--lr', type=float, help='自编码器学习率')
    model_group.add_argument('--inner_epochs', type=int, help='每个 ADMM 迭代中自编码器的训练轮数')
    model_group.add_argument('--beta', type=float, help='ADMM 惩罚参数初始值')
    model_group.add_argument('--beta_max', type=float, help='ADMM 惩罚参数最大值')
    model_group.add_argument('--tol', type=float, help='收敛容差')
    model_group.add_argument('--hidden_dims', type=str, help='隐藏层维度列表，如 "200,100"')
    model_group.add_argument('--latent_dim', type=int, help='潜在表示维度')
    model_group.add_argument('--pretrain_lr', type=float, help='预训练学习率')
    model_group.add_argument('--pretrain_epochs', type=int, help='预训练最大轮数')
    model_group.add_argument('--pretrain_batch_size', type=int, help='预训练batch大小，0表示full batch')
    model_group.add_argument('--use_pretrain_cache', type=bool, help='是否使用预训练缓存')
    model_group.add_argument('--random_seed', type=str, help='随机种子，设置为 None 表示不设置种子')
    model_group.add_argument('--cv_init_method', type=int, help='Cv初始化方式，0: 基本SSC(原始X), 1: SSC(原始X), 2: SSC_ADMM(原始X), 3: 零矩阵, 4: 随机矩阵')
    model_group.add_argument('--m', type=int, help='锚点数')
    model_group.add_argument('--use_anchor', type=str, help='是否使用锚点，True/False，False表示使用原始HRMC架构')
    model_group.add_argument('--lightweight_mode', type=str, help='轻量级模式，True/False，True时不计算损失和可视化，加速训练')
    
    # 数据集参数
    dataset_group = parser.add_argument_group('数据集参数')
    dataset_group.add_argument('--dataset', type=str, help='数据集名称')
    
    # 训练参数
    training_group = parser.add_argument_group('训练参数')
    training_group.add_argument('--device', type=str, help='计算设备 (cuda 或 cpu)')
    
    return parser.parse_args()


def load_config(config_file=None, cli_args=None):
    """
    加载配置文件
    
    Args:
        config_file: 配置文件路径，默认为None
        cli_args: 命令行参数，默认为None
    
    Returns:
        Config对象
    """
    if config_file is None:
        # 默认使用configs目录下的default_config.yaml
        config_file = os.path.join('configs', 'default_config.yaml')
    
    return Config(config_file, cli_args)
