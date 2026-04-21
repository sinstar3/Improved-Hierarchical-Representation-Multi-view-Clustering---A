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

from .dataloader import DataLoader

__all__ = ['DataLoader']
