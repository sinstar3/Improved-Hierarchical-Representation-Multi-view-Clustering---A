"""
网络模块

包含以下功能：
1. 单隐藏层自编码器 (SingleLayerAE)
2. 完整自编码器 (Autoencoder)

主要用于 HRMC 模型中的特征提取和表示学习。
"""

from .network import (
    Autoencoder,
    SingleLayerAE,
    pretrain_single_layer_ae,
    update_single_ae
)

__all__ = [
    'Autoencoder',
    'SingleLayerAE',
    'pretrain_single_layer_ae',
    'update_single_ae'
]
