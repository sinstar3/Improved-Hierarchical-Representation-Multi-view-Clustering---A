# IHRMC-A (Improved Hierarchical Representation Multi-view Clustering - A)

## 项目概述

IHRMC-A 是一个改进的分层表示多视图聚类算法实现，基于 HRMC(Hierarchical Representation for Multi-view Clustering: From
Intra-sample to Intra-view to Inter-view) 算法进行了多项改进，包括锚点机制、缺失掩码机制和自适应权重。

该算法在 HRMC 的基础上进行了以下改进：
- **锚点表示学习**：使用锚点系数矩阵降低计算复杂度
- **缺失掩码机制**：处理视图缺失数据，提高算法鲁棒性
- **自适应权重**：动态调整视图权重，优化聚类性能
- **张量低秩约束**：捕捉视图间的高阶相关性
- **ADMM 优化**：高效求解复杂的优化问题

## 目录结构

```
IHRMC-A/
├── agent/                  # 智能体/模型模块
│   ├── __init__.py
│   └── IHRMC_A.py         # IHRMC-A 主模型实现
├── config/                 # 配置管理模块
│   ├── __init__.py
│   ├── config.py          # Config 类和命令行参数解析
│   └── constants.py       # 常量定义
├── data/                   # 数据加载模块
│   ├── __init__.py
│   ├── dataloader.py      # DataLoader 类
│   ├── MSRC.mat          # MSRC 数据集
│   ├── BBC.mat            # BBC 数据集
│   ├── BBCSport.mat       # BBCSport 数据集
│   └── 100leaves.mat      # 100leaves 数据集
├── draw/                   # 可视化模块
│   ├── __init__.py
│   └── visualization.py   # Visualizer 类
├── net/                    # 神经网络模块
│   ├── __init__.py
│   └── network.py         # Autoencoder, SingleLayerAE
├── solver/                 # 求解器模块
│   ├── __init__.py
│   └── solve.py          # SSC, t-SVT, TNN 等算法
├── utils/                  # 工具函数模块
│   ├── __init__.py
│   ├── logger.py          # 日志模块
│   ├── profiling.py       # 性能分析
│   ├── validators.py      # 参数验证
│   ├── utils.py           # 工具函数
│   └── output_manager.py  # 输出管理
├── configs/                # 配置文件
│   └── default_config.yaml # 默认配置
├── results/                # 实验输出
│   ├── single/            # 单次实验结果
│   ├── ablation/          # 消融实验结果
│   ├── sensitivity/       # 参数敏感性分析结果
│   └── robustness/        # 鲁棒性测试结果
├── main.py                 # 主脚本
├── compare.py              # 对比实验脚本
├── ablation_test.py        # 消融实验脚本
├── sensitivity_test.py     # 参数敏感性分析脚本
├── robustness_test.py      # 鲁棒性测试脚本
└── README.md              # 项目说明
```

## 模块说明

### agent 模块
- **`IHRMC_A.py`**：IHRMC-A 主模型实现，基于 ADMM 优化算法，支持锚点表示和掩码机制

### config 模块
- **`config.py`**：配置管理类，支持 YAML 配置文件和命令行参数
- **`constants.py`**：项目常量定义（默认参数、支持的数据集等）

### data 模块
- **`dataloader.py`**：多视图数据集加载和预处理，支持 TF-IDF、L2 归一化等

### draw 模块
- **`visualization.py`**：可视化功能，包括矩阵可视化、聚类结果、训练历史等

### net 模块
- **`network.py`**：自编码器网络结构和训练函数

### solver 模块
- **`solve.py`**：SSC（稀疏子空间聚类）、t-SVT（张量奇异值阈值）等算法

### utils 模块
- **`logger.py`**：统一的日志记录功能
- **`profiling.py`**：性能分析工具（计时器、内存监控）
- **`validators.py`**：参数验证功能
- **`utils.py`**：工具函数（聚类评估、矩阵分析等）
- **`output_manager.py`**：输出目录管理和重命名

## 核心功能

### IHRMC-A 模型特点

1. **锚点表示学习**：使用锚点系数矩阵降低计算复杂度，从 O(N²) 降低到 O(mN)
2. **缺失掩码机制**：处理视图缺失数据，提高算法在缺失视图情况下的鲁棒性
3. **自适应权重**：动态调整各视图的权重，优化聚类性能
4. **张量低秩约束**：使用 t-SVD 捕捉视图间的高阶相关性
5. **ADMM 优化**：高效求解复杂的优化问题
6. **谱聚类**：基于学习到的相似矩阵进行最终聚类

## 环境要求

- Python 3.8+
- PyTorch 1.10+
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- Seaborn
- PyYAML
- pandas
- openpyxl

## 安装

```bash
# 克隆项目
git clone <repository-url>
cd IHRMC-A

# 安装依赖
pip install torch numpy scipy scikit-learn matplotlib seaborn pyyaml pandas openpyxl
```

## 快速开始

### 1. 运行基本实验

```bash
python main.py
```

这将使用默认配置在 MSRC 数据集上运行 IHRMC-A 算法。

### 2. 对比实验

```bash
python compare.py
```

对比 HRMC 和 IHRMC-A 算法的性能差异。

### 3. 消融实验

```bash
python ablation_test.py
```

运行消融实验，验证锚点机制和掩码机制的有效性。实验包括：
- HRMC（原始算法）
- HRMC + 锚点机制
- HRMC + 掩码机制
- IHRMC-A（完整模型）

### 4. 参数敏感性分析

```bash
python sensitivity_test.py
```

分析各超参数对算法性能的影响。修改 `sensitivity_test.py` 中的 `PARAM_TO_ANALYZE` 变量选择要分析的参数：
- `lambda1`：网络权重 L2 正则化系数
- `lambda2`：自表达损失权重
- `lambda3`：张量核范数权重
- `beta`：ADMM 惩罚参数初始值
- `m`：锚点数

### 5. 鲁棒性测试

```bash
python robustness_test.py
```

测试算法对视图缺失率的鲁棒性，VMR 从 0.1 到 0.7，步长 0.1。

## 配置管理

### 配置文件

默认配置文件为 `configs/default_config.yaml`：

```yaml
# 数据集配置
dataset: 
  name: "MSRC"
  data_dir: "./data"
  normalize_method: l2
  missing_rate: 0
  missing_mode: "sample"
  missing_seed: 42

# 模型配置
model:
  # 核心超参数
  lambda1: 0.001    # 网络权重L2正则化系数
  lambda2: 0.07     # 自表达损失权重
  lambda3: 0.1      # 张量核范数权重
  beta: 0.0001      # ADMM 惩罚参数初始值
  m: 70             # 锚点数

  # 训练控制参数
  max_iters: 100    # ADMM最大迭代次数
  T: 60             # 自适应权重预热轮数
  iter_t: 10        # 每个 ADMM 迭代中自编码器的训练轮数
  batch_size: 1     # 批次大小
  epochs: 2         # 训练轮数
  lr: 0.0009765625  # 学习率 (2^-10)
  
  # 随机种子
  random_seed: 42
  
  # 功能开关
  use_anchor: True          # 是否使用锚点
  use_adaptive_weight: True # 是否使用自适应权重
  use_mask: True            # 是否使用缺失数据掩码
  lightweight_mode: True    # 轻量级模式
  use_pre_cache: False      # 是否使用预训练缓存
  
  # 模型结构参数
  hidden_dims: [200, 100]   # 自编码器隐藏层维度
  latent_dim: 100           # 自编码器潜在表示维度

  # 初始化和处理
  cv_init_method: 1  # Cv初始化方式
  negative_handling: 0  # S矩阵负值处理方法
  
  # 收敛参数
  tol: 1e-7         # 收敛容差
  beta_max: 10      # 惩罚参数最大值

# 训练配置
training:
  device: "cuda"  # 计算设备

# 输出配置
output:
  prefix: "exp"
  save_results: true
  save_logs: true
  generate_plots: true
```

## 数据集支持

| 数据集 | 样本数 | 视图数 | 类别数 | 说明 |
|--------|--------|--------|--------|------|
| MSRC | 210 | 4 | 7 | 多视图对象识别 |
| 100Leaves | 1600 | 3 | 100 | 叶子分类 |
| BBC | 685 | 4 | 5 | 新闻分类 |
| BBCSport | 544 | 2 | 5 | BBC 体育新闻 |

## 实验结果

### 输出目录结构

```
results/
├── single/                    # 单次实验结果
│   ├── MSRC/
│   │   └── 0417-1431-0.95-0.90-0.89-0.001-0.07-0.1-0.0001/
│   │       ├── experiment_results.json
│   │       ├── training_history.png
│   │       ├── S_matrix.png
│   │       └── confusion_matrix.png
├── ablation/                  # 消融实验结果
│   └── ablation_bbc_vmr0.3.xlsx
├── sensitivity/               # 参数敏感性分析结果
│   ├── sensitivity_MSRC_vmr0_lambda1.xlsx
│   ├── sensitivity_MSRC_vmr0_lambda2.xlsx
│   ├── sensitivity_MSRC_vmr0_lambda3.xlsx
│   ├── sensitivity_MSRC_vmr0_beta.xlsx
│   └── sensitivity_MSRC_vmr0_m.xlsx
└── robustness/                # 鲁棒性测试结果
    └── IHRMC_A_bbc_vmr_analysis.xlsx
```

### 结果文件

- `experiment_results.json`：实验参数和指标
- `*_training_history.png`：训练历史曲线
- `*_S_matrix.png`：相似矩阵可视化
- `*_confusion_matrix.png`：混淆矩阵
- `*.xlsx`：实验结果表格

## 核心算法

### IHRMC-A 模型

IHRMC-A 模型由以下几个关键部分组成：

1. **自编码器**：学习每个视图的低维非线性表示
2. **锚点表示学习**：使用锚点系数矩阵降低计算复杂度
3. **缺失掩码机制**：处理视图缺失数据
4. **自适应权重**：动态调整视图权重
5. **张量核范数约束**：捕捉视图间的高阶相关性
6. **ADMM 优化**：高效求解复杂的优化问题
7. **谱聚类**：基于学习到的相似矩阵进行最终聚类

### 优化过程

IHRMC-A 使用 ADMM 算法进行优化，主要步骤包括：

1. **更新自编码器参数**：预训练 + 微调
2. **更新锚点系数矩阵**：使用闭式解或 PGD
3. **更新统一相似矩阵**：各视图的加权平均
4. **更新张量变量**：使用 t-SVT 算法
5. **更新拉格朗日乘子**
6. **更新惩罚参数**

## 参数敏感性分析

### 参数范围

| 参数 | 范围 | 步长策略 |
|------|------|----------|
| lambda1 | 0.0001 - 0.1 | 分段线性（0.0001, 0.001, 0.005, 0.05） |
| lambda2 | 0.001 - 1.0 | 分段线性（0.001, 0.005, 0.05） |
| lambda3 | 0.001 - 1.0 | 分段线性（0.001, 0.005, 0.05） |
| beta | 0.0001 - 2.0 | 分段线性（0.0001, 0.001, 0.005, 0.05, 0.5） |
| m | c - n | 以 c 为间隔（c=聚类数, n=样本数） |

### 使用方法

修改 `sensitivity_test.py` 中的配置：

```python
# ========== 配置部分 ==========
PARAM_TO_ANALYZE = 'lambda1'  # 选择要分析的参数
DATASET_NAME = 'MSRC'          # 数据集名称
VMR = 0.1                      # 视图缺失率
RUNS = 1                       # 每个参数的运行次数
# ============================
```

然后运行：

```bash
python sensitivity_test.py
```

## 许可证

本项目采用 MIT 许可证。
