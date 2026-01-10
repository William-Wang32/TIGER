# TIGER 项目运行指南

## 1. 代码结构说明

以下是项目中主要文件和目录的功能说明：

### 核心代码
- **`main.py`**:
  程序的主入口。负责参数解析、数据加载、模型初始化以及整个训练流程的控制。
- **`train_eval.py`**:
  包含具体的训练 (`train`)、测试 (`test`) 和评估 (`eval`) 函数。定义了每个 epoch 的循环逻辑以及指标计算（F1, ROC-AUC, PR-AUC 等）。
- **`data_process.py`**:
  数据预处理脚本。负责将 SMILES 字符串转换为分子图、生成子图、读取网络结构等。
- **`utils.py`**:
  包含各种通用工具函数。

### 模型定义 (`model/`)
- **`model/tiger.py`**: TIGER 模型的主体架构定义。
- **`model/GraphTransformer.py`**: 这里的 Graph Transformer 组件实现。

### 辅助模块 (`randomWalk/`)
- **`randomWalk/walker.py`, `randomWalk/node2vec.py`**: 实现随机游走算法，用于图特征提取。

### 运行脚本
- **`run_drugbank_experiments.sh`**: 针对 **DrugBank** 数据集的一键运行脚本，依次运行多种提取器配置 (randomWalk, khop-subtree, probability)。
- **`run_kegg_experiments.sh`**: 针对 **KEGG** 数据集的一键运行脚本。
- **`run_ogbl_biokg_experiments.sh`**: 针对 **OGB-BioKG** 数据集的一键运行脚本。

### 数据目录
- **`dataset/`**: 存放原始数据集文件（如 DrugBank, KEGG, OGB-Biokg 的数据），为方便提交作业并未置入数据，需手动粘贴。
- **`data/`**: 存放预处理后的数据。
- **`best_save/`**: 存放训练结果，包括训练日志和最终的平均指标(ACC、F1、AUC、AUPR)。

## 2. 必要的配置文件

本项目**不依赖**外部的配置文件（如 `.yaml` 或 `.json` 配置文件）。所有的模型超参数、数据集路径配置和训练选项均通过 **命令行参数 (Argparse)** 在 `main.py` 中定义。

可以在 `main.py` 的 `init_args` 函数中查看所有可用的参数及其默认值。

关键参数说明：
- `--dataset`: 选择数据集 (`drugbank`, `kegg`, `ogbl-biokg`)。
- `--extractor`: 选择子图提取策略 (`randomWalk`, `khop-subtree`, `probability`)。
- `--fusion`: 双通道融合策略 (`concat`, `gated`)。
- `--mi_objective`: 互信息目标函数 (`bce`, `infonce`)。
- `--attn_norm`: Attention 归一化方式 (`softmax`, `ratio`)。
- `--model_episodes`: 训练轮次。

## 3. 如何运行代码

### 方法一：使用预置脚本

项目提供了针对三个主要数据集的 Shell 脚本，可以一键运行完整的实验流程(原版未改进)。

1.  **给脚本添加执行权限**：
    ```bash
    chmod +x run_drugbank_experiments.sh
    chmod +x run_kegg_experiments.sh
    chmod +x run_ogbl_biokg_experiments.sh
    ```

2.  **运行脚本**：
    例如，运行 DrugBank 数据集的实验：
    ```bash
    ./run_drugbank_experiments.sh
    ```
    该脚本会依次执行使用不同提取器（RandomWalk, K-hop Subtree, Probability）的实验。

### 方法二：手动运行 `main.py`

您也可以直接运行 Python 命令，手动指定参数进行实验。

**示例 1：在 DrugBank 数据集上使用 RandomWalk 提取器运行**
```bash
python main.py --dataset drugbank --extractor randomWalk --attn_norm ratio --attn_dropout 0.0 --edge_dropout 0.0 --mi_objective bce
# 与 run_drugbank_experiments.sh 等脚本中的指令相同
```

**示例 2：改进方法的运行**
使用 `infonce` 作为互信息目标，`softmax` 作为注意力归一化，在 DrugBank 数据集上使用 RandomWalk 提取器：
```bash
python main.py --dataset drugbank --extractor randomWalk \
    --attn_norm softmax --attn_dropout 0.1 --edge_dropout 0.1 \
    --mi_objective infonce --mi_temp 0.2
```

## 4. 输出结果

训练过程中的日志会打印在控制台。
模型训练的最佳结果（模型权重或指标记录）通常保存在 `best_save/` 目录下，根据不同的数据集和提取器分文件夹存储。
