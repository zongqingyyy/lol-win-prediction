# 基于飞桨的《英雄联盟》对局胜负预测

使用百度飞桨（PaddlePaddle）构建 Wide & Deep 模型，根据《英雄联盟》单局玩家行为数据预测对局胜负。项目完整覆盖特征工程、模型训练、5 折交叉验证集成与提交文件生成。

## 项目简介

- 训练集 180,000 条玩家对局记录，测试集 20,000 条；每条记录包含 40 个原始特征和标签 `win`（0 失败 / 1 胜利）。
- 原始特征覆盖战斗、伤害、生存治疗、视野控制四类，存在明显的长尾分布与量纲差异。
- 经过三轮迭代，测试准确率从 **82.34%** 提升至 **86.045%**：

| 版本 | 特征工程 | 模型结构 | 验证准确率 | 测试准确率 |
| --- | --- | --- | --- | --- |
| V1 | 7 个基础比率特征，无标准化 | 3 层 MLP | 82.1% | 82.34% |
| V2 | 52 维增强特征 + StandardScaler | 残差网络 | 85.2% | 85.54% |
| V3 | 60+ 维特征 + 分桶特征 + Yeo-Johnson 变换 | Wide & Deep + 5 折集成 | 86.05% | **86.045%** |

仓库中的 `train.py` 对应最终的 **V3** 方案。

## 方法要点

**特征工程**

- 比率与效率特征：KDA、击杀/死亡比、伤害占比、承受伤害比、视野效率、治疗效率、推塔伤害占比、单位金币伤害等。
- 多杀特征：`multikill_score` 对双杀、三杀、四杀、五杀加权，并保留是否发生五杀/四杀的二值特征。
- 分桶特征：对多杀分、击杀、死亡、助攻按游戏经验划分 4 档，为 Wide 部分提供显式规则。
- 数据变换：使用 `PowerTransformer(method="yeo-johnson")` 处理右偏的伤害、治疗类特征，使其接近正态分布。

**模型结构**

- Wide 部分：单层线性层，记忆低阶特征组合。
- Deep 部分：`input → 1024 → 512 → 256 → 64 → 1` 的全连接网络，配合 BatchNorm、ReLU 与逐层递增的 Dropout。
- 两部分 logit 相加后经 Sigmoid 输出胜率，使用 `BCELoss` 训练。

**训练策略**

- `AdamW` 优化器，权重衰减 `1e-4`，初始学习率 `0.003`。
- `ReduceOnPlateau` 根据验证损失衰减学习率（`factor=0.5`，`patience=5`，最低 `1e-6`）。
- 早停：验证准确率连续 20 轮不提升则终止。
- `StratifiedKFold` 5 折集成：每折独立训练并保存最佳模型，测试集取 5 个模型预测概率的算术平均，以 0.5 为阈值输出类别。

## 快速开始

1. 安装依赖：

```bash
pip install -r requirements.txt
```

> 如需 GPU 训练，请根据本机 CUDA 版本安装对应的 `paddlepaddle-gpu`。

2. 将数据放入 `data/` 目录（`pandas` 可直接读取 zip 内的单个 CSV）：

```text
data/
├── train.csv.zip
└── test.csv.zip
```

3. 运行完整训练：

```bash
python train.py --train data/train.csv.zip --test data/test.csv.zip
```

4. 常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--train` / `--test` | `data/train.csv.zip` / `data/test.csv.zip` | 数据路径 |
| `--output-dir` | `outputs` | 模型与提交文件目录 |
| `--folds` | `5` | 交叉验证折数 |
| `--epochs` | `200` | 每折最大训练轮数 |
| `--batch-size` | `512` | 批大小 |
| `--lr` | `0.003` | 初始学习率 |
| `--patience` | `20` | 早停耐心值 |
| `--device` | `auto` | `auto` / `cpu` / `gpu` |

快速验证流程是否跑通（不追求精度）：

```bash
python train.py --folds 2 --epochs 3 --batch-size 64
```

## 输出

```text
outputs/
├── models/
│   └── best_fold{0..4}.pdparams   # 每折最佳模型参数
├── submission.csv                 # 测试集预测结果（win 列）
└── submission.zip                 # 可直接提交的压缩包
```

## 目录结构

```text
lol-win-prediction/
├── train.py           # 训练、集成与提交脚本
├── requirements.txt   # Python 依赖
├── .gitignore         # 忽略数据、模型与输出
└── README.md
```

## 说明

- 数据集体积较大且受赛题约束，**未包含在仓库中**，请自行放入 `data/` 目录。
- 课程报告的完整正文（含封面、个人信息与评审表）未上传，本 README 仅摘取赛题背景、方法设计与实验结果等适合公开的部分。

## 参考

1. Cheng H T, Koc L, Harmsen J, et al. Wide & Deep Learning for Recommender Systems. DLRS, 2016.
2. He K, Zhang X, Ren S, et al. Deep Residual Learning for Image Recognition. CVPR, 2016.
3. Ioffe S, Szegedy C. Batch Normalization. ICML, 2015.
4. Yeo I K, Johnson R A. A New Family of Power Transformations to Improve Normality or Symmetry. Biometrika, 2000.
