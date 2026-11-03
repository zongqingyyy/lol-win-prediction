"""基于飞桨 Wide & Deep 的《英雄联盟》对局胜负预测。

原始数据包含玩家单局行为特征，标签 win 表示该局是否获胜（0/1）。
本脚本完成特征工程、Yeo-Johnson 变换、Wide & Deep 训练、分层 K 折集成与提交文件生成。

用法：
    python train.py --train data/train.csv.zip --test data/test.csv.zip
"""

import argparse
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PowerTransformer

warnings.filterwarnings("ignore", category=UserWarning, module="paddle.nn.layer.norm")

SEED = 2024


def parse_args():
    parser = argparse.ArgumentParser(description="英雄联盟对局胜负预测（Wide & Deep + 5 折集成）")
    parser.add_argument("--train", default="data/train.csv.zip", help="训练集路径（csv 或 csv.zip）")
    parser.add_argument("--test", default="data/test.csv.zip", help="测试集路径（csv 或 csv.zip）")
    parser.add_argument("--output-dir", default="outputs", help="模型与提交文件输出目录")
    parser.add_argument("--folds", type=int, default=5, help="交叉验证折数")
    parser.add_argument("--epochs", type=int, default=200, help="每折最大训练轮数")
    parser.add_argument("--batch-size", type=int, default=512, help="批大小")
    parser.add_argument("--lr", type=float, default=0.003, help="初始学习率")
    parser.add_argument("--patience", type=int, default=20, help="验证准确率早停耐心值")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"], help="运行设备")
    return parser.parse_args()


def add_features(df):
    """构造比率、效率、多杀与分桶特征。"""
    df = df.copy()
    df["kda"] = (df["kills"] + df["assists"]) / (df["deaths"] + 1)
    df["kill_death_ratio"] = df["kills"] / (df["deaths"] + 1)
    df["assist_death_ratio"] = df["assists"] / (df["deaths"] + 1)
    df["multikill_score"] = (
        df["doublekills"] * 1
        + df["triplekills"] * 2
        + df["quadrakills"] * 3
        + df["pentakills"] * 4
    )
    df["has_pentakill"] = (df["pentakills"] > 0).astype(int)
    df["has_quadrakill"] = (df["quadrakills"] > 0).astype(int)

    total_dmg = df["totdmgdealt"] + df["totdmgtochamp"] + 1
    df["dmg_to_champ_ratio"] = df["totdmgtochamp"] / total_dmg
    df["dmg_dealt_ratio"] = df["totdmgdealt"] / total_dmg
    df["taken_to_total"] = df["totdmgtaken"] / total_dmg
    df["true_dmg_ratio"] = df["truedmgdealt"] / (df["totdmgdealt"] + 1)
    df["phys_to_magic"] = df["physicaldmgdealt"] / (df["magicdmgdealt"] + 1)
    df["phys_to_champ_ratio"] = df["physdmgtochamp"] / (df["totdmgtochamp"] + 1)

    df["survival_per_death"] = df["longesttimespentliving"] / (df["deaths"] + 1)
    df["vision_per_life"] = (df["wardsplaced"] + df["wardskilled"]) / (df["deaths"] + 1)
    df["vision_score"] = df["wardsplaced"] + df["wardskilled"]
    df["cc_per_death"] = df["timecc"] / (df["deaths"] + 1)
    df["turret_dmg_ratio"] = df["dmgtoturrets"] / (df["totdmgdealt"] + 1)
    df["heal_ratio"] = df["totheal"] / (df["totdmgtaken"] + 1)
    df["heal_units_ratio"] = df["totunitshealed"] / (df["totheal"] + 1)
    df["dmg_per_gold_est"] = df["totdmgtochamp"] / ((df["kills"] * 300 + df["assists"] * 150) + 1)

    # 分桶特征：把连续战斗指标映射为局势等级
    df["multikill_bin"] = pd.cut(df["multikill_score"], bins=[-1, 0, 2, 5, 100], labels=[0, 1, 2, 3]).astype(int)
    df["kills_bin"] = pd.cut(df["kills"], bins=[-1, 3, 8, 15, 100], labels=[0, 1, 2, 3]).astype(int)
    df["deaths_bin"] = pd.cut(df["deaths"], bins=[-1, 2, 5, 10, 100], labels=[0, 1, 2, 3]).astype(int)
    df["assists_bin"] = pd.cut(df["assists"], bins=[-1, 5, 15, 30, 100], labels=[0, 1, 2, 3]).astype(int)

    df.replace([np.inf, -np.inf], 0, inplace=True)
    return df


class WideDeep(nn.Layer):
    """Wide 部分记忆低阶规则，Deep 部分学习高阶非线性交互。"""

    def __init__(self, input_dim):
        super().__init__()
        self.wide = nn.Linear(input_dim, 1)
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1D(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1D(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1D(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.final = nn.Sigmoid()

    def forward(self, x):
        return self.final(self.wide(x) + self.deep(x))


def main():
    args = parse_args()

    if args.device == "auto":
        paddle.set_device("gpu" if paddle.is_compiled_with_cuda() else "cpu")
    else:
        paddle.set_device(args.device)
    paddle.seed(SEED)
    np.random.seed(SEED)

    print(f"使用设备: {paddle.get_device()}")

    output_dir = Path(args.output_dir)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # ================== 读数据 ==================
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    print(f"训练集: {train.shape}, 测试集: {test.shape}")

    # ================== 特征工程 ==================
    train = add_features(train)
    test = add_features(test)

    drop_cols = ["id", "win"]
    X = train.drop(columns=[c for c in drop_cols if c in train.columns])
    y = train["win"].values
    X_test = test.drop(columns=[c for c in drop_cols if c in test.columns])

    X = X.fillna(0)
    X_test = X_test.fillna(0)

    # ================== 标准化 ==================
    num_cols = X.columns.tolist()
    preprocessor = ColumnTransformer([("num", PowerTransformer(method="yeo-johnson"), num_cols)])
    X_scaled = preprocessor.fit_transform(X)
    X_test_scaled = preprocessor.transform(X_test)

    input_dim = X_scaled.shape[1]
    print(f"特征维度: {input_dim}")

    # ================== K 折集成 ==================
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    test_preds_proba = np.zeros((X_test_scaled.shape[0], args.folds))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
        print(f"\n{'=' * 50}")
        print(f"Fold {fold + 1}/{args.folds}")
        X_tr, X_v = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_v = y[train_idx], y[val_idx]

        train_x = paddle.to_tensor(X_tr, dtype="float32")
        train_y = paddle.to_tensor(y_tr.reshape(-1, 1), dtype="float32")
        val_x = paddle.to_tensor(X_v, dtype="float32")
        val_y = paddle.to_tensor(y_v.reshape(-1, 1), dtype="float32")

        model = WideDeep(input_dim)
        loss_fn = nn.BCELoss()
        scheduler = optim.lr.ReduceOnPlateau(
            learning_rate=args.lr,
            mode="min",
            factor=0.5,
            patience=5,
            threshold=1e-4,
            min_lr=1e-6,
        )
        optimizer = optim.AdamW(
            parameters=model.parameters(),
            learning_rate=scheduler,
            weight_decay=1e-4,
        )

        train_dataset = paddle.io.TensorDataset([train_x, train_y])
        train_loader = paddle.io.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )

        best_val_acc = 0.0
        no_improve = 0
        model_path = model_dir / f"best_fold{fold}.pdparams"

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            for batch_x, batch_y in train_loader:
                pred = model(batch_x)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
                total_loss += loss.item()

            model.eval()
            with paddle.no_grad():
                val_pred = model(val_x)
                val_loss = loss_fn(val_pred, val_y).item()
                val_pred_label = (val_pred.numpy() >= 0.5).astype(int).reshape(-1)
                val_acc = accuracy_score(y_v, val_pred_label)

            scheduler.step(val_loss)
            current_lr = optimizer.get_lr()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1:3d}/{args.epochs} | LR: {current_lr:.2e} | "
                    f"Train Loss: {total_loss / len(train_loader):.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                paddle.save(model.state_dict(), str(model_path))
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"早停于 Epoch {epoch + 1}")
                    break

        print(f"Fold {fold + 1} 最佳验证准确率: {best_val_acc:.4f}")
        fold_scores.append(best_val_acc)

        # 加载该折最佳模型，预测测试集
        model.set_state_dict(paddle.load(str(model_path)))
        model.eval()
        with paddle.no_grad():
            test_prob = model(paddle.to_tensor(X_test_scaled, dtype="float32")).numpy().flatten()
        test_preds_proba[:, fold] = test_prob

    # ================== 集成结果 ==================
    mean_proba = test_preds_proba.mean(axis=1)
    test_pred = (mean_proba >= 0.5).astype(int)

    print(f"\n各折验证准确率: {fold_scores}")
    print(f"平均验证准确率: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")

    # ================== 生成提交文件 ==================
    submission = pd.DataFrame({"win": test_pred})
    submission_path = output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False, encoding="utf-8")

    submission_zip = output_dir / "submission.zip"
    with zipfile.ZipFile(submission_zip, "w") as zf:
        zf.write(submission_path, arcname="submission.csv")

    print(f"已生成 {submission_zip}，可直接提交。")


if __name__ == "__main__":
    main()
