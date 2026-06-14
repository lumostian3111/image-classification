"""
图像分类系统 - 工具模块
包含训练辅助、日志记录、可视化等功能
"""

import os
import json
import time
import random
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from torch.utils.tensorboard import SummaryWriter

from config import Config


# ==================== 随机种子 ====================

def set_seed(seed: int = 42):
    """固定随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================== 日志与保存 ====================

class AverageMeter:
    """跟踪并计算平均值和当前值"""

    def __init__(self, name: str, fmt: str = ":6.4f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = f"{self.name} {self.fmt} ({self.fmt})"
        return fmtstr.format(self.val, self.avg)


def get_timestamp() -> str:
    """获取格式化的时间戳"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_time(seconds: float) -> str:
    """格式化时间为可读字符串"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"


# ==================== 模型保存与加载 ====================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_acc: float,
    class_names: List[str],
    filename: Optional[str] = None,
    is_best: bool = False,
):
    """保存检查点"""
    if filename is None:
        filename = f"checkpoint_epoch_{epoch+1}.pth"

    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_acc": best_acc,
        "class_names": class_names,
        "model_name": Config.MODEL_NAME,
        "num_classes": Config.NUM_CLASSES,
        "timestamp": get_timestamp(),
    }

    filepath = os.path.join(Config.CHECKPOINT_DIR, filename)
    torch.save(checkpoint, filepath)
    print(f"  [保存] 检查点已保存到: {filepath}")

    if is_best:
        best_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        torch.save(checkpoint, best_path)
        print(f"  [最佳] 最佳模型已保存到: {best_path}")


def load_checkpoint(filepath: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> dict:
    """加载检查点"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"检查点文件不存在: {filepath}")

    checkpoint = torch.load(filepath, map_location=Config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"  [加载] 已从 {filepath} 加载模型 (Epoch {checkpoint['epoch']}, "
          f"Best Acc: {checkpoint['best_acc']:.4f})")

    return checkpoint


# ==================== 评估指标 ====================

def compute_metrics(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    class_names: List[str],
) -> Dict:
    """计算所有分类评估指标"""
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    # 每个类别的指标
    per_class_precision, per_class_recall, per_class_f1, per_class_support = (
        precision_recall_fscore_support(all_labels, all_preds, zero_division=0)
    )

    metrics = {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "per_class": {},
    }

    for i, name in enumerate(class_names):
        metrics["per_class"][name] = {
            "precision": round(per_class_precision[i], 4),
            "recall": round(per_class_recall[i], 4),
            "f1": round(per_class_f1[i], 4),
            "support": int(per_class_support[i]),
        }

    return metrics


# ==================== 可视化 ====================

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True,
):
    """绘制混淆矩阵"""
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.8),
                                     max(5, len(class_names) * 0.7)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 在格子中标注数值
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    ax.set_title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")
    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(Config.OUTPUT_DIR, "confusion_matrix.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [保存] 混淆矩阵已保存到: {save_path}")

    return save_path


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
):
    """绘制训练历史曲线"""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss 曲线
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=1.5)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy 曲线
    axes[1].plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=1.5)
    axes[1].plot(epochs, history["val_acc"], "r-", label="Val Acc", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(Config.OUTPUT_DIR, "training_history.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [保存] 训练曲线已保存到: {save_path}")

    return save_path


def plot_sample_predictions(
    images: List[torch.Tensor],
    true_labels: List[str],
    pred_labels: List[str],
    class_names: List[str],
    save_path: Optional[str] = None,
    num_samples: int = 16,
):
    """可视化预测结果样本"""
    num_samples = min(num_samples, len(images))
    cols = 4
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if num_samples > 1 else [axes]

    from dataset import denormalize

    for idx in range(num_samples):
        img = denormalize(images[idx].cpu()).clamp(0, 1)
        img = img.permute(1, 2, 0).numpy()
        axes[idx].imshow(img)
        color = "green" if true_labels[idx] == pred_labels[idx] else "red"
        axes[idx].set_title(
            f"True: {true_labels[idx]}\nPred: {pred_labels[idx]}",
            color=color, fontsize=9,
        )
        axes[idx].axis("off")

    for idx in range(num_samples, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(Config.OUTPUT_DIR, "sample_predictions.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [保存] 预测样本已保存到: {save_path}")

    return save_path


# ==================== TensorBoard ====================

class TensorBoardLogger:
    """TensorBoard 日志记录器"""

    def __init__(self, log_dir: Optional[str] = None):
        log_dir = log_dir or os.path.join(Config.LOG_DIR, get_timestamp())
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"  TensorBoard 日志目录: {log_dir}")

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        self.writer.add_histogram(tag, values, step)

    def log_graph(self, model: nn.Module, input_tensor: torch.Tensor):
        self.writer.add_graph(model, input_tensor)

    def close(self):
        self.writer.close()


# ==================== 类别信息 ====================

def save_class_names(class_names: List[str], filepath: Optional[str] = None):
    """保存类别名称到 JSON 文件"""
    if filepath is None:
        filepath = os.path.join(Config.OUTPUT_DIR, "class_names.json")

    data = {
        "num_classes": len(class_names),
        "class_names": class_names,
        "class_to_idx": {name: i for i, name in enumerate(class_names)},
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  [保存] 类别信息已保存到: {filepath}")


def load_class_names(filepath: str) -> List[str]:
    """从 JSON 文件加载类别名称"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["class_names"]
