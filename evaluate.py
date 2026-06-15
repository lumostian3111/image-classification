"""
图像分类系统 - 评估模块
在测试集上全面评估模型，生成混淆矩阵、分类报告和可视化
"""

import os
import json
from typing import Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import create_dataloaders, denormalize
from models import create_model
from utils import (
    compute_metrics,
    plot_confusion_matrix,
    plot_sample_predictions,
    save_class_names,
    load_checkpoint,
    get_timestamp,
    AverageMeter,
)


@torch.no_grad()
def evaluate(
    checkpoint_path: str = None,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    评估模型在测试集上的表现

    参数:
        checkpoint_path: 模型检查点路径
        output_dir: 输出目录

    返回:
        评估指标字典
    """
    Config.create_dirs()

    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if output_dir is None:
        output_dir = Config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(Config.DEVICE)
    print(f"\n  使用设备: {device}")

    # ---- 加载数据 ----
    print("\n" + "-" * 50)
    print("  [1/4] 加载测试数据...")
    print("-" * 50)

    # 先尝试加载数据以获取类别信息
    try:
        train_loader, val_loader, test_loader, class_names = create_dataloaders()
    except FileNotFoundError:
        # 如果没有训练集，从检查点加载 class_names
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        class_names = checkpoint.get("class_names", [])

    print(f"  类别数: {len(class_names)}")
    print(f"  类别名称: {class_names}")
    print(f"  测试样本数: {len(test_loader.dataset)}")

    # ---- 加载模型 ----
    print("\n" + "-" * 50)
    print("  [2/4] 加载模型...")
    print("-" * 50)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 从检查点恢复模型配置
    if "model_name" in checkpoint:
        Config.MODEL_NAME = checkpoint["model_name"]

    if "num_classes" in checkpoint:
        Config.NUM_CLASSES = checkpoint["num_classes"]
    else:
        Config.NUM_CLASSES = len(class_names)

    if len(class_names) == 0 and "class_names" in checkpoint:
        class_names = checkpoint["class_names"]

    model = create_model(
        num_classes=Config.NUM_CLASSES,
        model_name=Config.MODEL_NAME,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"  已加载模型 (Epoch {checkpoint.get('epoch', '?')}, "
          f"Best Acc: {checkpoint.get('best_acc', 0):.4f})")

    # ---- 推理 ----
    print("\n" + "-" * 50)
    print("  [3/4] 在测试集上推理...")
    print("-" * 50)

    all_labels = []
    all_preds = []
    all_probs = []
    sample_images = []
    sample_true = []
    sample_pred = []

    pbar = tqdm(test_loader, desc="  Evaluating", ncols=100)
    for images, labels in pbar:
        images = images.to(device)
        labels_gpu = labels.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        # 收集样本用于可视化
        if len(sample_images) < 16:
            for i in range(len(images)):
                if len(sample_images) >= 16:
                    break
                sample_images.append(images[i].cpu())
                sample_true.append(class_names[labels[i].item()])
                sample_pred.append(class_names[preds[i].item()])

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # ---- 计算指标 ----
    print("\n" + "-" * 50)
    print("  [4/4] 计算评估指标...")
    print("-" * 50)

    metrics = compute_metrics(all_labels, all_preds, class_names)

    # 打印结果
    print("\n" + "=" * 60)
    print(" " * 18 + "评估结果")
    print("=" * 60)
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"  Recall (weighted):    {metrics['recall_weighted']:.4f}")
    print(f"  F1 Score (weighted):  {metrics['f1_weighted']:.4f}")
    print("-" * 60)
    print(f"  {'Class':<20s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    print("-" * 60)

    for class_name, class_metrics in metrics["per_class"].items():
        print(
            f"  {class_name:<20s} "
            f"{class_metrics['precision']:>10.4f} "
            f"{class_metrics['recall']:>10.4f} "
            f"{class_metrics['f1']:>10.4f} "
            f"{class_metrics['support']:>10d}"
        )
    print("=" * 60)

    # ---- 保存结果 ----
    # 保存指标 JSON
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n  [保存] 评估指标已保存到: {metrics_path}")

    # 混淆矩阵
    cm = confusion_matrix_with_sklearn(all_labels, all_preds, len(class_names))
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, save_path=cm_path)

    # 预测样本可视化
    samples_path = os.path.join(output_dir, "sample_predictions.png")
    plot_sample_predictions(
        sample_images, sample_true, sample_pred,
        class_names, save_path=samples_path,
    )

    # 保存类别名称
    save_class_names(class_names, os.path.join(output_dir, "class_names.json"))

    return metrics


def confusion_matrix_with_sklearn(y_true, y_pred, num_classes):
    """使用 sklearn 计算混淆矩阵"""
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_true, y_pred, labels=range(num_classes))
