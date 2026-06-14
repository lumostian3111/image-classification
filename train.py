"""
图像分类系统 - 训练模块
实现完整的训练循环，包含验证、早停、学习率调度、TensorBoard 记录
"""

import os
import sys
import copy
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from config import Config
from dataset import create_dataloaders
from models import create_model, count_parameters
from utils import (
    AverageMeter,
    format_time,
    compute_metrics,
    save_checkpoint,
    TensorBoardLogger,
    save_class_names,
    set_seed,
    get_timestamp,
)


def get_optimizer(model: nn.Module) -> optim.Optimizer:
    """根据配置创建优化器"""
    name = Config.OPTIMIZER.lower()
    params = model.parameters()

    if name == "adam":
        return optim.Adam(params, lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    elif name == "adamw":
        return optim.AdamW(params, lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    elif name == "sgd":
        return optim.SGD(
            params, lr=Config.LEARNING_RATE,
            momentum=Config.MOMENTUM, weight_decay=Config.WEIGHT_DECAY,
        )
    else:
        raise ValueError(f"不支持的优化器: {name}，可选: adam, adamw, sgd")


def get_scheduler(optimizer: optim.Optimizer) -> optim.lr_scheduler.LRScheduler:
    """创建学习率调度器"""
    return lr_scheduler.StepLR(
        optimizer, step_size=Config.LR_STEP_SIZE, gamma=Config.LR_GAMMA
    )


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, val_acc: float) -> bool:
        score = val_acc

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.should_stop


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: Optional[TensorBoardLogger] = None,
) -> Tuple[float, float]:
    """训练一个 epoch"""
    model.train()

    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1:3d} [Train]", ncols=100)
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        batch_size = images.size(0)
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()

        losses.update(loss.item(), batch_size)
        top1.update(acc.item(), batch_size)

        # 更新进度条
        pbar.set_postfix({"Loss": f"{losses.avg:.4f}", "Acc": f"{top1.avg:.4f}"})

        # TensorBoard 记录 (每 LOG_INTERVAL 个 batch)
        if writer and batch_idx % Config.LOG_INTERVAL == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.log_scalar("Train/Loss_step", loss.item(), global_step)
            writer.log_scalar("Train/Acc_step", acc.item(), global_step)

    return losses.avg, top1.avg


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: Optional[TensorBoardLogger] = None,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """验证模型"""
    model.eval()

    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")

    all_labels = []
    all_preds = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1:3d} [Val  ]", ncols=100)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = images.size(0)
        _, preds = torch.max(outputs, 1)

        losses.update(loss.item(), batch_size)
        top1.update((preds == labels).float().mean().item(), batch_size)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        pbar.set_postfix({"Loss": f"{losses.avg:.4f}", "Acc": f"{top1.avg:.4f}"})

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # TensorBoard 记录 epoch 级别指标
    if writer:
        writer.log_scalar("Val/Loss", losses.avg, epoch + 1)
        writer.log_scalar("Val/Acc", top1.avg, epoch + 1)

    return losses.avg, top1.avg, all_labels, all_preds


def train(
    resume_from: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    主训练函数

    参数:
        resume_from: 检查点路径，用于恢复训练

    返回:
        训练历史记录
    """
    set_seed(42)
    Config.create_dirs()
    Config.print_config()

    device = torch.device(Config.DEVICE)
    print(f"\n  使用设备: {device}")

    # ---- 1. 加载数据 ----
    print("\n" + "-" * 50)
    print("  [1/5] 加载数据集...")
    print("-" * 50)
    train_loader, val_loader, test_loader, class_names = create_dataloaders()

    print(f"\n  类别数: {len(class_names)}")
    print(f"  类别名称: {class_names}")
    save_class_names(class_names)

    # ---- 2. 创建模型 ----
    print("\n" + "-" * 50)
    print("  [2/5] 创建模型...")
    print("-" * 50)
    model = create_model(num_classes=len(class_names))
    model = model.to(device)

    # 打印参数统计
    params = count_parameters(model)
    print(f"  总参数量:     {params['total_millions']}M")
    print(f"  可训练参数量: {params['trainable_millions']}M")
    print(f"  冻结参数量:   {params['frozen'] / 1e6:.2f}M")

    # ---- 3. 损失函数与优化器 ----
    print("\n" + "-" * 50)
    print("  [3/5] 配置优化器与损失函数...")
    print("-" * 50)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    print(f"  损失函数: CrossEntropyLoss")
    print(f"  优化器:   {Config.OPTIMIZER} (lr={Config.LEARNING_RATE})")
    print(f"  调度器:   StepLR (step={Config.LR_STEP_SIZE}, gamma={Config.LR_GAMMA})")

    # ---- 恢复训练 ----
    start_epoch = 0
    best_acc = 0.0
    if resume_from:
        checkpoint = load_checkpoint(resume_from, model, optimizer)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]

    # ---- 4. TensorBoard ----
    writer = TensorBoardLogger() if Config.USE_TENSORBOARD else None

    # ---- 5. 训练循环 ----
    print("\n" + "-" * 50)
    print("  [4/5] 开始训练...")
    print("-" * 50)

    early_stopping = EarlyStopping(patience=Config.PATIENCE, min_delta=Config.MIN_DELTA) if Config.EARLY_STOPPING else None

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    total_start = time.time()

    for epoch in range(start_epoch, Config.EPOCHS):
        epoch_start = time.time()

        # 当前学习率
        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )

        # 验证
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, device, epoch, writer
        )

        # 更新学习率
        scheduler.step()

        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start

        # 打印 epoch 摘要
        print(
            f"\n  Epoch {epoch+1:3d}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f} | Time: {format_time(epoch_time)}"
        )

        # TensorBoard epoch 汇总
        if writer:
            writer.log_scalars("Epoch/Summary", {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }, epoch + 1)
            writer.log_scalar("LR", current_lr, epoch + 1)

        # 保存最佳模型
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc

        if (epoch + 1) % 5 == 0 or is_best:
            save_checkpoint(
                model, optimizer, epoch, best_acc, class_names, is_best=is_best
            )

        # 早停检查
        if early_stopping and Config.EARLY_STOPPING:
            if early_stopping(val_acc):
                print(f"\n  ⏹ 早停触发! 最佳验证准确率: {best_acc:.4f}")
                break

    total_time = time.time() - total_start
    print(f"\n  ✅ 训练完成! 总耗时: {format_time(total_time)}")
    print(f"  🏆 最佳验证准确率: {best_acc:.4f}")

    # ---- 保存最终模型 ----
    save_checkpoint(model, optimizer, Config.EPOCHS - 1, best_acc, class_names,
                    filename="final_model.pth")

    if writer:
        writer.close()

    return history
