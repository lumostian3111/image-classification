"""
演示数据集准备脚本
使用 torchvision 内置数据集 (CIFAR-10) 自动下载并转换为 ImageFolder 格式
"""

import os
import sys
from pathlib import Path
import shutil

import torchvision
import torchvision.datasets as datasets
from PIL import Image


def prepare_cifar10(root_dir: str):
    """下载 CIFAR-10 并转换为 ImageFolder 格式"""
    root = Path(root_dir)
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"

    # 类别名称
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    print("正在下载 CIFAR-10 训练集...")
    train_data = datasets.CIFAR10(root=str(root), train=True, download=True)
    print(f"  训练集: {len(train_data)} 张")

    print("正在下载 CIFAR-10 测试集...")
    test_data = datasets.CIFAR10(root=str(root), train=False, download=True)
    print(f"  测试集: {len(test_data)} 张")

    # 创建目录
    for split, dataset, start_idx, end_idx in [
        ("train", train_data, 0, 40000),
        ("val", train_data, 40000, 50000),
        ("test", test_data, 0, 10000),
    ]:
        split_dir = root / split
        for cls_name in class_names:
            os.makedirs(split_dir / cls_name, exist_ok=True)

        print(f"\n处理 {split} 集 ({start_idx}-{end_idx})...")
        for idx in range(start_idx, end_idx):
            img, label = dataset[idx]
            cls_name = class_names[label]
            save_path = split_dir / cls_name / f"{idx:05d}.png"
            img.save(str(save_path))

        print(f"  {split}: {end_idx - start_idx} 张保存完成")

    # 清理 CIFAR-10 原始文件
    for f in root.glob("cifar-10-*"):
        if f.is_dir():
            shutil.rmtree(f)

    print(f"\n[完成] 数据集准备完成! 共 {len(class_names)} 个类别")
    print(f"   训练集: 40000 张")
    print(f"   验证集: 10000 张")
    print(f"   测试集: 10000 张")
    print(f"   类别: {class_names}")


def prepare_flowers102(root_dir: str):
    """下载 Flowers102 并转换为 ImageFolder 格式 (适合毕设演示)"""
    root = Path(root_dir)

    print("正在下载 Oxford 102 Flowers 数据集...")
    train_data = datasets.Flowers102(root=str(root), split="train", download=True)
    val_data = datasets.Flowers102(root=str(root), split="val", download=True)
    test_data = datasets.Flowers102(root=str(root), split="test", download=True)

    print(f"  训练集: {len(train_data)} 张")
    print(f"  验证集: {len(val_data)} 张")
    print(f"  测试集: {len(test_data)} 张")

    # 获取类别名称 (Flowers102 使用数字 ID)
    class_names = [f"flower_{i}" for i in range(102)]

    for split_name, dataset in [
        ("train", train_data),
        ("val", val_data),
        ("test", test_data),
    ]:
        split_dir = root / split_name
        for cls_name in class_names:
            os.makedirs(split_dir / cls_name, exist_ok=True)

        print(f"\n处理 {split_name} 集...")
        for idx in range(len(dataset)):
            img, label = dataset[idx]
            cls_name = class_names[label]
            save_path = split_dir / cls_name / f"{idx:05d}.jpg"
            img.save(str(save_path))

        print(f"  {split_name}: {len(dataset)} 张保存完成")

    # 清理原始文件
    for f in root.glob("flowers-102*"):
        if f.is_dir():
            shutil.rmtree(f)

    print(f"\n[完成] 数据集准备完成! 共 102 个类别")
    print(f"   训练集: {len(train_data)} 张")
    print(f"   验证集: {len(val_data)} 张")
    print(f"   测试集: {len(test_data)} 张")


if __name__ == "__main__":
    dataset_choice = sys.argv[1] if len(sys.argv) > 1 else "cifar10"

    from config import Config

    if dataset_choice.lower() == "flowers102":
        prepare_flowers102(Config.DATA_DIR)
    else:
        prepare_cifar10(Config.DATA_DIR)
