"""
图像分类系统 - 数据集模块
负责数据加载、预处理和数据增强
"""

import os
import random
from typing import Tuple, Optional, List

import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import datasets, transforms
from PIL import Image

from config import Config


def get_train_transforms() -> transforms.Compose:
    """获取训练集数据增强变换"""
    aug = Config.TRAIN_AUGMENTATION
    transform_list = []

    if aug.get("random_resized_crop", True):
        transform_list.append(
            transforms.RandomResizedCrop(Config.IMAGE_SIZE, scale=(0.8, 1.0))
        )
    else:
        transform_list.append(transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)))

    if aug.get("random_horizontal_flip", True):
        transform_list.append(transforms.RandomHorizontalFlip())

    if aug.get("random_vertical_flip", False):
        transform_list.append(transforms.RandomVerticalFlip())

    if aug.get("random_rotation", 0) > 0:
        transform_list.append(transforms.RandomRotation(aug["random_rotation"]))

    if aug.get("color_jitter"):
        cj = aug["color_jitter"]
        transform_list.append(
            transforms.ColorJitter(
                brightness=cj.get("brightness", 0),
                contrast=cj.get("contrast", 0),
                saturation=cj.get("saturation", 0),
                hue=cj.get("hue", 0),
            )
        )

    transform_list.append(transforms.ToTensor())
    transform_list.append(
        transforms.Normalize(
            mean=aug.get("normalize_mean", [0.485, 0.456, 0.406]),
            std=aug.get("normalize_std", [0.229, 0.224, 0.225]),
        )
    )

    return transforms.Compose(transform_list)


def get_val_transforms() -> transforms.Compose:
    """获取验证/测试集预处理变换"""
    aug = Config.VAL_AUGMENTATION
    transform_list = []

    resize = aug.get("resize", 256)
    crop = aug.get("center_crop", 224)

    transform_list.append(transforms.Resize(resize))
    transform_list.append(transforms.CenterCrop(crop))
    transform_list.append(transforms.ToTensor())
    transform_list.append(
        transforms.Normalize(
            mean=aug.get("normalize_mean", [0.485, 0.456, 0.406]),
            std=aug.get("normalize_std", [0.229, 0.224, 0.225]),
        )
    )

    return transforms.Compose(transform_list)


def get_test_transforms() -> transforms.Compose:
    """获取测试集预处理变换 (与验证集相同)"""
    return get_val_transforms()


def load_data(data_dir: str, train: bool = True) -> datasets.ImageFolder:
    """加载图像文件夹数据集"""
    transform = get_train_transforms() if train else get_val_transforms()
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    return dataset


def create_dataloaders(
    train_dir: Optional[str] = None,
    val_dir: Optional[str] = None,
    test_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    创建训练、验证、测试数据加载器

    返回:
        train_loader, val_loader, test_loader, class_names
    """
    train_dir = train_dir or Config.TRAIN_DIR
    val_dir = val_dir or Config.VAL_DIR
    test_dir = test_dir or Config.TEST_DIR

    # ---- 训练集 ----
    if os.path.isdir(train_dir) and len(os.listdir(train_dir)) > 0:
        full_train_dataset = load_data(train_dir, train=True)
        class_names = full_train_dataset.classes
        num_classes = len(class_names)
        Config.NUM_CLASSES = num_classes

        # 如果没有单独的验证集，从训练集中划分
        if not os.path.isdir(val_dir) or len(os.listdir(val_dir)) == 0:
            val_size = int(len(full_train_dataset) * Config.VAL_SPLIT)
            train_size = len(full_train_dataset) - val_size
            train_dataset, val_dataset = random_split(
                full_train_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            print(f"  [自动划分] 训练集: {train_size} 张, 验证集: {val_size} 张")
        else:
            train_dataset = full_train_dataset
            # 加载单独的验证集
            val_dataset = datasets.ImageFolder(
                root=val_dir, transform=get_val_transforms()
            )
            print(f"  [训练集] {len(train_dataset)} 张图片")
            print(f"  [验证集] {len(val_dataset)} 张图片")
    else:
        raise FileNotFoundError(
            f"训练集目录不存在或为空: {train_dir}\n"
            f"请将训练图片按类别放入子文件夹中。\n"
            f"示例结构:\n"
            f"  data/train/\n"
            f"    ├── cat/\n"
            f"    │   ├── cat001.jpg\n"
            f"    │   └── cat002.jpg\n"
            f"    └── dog/\n"
            f"        ├── dog001.jpg\n"
            f"        └── dog002.jpg"
        )

    # ---- 验证集 (若已从训练集划分则跳过) ----
    if "val_dataset" not in dir():
        if os.path.isdir(val_dir) and len(os.listdir(val_dir)) > 0:
            val_dataset = datasets.ImageFolder(
                root=val_dir, transform=get_val_transforms()
            )

    # ---- 测试集 ----
    if os.path.isdir(test_dir) and len(os.listdir(test_dir)) > 0:
        test_dataset = datasets.ImageFolder(
            root=test_dir, transform=get_test_transforms()
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        print(f"  [测试集] {len(test_dataset)} 张图片")
    else:
        # 没有测试集时，从验证集复制引用
        print(f"  [提示] 未找到测试集，将使用验证集替代")
        test_loader = None

    # ---- 创建 DataLoader ----
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    if test_loader is None:
        test_loader = val_loader

    return train_loader, val_loader, test_loader, class_names


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """反归一化图像张量，用于可视化"""
    mean = torch.tensor(Config.VAL_AUGMENTATION["normalize_mean"]).view(3, 1, 1)
    std = torch.tensor(Config.VAL_AUGMENTATION["normalize_std"]).view(3, 1, 1)
    return tensor * std + mean


class InferenceTransform:
    """单张图片推理时的预处理"""

    def __init__(self):
        self.transform = get_val_transforms()

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image).unsqueeze(0)  # 添加 batch 维度
