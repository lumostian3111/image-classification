"""
图像分类系统 - 模型定义模块
支持多种主流分类网络，通过配置切换
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

from config import Config


class CustomCNN(nn.Module):
    """自定义轻量级 CNN，适合小数据集快速实验"""

    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3x224x224 -> 32x112x112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 32x112x112 -> 64x56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 64x56x56 -> 128x28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 128x28x28 -> 256x14x14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 5: 256x14x14 -> 512x7x7
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_model(
    num_classes: Optional[int] = None,
    model_name: Optional[str] = None,
    pretrained: Optional[bool] = None,
) -> nn.Module:
    """
    创建分类模型

    参数:
        num_classes: 类别数 (None 则使用 Config.NUM_CLASSES)
        model_name: 模型名称
        pretrained: 是否使用预训练权重

    返回:
        PyTorch 模型
    """
    num_classes = num_classes or Config.NUM_CLASSES
    model_name = model_name or Config.MODEL_NAME
    pretrained = pretrained if pretrained is not None else Config.PRETRAINED

    if num_classes is None:
        raise ValueError("请先设置 NUM_CLASSES 或传入 num_classes 参数")

    print(f"\n  创建模型: {model_name} (类别数={num_classes}, 预训练={pretrained})")

    if model_name == "custom_cnn":
        model = CustomCNN(num_classes)

    elif model_name == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(Config.DROPOUT),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "resnet34":
        model = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(Config.DROPOUT),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(Config.DROPOUT),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights="IMAGENET1K_V1" if pretrained else None
        )
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(Config.DROPOUT),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(
            weights="IMAGENET1K_V2" if pretrained else None
        )
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(Config.DROPOUT),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "densenet121":
        model = models.densenet121(
            weights="IMAGENET1K_V1" if pretrained else None
        )
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(Config.DROPOUT),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "vgg16":
        model = models.vgg16(weights="IMAGENET1K_V1" if pretrained else None)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"不支持的模型: {model_name}\n"
            f"可选: custom_cnn, resnet18, resnet34, resnet50, "
            f"efficientnet_b0, mobilenet_v2, densenet121, vgg16"
        )

    return model


def count_parameters(model: nn.Module) -> dict:
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_millions": round(total / 1e6, 2),
        "trainable_millions": round(trainable / 1e6, 2),
    }
