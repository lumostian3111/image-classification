"""
图像分类系统 - 配置文件
集中管理所有可调参数
"""

import os
import torch


class Config:
    # ==================== 路径配置 ====================
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "val")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
    LOG_DIR = os.path.join(ROOT_DIR, "logs")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

    # ==================== 数据集配置 ====================
    # 图像输入尺寸
    IMAGE_SIZE = 224
    # 批次大小
    BATCH_SIZE = 32
    # 数据加载线程数
    NUM_WORKERS = 4
    # 验证集比例 (当未预先划分时，从训练集划分)
    VAL_SPLIT = 0.2
    # 类别名称 (自动从文件夹检测，也可以手动指定)
    CLASS_NAMES = None  # None = 自动检测

    # ==================== 模型配置 ====================
    # 模型类型: "resnet18", "resnet34", "resnet50", "efficientnet_b0",
    #          "mobilenet_v2", "densenet121", "vgg16", "custom_cnn"
    MODEL_NAME = "resnet18"
    # 是否使用预训练权重
    PRETRAINED = True
    # 分类类别数 (自动检测)
    NUM_CLASSES = None
    # Dropout 比例
    DROPOUT = 0.5

    # ==================== 训练配置 ====================
    # 训练轮数
    EPOCHS = 50
    # 学习率
    LEARNING_RATE = 0.001
    # 学习率调度器步长 (每多少轮衰减)
    LR_STEP_SIZE = 15
    # 学习率衰减因子
    LR_GAMMA = 0.1
    # 权重衰减 (L2 正则化)
    WEIGHT_DECAY = 1e-4
    # 优化器: "adam", "sgd", "adamw"
    OPTIMIZER = "adam"
    # SGD 动量
    MOMENTUM = 0.9

    # ==================== 早停配置 ====================
    # 是否启用早停
    EARLY_STOPPING = True
    # 容忍轮数
    PATIENCE = 10
    # 监控指标最小变化阈值
    MIN_DELTA = 0.001

    # ==================== 设备配置 ====================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==================== 日志配置 ====================
    # 是否使用 TensorBoard
    USE_TENSORBOARD = True
    # 每多少批次打印一次训练日志
    LOG_INTERVAL = 10

    # ==================== 数据增强配置 ====================
    # 训练集增强
    TRAIN_AUGMENTATION = {
        "random_resized_crop": True,
        "random_horizontal_flip": True,
        "random_vertical_flip": False,
        "random_rotation": 15,       # 旋转角度范围
        "color_jitter": {             # 颜色抖动
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.1,
        },
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    }

    # 验证/测试集预处理
    VAL_AUGMENTATION = {
        "resize": 256,
        "center_crop": 224,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    }

    @classmethod
    def create_dirs(cls):
        """创建必要的目录"""
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.TRAIN_DIR, exist_ok=True)
        os.makedirs(cls.VAL_DIR, exist_ok=True)
        os.makedirs(cls.TEST_DIR, exist_ok=True)

    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("\n" + "=" * 60)
        print(" " * 15 + "系统配置")
        print("=" * 60)
        print(f"  设备:                {cls.DEVICE}")
        print(f"  模型:                {cls.MODEL_NAME}")
        print(f"  预训练:              {cls.PRETRAINED}")
        print(f"  图像尺寸:            {cls.IMAGE_SIZE}x{cls.IMAGE_SIZE}")
        print(f"  批次大小:            {cls.BATCH_SIZE}")
        print(f"  训练轮数:            {cls.EPOCHS}")
        print(f"  学习率:              {cls.LEARNING_RATE}")
        print(f"  优化器:              {cls.OPTIMIZER}")
        print(f"  早停:                {cls.EARLY_STOPPING}")
        print(f"  设备:                {cls.DEVICE}")
        print("=" * 60 + "\n")
