# 图像分类系统

基于 PyTorch 的深度学习图像分类系统，支持多种主流模型架构，提供完整的训练、评估和推理流程。

## 功能特性

- **多模型支持**: ResNet (18/34/50)、EfficientNet、MobileNet、DenseNet、VGG、自定义 CNN
- **迁移学习**: 支持 ImageNet 预训练权重微调
- **数据增强**: 随机裁剪、翻转、旋转、颜色抖动等
- **训练优化**: 学习率调度、早停机制、梯度裁剪
- **评估可视化**: 混淆矩阵、训练曲线、预测样本展示
- **TensorBoard**: 实时监控训练过程
- **命令行工具**: 训练、评估、推理一站式 CLI

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA (可选，用于 GPU 加速)

## 安装

```bash
# 克隆仓库
git clone https://github.com/lumostian3111/image-classification.git
cd image-classification

# 安装依赖
pip install -r requirements.txt
```

## 数据准备

将图片按类别放入 `data/` 目录的子文件夹中：

```
data/
├── train/
│   ├── cat/
│   │   ├── cat001.jpg
│   │   └── cat002.jpg
│   └── dog/
│       ├── dog001.jpg
│       └── dog002.jpg
├── val/          # 可选，未提供时自动从训练集划分 20%
│   ├── cat/
│   └── dog/
└── test/         # 可选，未提供时使用验证集
    ├── cat/
    └── dog/
```

## 使用方法

### 1. 训练模型

```bash
# 默认配置训练 (ResNet18)
python main.py train

# 指定模型和参数
python main.py train --model resnet50 --epochs 100 --lr 0.0001 --batch-size 64

# 从头训练 (不使用预训练权重)
python main.py train --pretrained false

# 恢复中断的训练
python main.py train --resume checkpoints/checkpoint_epoch_10.pth
```

### 2. 评估模型

```bash
python main.py evaluate --checkpoint checkpoints/best_model.pth
```

### 3. 预测图片

```bash
# 单张图片预测
python main.py predict cat.jpg --top-k 3

# 文件夹批量预测
python main.py predict-folder ./test_images/ --top-k 5
```

## 配置说明

编辑 `config.py` 修改默认配置：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `MODEL_NAME` | 模型架构 | `resnet18` |
| `IMAGE_SIZE` | 输入图像尺寸 | `224` |
| `BATCH_SIZE` | 批次大小 | `32` |
| `EPOCHS` | 训练轮数 | `50` |
| `LEARNING_RATE` | 学习率 | `0.001` |
| `OPTIMIZER` | 优化器 | `adam` |
| `PRETRAINED` | 是否预训练 | `True` |
| `EARLY_STOPPING` | 是否早停 | `True` |
| `PATIENCE` | 早停容忍轮数 | `10` |

### 可选模型

| 模型 | 参数量 | 适用场景 |
|------|--------|----------|
| `custom_cnn` | ~5M | 小数据集快速实验 |
| `resnet18` | ~11M | 通用分类任务 |
| `resnet34` | ~21M | 中等复杂度任务 |
| `resnet50` | ~25M | 高精度需求 |
| `efficientnet_b0` | ~5M | 移动端/轻量级 |
| `mobilenet_v2` | ~3M | 移动端/嵌入式 |
| `densenet121` | ~8M | 密集预测任务 |
| `vgg16` | ~138M | 大规模特征提取 |

## 项目结构

```
image-classification/
├── config.py              # 配置文件
├── dataset.py             # 数据加载与增强
├── train.py               # 训练模块
├── evaluate.py            # 评估模块
├── predict.py             # 推理模块
├── main.py                # CLI 主入口
├── utils.py               # 工具函数
├── models/
│   ├── __init__.py
│   └── classifier.py      # 模型定义
├── data/                  # 数据集
│   ├── train/
│   ├── val/
│   └── test/
├── checkpoints/           # 模型保存
├── logs/                  # TensorBoard 日志
├── outputs/               # 评估结果输出
└── requirements.txt       # 依赖清单
```

## 输出说明

训练/评估完成后，`outputs/` 目录中包含：

- `confusion_matrix.png` - 混淆矩阵图
- `training_history.png` - 训练损失/准确率曲线 (训练时生成)
- `sample_predictions.png` - 预测样本可视化
- `evaluation_metrics.json` - 评估指标 JSON
- `class_names.json` - 类别名称映射

## License

MIT
