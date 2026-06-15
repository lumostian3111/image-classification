"""
数据集准备脚本
- 下载 CIFAR-10 / Flowers102 并转换为 ImageFolder 格式
- 将自定义图片按比例分配到 train/val/test
"""

import os
import random
from pathlib import Path
import shutil

import torchvision
import torchvision.datasets as datasets
from PIL import Image


# 支持的图片格式
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


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


def split_to_splits(
    src_dir: str,
    ratios: tuple = (0.7, 0.15, 0.15),
    move: bool = False,
    seed: int = 42,
):
    """
    将源文件夹中的图片按比例分配到 data/train, data/val, data/test

    源文件夹结构:
        src_dir/
            ├── cat/       ← 文件夹名即类别名
            │   ├── 001.jpg
            │   └── 002.jpg
            └── other/
                ├── a.png
                └── b.png

    执行后:
        data/train/cat/     ← 70%
        data/val/cat/       ← 15%
        data/test/cat/      ← 15%
        data/train/other/   ← 70%
        data/val/other/     ← 15%
        data/test/other/    ← 15%

    参数:
        src_dir:   源文件夹路径（其下每个子文件夹=一个类别）
        ratios:    (train, val, test) 比例元组
        move:      True=移动文件, False=复制文件
        seed:      随机种子
    """
    src_path = Path(src_dir)
    if not src_path.is_dir():
        print(f"  [错误] 源文件夹不存在: {src_dir}")
        return

    # 确保比例和为 1
    assert abs(sum(ratios) - 1.0) < 0.001, f"比例之和应为 1.0，当前为 {sum(ratios)}"
    train_r, val_r, test_r = ratios

    from config import Config

    random.seed(seed)

    # 扫描所有子文件夹作为类别
    classes = [
        d for d in src_path.iterdir()
        if d.is_dir() and not d.name.startswith(".") and not d.name.startswith("_")
    ]

    if not classes:
        print(f"  [错误] {src_dir} 中没有找到子文件夹（每个子文件夹=一个类别）")
        print(f"  示例结构:")
        print(f"    {src_dir}/")
        print(f"      ├── class_a/")
        print(f"      │   ├── img1.jpg")
        print(f"      │   └── img2.png")
        print(f"      └── class_b/")
        print(f"          └── img3.jpg")
        return

    print(f"\n  找到 {len(classes)} 个类别: {[c.name for c in classes]}")
    print(f"  分割比例: 训练集 {train_r:.0%} / 验证集 {val_r:.0%} / 测试集 {test_r:.0%}")
    print(f"  操作方式: {'移动' if move else '复制'}\n")

    total_copied = 0

    for cls_dir in sorted(classes):
        cls_name = cls_dir.name

        # 收集该类别下所有图片
        images = set()
        for ext in IMAGE_EXTENSIONS:
            for p in cls_dir.iterdir():
                if p.is_file() and p.suffix.lower() == ext:
                    images.add(p)
        images = sorted(images)

        if not images:
            print(f"  [跳过] {cls_name}: 没有找到图片文件")
            continue

        # 随机打乱
        random.shuffle(images)

        # 按比例计算切分点
        n = len(images)
        n_train = round(n * train_r)
        n_val = round(n * val_r)
        # n_test 取剩余全部，避免因四舍五入丢失图片
        n_train = min(n_train, n - 2) if n >= 3 else n  # 至少留 2 张给 val/test
        n_val = min(n_val, n - n_train - 1) if n - n_train >= 2 else max(0, n - n_train - (1 if n - n_train > 1 else 0))

        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]

        # 分配到目标目录
        splits = [
            (Config.TRAIN_DIR, train_images, "train"),
            (Config.VAL_DIR, val_images, "val"),
            (Config.TEST_DIR, test_images, "test"),
        ]

        for target_root, img_list, split_name in splits:
            target_dir = Path(target_root) / cls_name
            target_dir.mkdir(parents=True, exist_ok=True)

            for img_path in img_list:
                dest = target_dir / img_path.name

                # 处理重名：已有则加后缀
                if dest.exists():
                    stem = img_path.stem
                    suffix = img_path.suffix
                    counter = 1
                    while dest.exists():
                        dest = target_dir / f"{stem}_{counter}{suffix}"
                        counter += 1

                if move:
                    shutil.move(str(img_path), str(dest))
                else:
                    shutil.copy2(str(img_path), str(dest))

        n_success = n_train + n_val + len(test_images)
        total_copied += n_success
        print(f"  [{cls_name}] 共 {n} 张 → train: {n_train} / val: {n_val} / test: {len(test_images)}")

    print(f"\n  [完成] 共处理 {len(classes)} 个类别, {total_copied} 张图片")
    print(f"  训练集: {Config.TRAIN_DIR}")
    print(f"  验证集: {Config.VAL_DIR}")
    print(f"  测试集: {Config.TEST_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="数据集准备工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ---- download ----
    dl_parser = subparsers.add_parser("download", help="下载内置数据集")
    dl_parser.add_argument("dataset", nargs="?", default="cifar10",
                           choices=["cifar10", "flowers102"],
                           help="数据集名称 (默认: cifar10)")

    # ---- split ----
    sp_parser = subparsers.add_parser("split", help="将图片按比例分配到 train/val/test")
    sp_parser.add_argument("--src", type=str, required=True,
                           help="源文件夹路径（其下每个子文件夹=一个类别）")
    sp_parser.add_argument("--ratio", type=float, nargs=3, default=[0.7, 0.15, 0.15],
                           metavar=("TRAIN", "VAL", "TEST"),
                           help="训练/验证/测试比例 (默认: 0.7 0.15 0.15)")
    sp_parser.add_argument("--move", action="store_true",
                           help="移动文件而非复制 (默认复制)")
    sp_parser.add_argument("--seed", type=int, default=42,
                           help="随机种子 (默认: 42)")

    args = parser.parse_args()

    from config import Config

    if args.command == "download":
        dataset_name = args.dataset
        if dataset_name == "flowers102":
            prepare_flowers102(Config.DATA_DIR)
        else:
            prepare_cifar10(Config.DATA_DIR)

    elif args.command == "split":
        split_to_splits(
            src_dir=args.src,
            ratios=tuple(args.ratio),
            move=args.move,
            seed=args.seed,
        )

    else:
        parser.print_help()
