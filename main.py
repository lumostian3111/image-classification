"""
图像分类系统 - 主入口
命令行界面，支持训练、评估、推理三种模式

用法:
    python main.py train                     # 训练模型
    python main.py train --resume checkpoint.pth  # 恢复训练
    python main.py evaluate                  # 评估模型
    python main.py evaluate --checkpoint checkpoints/best_model.pth
    python main.py predict image.jpg         # 单张图片预测
    python main.py predict-folder data/test/ # 文件夹批量预测
"""

import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="图像分类系统 - 基于 PyTorch 的图像分类工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py train                          # 训练模型
  python main.py train --epochs 100 --lr 0.0001 # 自定义参数训练
  python main.py evaluate                       # 评估最佳模型
  python main.py predict cat.jpg                # 预测单张图片
  python main.py predict-folder ./test_images/  # 批量预测
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ===== 训练 =====
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument(
        "--model", type=str, default=None,
        help="模型名称 (resnet18, resnet50, efficientnet_b0, mobilenet_v2, ...)"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=None,
        help="训练轮数"
    )
    train_parser.add_argument(
        "--lr", type=float, default=None,
        help="学习率"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=None,
        help="批次大小"
    )
    train_parser.add_argument(
        "--pretrained", type=str, default=None,
        choices=["true", "false"],
        help="是否使用预训练权重"
    )
    train_parser.add_argument(
        "--no-early-stopping", action="store_true",
        help="禁用早停"
    )
    train_parser.add_argument(
        "--resume", type=str, default=None,
        help="从检查点恢复训练"
    )
    train_parser.add_argument(
        "--data-dir", type=str, default=None,
        help="数据集根目录 (包含 train/val/test 子文件夹)"
    )

    # ===== 评估 =====
    eval_parser = subparsers.add_parser("evaluate", help="评估模型")
    eval_parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pth",
        help="模型检查点路径"
    )
    eval_parser.add_argument(
        "--data-dir", type=str, default=None,
        help="数据集根目录"
    )
    eval_parser.add_argument(
        "--output", type=str, default=None,
        help="评估结果输出目录"
    )

    # ===== 单张预测 =====
    pred_parser = subparsers.add_parser("predict", help="单张图片预测")
    pred_parser.add_argument("image", type=str, help="图片路径")
    pred_parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pth",
        help="模型检查点路径"
    )
    pred_parser.add_argument(
        "--top-k", type=int, default=3,
        help="返回 top-k 个预测结果"
    )

    # ===== 文件夹预测 =====
    folder_parser = subparsers.add_parser("predict-folder", help="文件夹批量预测")
    folder_parser.add_argument("folder", type=str, help="图片文件夹路径")
    folder_parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pth",
        help="模型检查点路径"
    )
    folder_parser.add_argument(
        "--top-k", type=int, default=3,
        help="返回 top-k 个预测结果"
    )
    folder_parser.add_argument(
        "--output", type=str, default=None,
        help="结果 JSON 输出路径"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # ---- 根据命令行参数更新配置 ----
    from config import Config

    if args.command == "train":
        if args.model:
            Config.MODEL_NAME = args.model
        if args.epochs:
            Config.EPOCHS = args.epochs
        if args.lr:
            Config.LEARNING_RATE = args.lr
        if args.batch_size:
            Config.BATCH_SIZE = args.batch_size
        if args.pretrained is not None:
            Config.PRETRAINED = args.pretrained.lower() == "true"
        if args.no_early_stopping:
            Config.EARLY_STOPPING = False
        if args.data_dir:
            Config.DATA_DIR = args.data_dir
            Config.TRAIN_DIR = str(Path(args.data_dir) / "train")
            Config.VAL_DIR = str(Path(args.data_dir) / "val")
            Config.TEST_DIR = str(Path(args.data_dir) / "test")

        from train import train as run_train
        run_train(resume_from=args.resume)

    elif args.command == "evaluate":
        if args.data_dir:
            Config.DATA_DIR = args.data_dir
            Config.TRAIN_DIR = str(Path(args.data_dir) / "train")
            Config.VAL_DIR = str(Path(args.data_dir) / "val")
            Config.TEST_DIR = str(Path(args.data_dir) / "test")

        from evaluate import evaluate as run_evaluate
        run_evaluate(checkpoint_path=args.checkpoint, output_dir=args.output)

    elif args.command == "predict":
        from predict import predict_single
        predict_single(
            image_path=args.image,
            checkpoint_path=args.checkpoint,
            top_k=args.top_k,
        )

    elif args.command == "predict-folder":
        from predict import predict_folder
        predict_folder(
            folder_path=args.folder,
            checkpoint_path=args.checkpoint,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
