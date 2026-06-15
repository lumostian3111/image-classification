"""
图像分类系统 - 推理模块
支持单张图片预测、批量预测和文件夹预测
"""

import os
import json
from typing import Union, List, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from config import Config
from models import create_model
from dataset import InferenceTransform
from utils import load_checkpoint, get_timestamp


class ImageClassifier:
    """图像分类推理器"""

    def __init__(
        self,
        checkpoint_path: str = None,
        class_names_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        初始化推理器

        参数:
            checkpoint_path: 训练好的模型路径
            class_names_path: 类别名称 JSON 文件路径
            device: 推理设备
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        self.device = torch.device(device or Config.DEVICE)

        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # 获取类别名称
        if class_names_path and os.path.exists(class_names_path):
            with open(class_names_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.class_names = data["class_names"]
        elif "class_names" in checkpoint:
            self.class_names = checkpoint["class_names"]
        else:
            raise ValueError("无法获取类别名称，请提供 class_names_path 参数")

        self.num_classes = len(self.class_names)
        Config.NUM_CLASSES = self.num_classes

        # 从检查点恢复模型类型
        if "model_name" in checkpoint:
            Config.MODEL_NAME = checkpoint["model_name"]

        # 加载模型
        self.model = create_model(
            num_classes=self.num_classes,
            model_name=Config.MODEL_NAME,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        # 预处理
        self.transform = InferenceTransform()

        print(f"  模型已加载: {checkpoint.get('model_name', 'unknown')}")
        print(f"  类别数: {self.num_classes}")
        print(f"  设备: {self.device}")
        print(f"  类别: {self.class_names}")

    @torch.no_grad()
    def predict(
        self,
        image_path: str,
        top_k: int = 3,
    ) -> dict:
        """
        预测单张图片

        参数:
            image_path: 图片路径
            top_k: 返回 top-k 个预测结果

        返回:
            包含预测结果的字典
        """
        # 加载与预处理
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        input_tensor = self.transform(image).to(self.device)

        # 推理
        outputs = self.model(input_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        # Top-k 结果
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_probs = probs[top_indices]
        top_names = [self.class_names[i] for i in top_indices]

        result = {
            "image_path": image_path,
            "image_size": original_size,
            "predictions": [],
        }

        for i, (name, prob) in enumerate(zip(top_names, top_probs)):
            result["predictions"].append({
                "rank": i + 1,
                "class": name,
                "class_id": int(top_indices[i]),
                "confidence": round(float(prob), 6),
                "confidence_pct": f"{prob * 100:.2f}%",
            })

        return result

    def predict_batch(
        self,
        image_paths: List[str],
        top_k: int = 1,
    ) -> List[dict]:
        """批量预测多张图片"""
        results = []
        for path in image_paths:
            try:
                result = self.predict(path, top_k=top_k)
                results.append(result)
            except Exception as e:
                results.append({
                    "image_path": path,
                    "error": str(e),
                })

        return results

    def predict_folder(
        self,
        folder_path: str,
        top_k: int = 3,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
        output_json: Optional[str] = None,
    ) -> List[dict]:
        """预测整个文件夹的图片"""
        folder = Path(folder_path)
        image_paths = []

        for ext in extensions:
            image_paths.extend(folder.glob(f"*{ext}"))
            image_paths.extend(folder.glob(f"*{ext.upper()}"))

        image_paths = sorted([str(p) for p in image_paths])

        if not image_paths:
            print(f"  警告: 在 {folder_path} 中未找到图片文件")
            return []

        print(f"  找到 {len(image_paths)} 张图片")

        results = self.predict_batch(image_paths, top_k=top_k)

        # 保存结果
        if output_json is None:
            output_json = os.path.join(Config.OUTPUT_DIR, f"predictions_{get_timestamp()}.json")

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"  预测结果已保存到: {output_json}")

        return results

    def print_result(self, result: dict):
        """格式化打印预测结果"""
        print("\n" + "=" * 50)
        print(f"  图片: {result['image_path']}")
        print(f"  尺寸: {result['image_size']}")
        print("-" * 50)
        print(f"  {'排名':<6s} {'类别':<20s} {'置信度':>12s}")
        print("-" * 50)

        for pred in result["predictions"]:
            print(
                f"  {pred['rank']:<6d} "
                f"{pred['class']:<20s} "
                f"{pred['confidence_pct']:>12s}"
            )
        print("=" * 50)


def predict_single(
    image_path: str,
    checkpoint_path: str = None,
    top_k: int = 3,
):
    """便捷函数: 预测单张图片并打印结果"""
    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    classifier = ImageClassifier(checkpoint_path=checkpoint_path)
    result = classifier.predict(image_path, top_k=top_k)
    classifier.print_result(result)
    return result


def predict_folder(
    folder_path: str,
    checkpoint_path: str = None,
    top_k: int = 3,
):
    """便捷函数: 预测整个文件夹"""
    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    classifier = ImageClassifier(checkpoint_path=checkpoint_path)
    return classifier.predict_folder(folder_path, top_k=top_k)
