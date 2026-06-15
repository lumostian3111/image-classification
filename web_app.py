"""
图像分类系统 - Web 界面
Flask 驱动的浏览器交互界面
"""

import os
import shutil

# 修复 Windows 上 OpenMP DLL 冲突问题
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import threading
import time
import webbrowser
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory, url_for,
)

from config import Config
from models import create_model, count_parameters
from predict import ImageClassifier
from dataset import get_val_transforms
from prepare_data import split_to_splits, IMAGE_EXTENSIONS

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(Config.ROOT_DIR, "static", "uploads")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB
# 临时目录：用于接收用户上传的图片，分配完成后自动清理
TEMP_IMPORT_DIR = os.path.join(Config.ROOT_DIR, "temp_imports")

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ==================== 全局状态 ====================

def _init_training_status():
    """初始化训练状态，检测已有模型"""
    best_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_path):
        try:
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            return {
                "running": False,
                "message": f"已有训练模型 (Epoch {ckpt.get('epoch', '?')}, "
                           f"Best Acc: {ckpt.get('best_acc', 0):.2%})",
                "progress": 0,
                "epoch": ckpt.get("epoch", 0),
                "total_epochs": ckpt.get("epoch", 0),
                "train_loss": 0,
                "train_acc": 0,
                "val_loss": 0,
                "val_acc": ckpt.get("best_acc", 0),
                "best_acc": ckpt.get("best_acc", 0),
                "history": [],
            }
        except Exception:
            pass
    return {
        "running": False,
        "message": "尚未训练",
        "progress": 0,
        "epoch": 0,
        "total_epochs": 0,
        "train_loss": 0,
        "train_acc": 0,
        "val_loss": 0,
        "val_acc": 0,
        "best_acc": 0,
        "history": [],
    }

training_status = _init_training_status()


# ==================== 页面路由 ====================

@app.route("/")
def index():
    """主页"""
    # 检查模型是否可用
    model_ready = os.path.exists(os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"))
    class_names = []
    if model_ready:
        try:
            ckpt = torch.load(os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"), map_location="cpu", weights_only=False)
            class_names = ckpt.get("class_names", [])
        except Exception:
            pass

    # 扫描 data/train 获取数据集中的实际类别
    data_classes = []
    train_dir = Config.TRAIN_DIR
    if os.path.isdir(train_dir):
        for d in sorted(os.listdir(train_dir)):
            full = os.path.join(train_dir, d)
            if os.path.isdir(full) and not d.startswith("."):
                count = len([f for f in os.listdir(full)
                           if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS])
                data_classes.append({"name": d, "count": count})

    return render_template("index.html",
                         model_ready=model_ready,
                         class_names=class_names,
                         data_classes=data_classes)


# ==================== 预测 API ====================

@app.route("/predict", methods=["POST"])
def predict():
    """上传图片并预测"""
    if "file" not in request.files:
        return jsonify({"error": "请选择图片文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "请选择图片文件"}), 400

    # 保存上传文件
    import uuid
    ext = Path(file.filename).suffix or ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # 检查模型
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        return jsonify({"error": "模型尚未训练，请先训练模型"}), 400

    try:
        classifier = ImageClassifier(checkpoint_path=checkpoint_path)
        result = classifier.predict(filepath, top_k=5)
        return jsonify({
            "success": True,
            "filename": filename,
            "result": result,
        })
    except Exception as e:
        return jsonify({"error": f"预测失败: {str(e)}"}), 500


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# ==================== 训练 API ====================

@app.route("/start-training", methods=["POST"])
def start_training():
    """启动后台训练"""
    global training_status

    if training_status["running"]:
        return jsonify({"error": "训练已在运行中"}), 400

    data = request.get_json() or {}
    Config.MODEL_NAME = data.get("model", "custom_cnn")
    Config.EPOCHS = int(data.get("epochs", 5))
    Config.LEARNING_RATE = float(data.get("lr", 0.001))
    Config.BATCH_SIZE = int(data.get("batch_size", 32))

    threading.Thread(target=_run_training, daemon=True).start()

    return jsonify({"success": True, "message": "训练已启动"})


@app.route("/training-status")
def get_training_status():
    """获取训练进度"""
    return jsonify(training_status)


def _run_training():
    """后台训练线程"""
    global training_status

    try:
        training_status["running"] = True
        training_status["message"] = "正在加载数据..."
        training_status["progress"] = 0
        training_status["history"] = []

        from dataset import create_dataloaders
        import torch.nn as nn
        import torch.optim as optim

        device = torch.device(Config.DEVICE)
        Config.create_dirs()

        # 加载数据
        train_loader, val_loader, test_loader, class_names = create_dataloaders()
        num_classes = len(class_names)

        # 创建模型
        training_status["message"] = "正在创建模型..."
        model = create_model(num_classes=num_classes)
        model = model.to(device)

        # 训练配置
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

        best_acc = 0.0
        training_status["total_epochs"] = Config.EPOCHS

        for epoch in range(Config.EPOCHS):
            training_status["epoch"] = epoch + 1
            training_status["message"] = f"训练中 Epoch {epoch+1}/{Config.EPOCHS}..."

            # 训练
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                train_loss += loss.item() * images.size(0)
                train_correct += (preds == labels).sum().item()
                train_total += images.size(0)

                # 更新进度
                batch_progress = (batch_idx + 1) / len(train_loader) * 100
                training_status["progress"] = round(batch_progress, 1)
                training_status["train_loss"] = round(train_loss / train_total, 4)
                training_status["train_acc"] = round(train_correct / train_total, 4)

            scheduler.step()

            # 验证
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    val_loss += loss.item() * images.size(0)
                    val_correct += (preds == labels).sum().item()
                    val_total += images.size(0)

            val_acc = val_correct / val_total
            val_l = val_loss / val_total
            train_acc = train_correct / train_total
            train_l = train_loss / train_total

            training_status["val_loss"] = round(val_l, 4)
            training_status["val_acc"] = round(val_acc, 4)
            training_status["train_loss"] = round(train_l, 4)
            training_status["train_acc"] = round(train_acc, 4)

            if val_acc > best_acc:
                best_acc = val_acc
                training_status["best_acc"] = round(best_acc, 4)
                # 保存最佳模型
                from utils import save_checkpoint
                save_checkpoint(model, optimizer, epoch, best_acc, class_names, is_best=True)

            training_status["history"].append({
                "epoch": epoch + 1,
                "train_loss": round(train_l, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_l, 4),
                "val_acc": round(val_acc, 4),
            })

        training_status["message"] = f"训练完成! 最佳准确率: {best_acc:.4f}"
        training_status["running"] = False

        # 保存最终模型
        from utils import save_checkpoint
        save_checkpoint(model, optimizer, Config.EPOCHS - 1, best_acc, class_names, filename="final_model.pth")

    except Exception as e:
        training_status["running"] = False
        training_status["message"] = f"训练出错: {str(e)}"


# ==================== 评估 API ====================

@app.route("/evaluate", methods=["POST"])
def run_evaluation():
    """运行评估"""
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        return jsonify({"error": "模型尚未训练"}), 400

    try:
        from evaluate import evaluate as eval_func
        metrics = eval_func(checkpoint_path=checkpoint_path)
        return jsonify({"success": True, "metrics": metrics})
    except Exception as e:
        return jsonify({"error": f"评估失败: {str(e)}"}), 500


# ==================== 数据导入 API ====================

@app.route("/list-classes")
def list_classes():
    """列出当前数据集中已有的类别"""
    train_dir = Config.TRAIN_DIR
    classes = []
    if os.path.isdir(train_dir):
        for d in sorted(os.listdir(train_dir)):
            full = os.path.join(train_dir, d)
            if os.path.isdir(full) and not d.startswith("."):
                # 统计各 split 中的图片数
                train_count = len([
                    f for f in os.listdir(full)
                    if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
                ])
                val_count = 0
                val_dir = os.path.join(Config.VAL_DIR, d)
                if os.path.isdir(val_dir):
                    val_count = len([
                        f for f in os.listdir(val_dir)
                        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
                    ])
                test_count = 0
                test_dir = os.path.join(Config.TEST_DIR, d)
                if os.path.isdir(test_dir):
                    test_count = len([
                        f for f in os.listdir(test_dir)
                        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
                    ])
                classes.append({
                    "name": d,
                    "train": train_count,
                    "val": val_count,
                    "test": test_count,
                    "total": train_count + val_count + test_count,
                })
    return jsonify({"classes": classes, "total": len(classes)})


@app.route("/import-class", methods=["POST"])
def import_class():
    """接收用户上传的图片 + 类别名，调用 split_to_splits 分配"""
    class_name = (request.form.get("class_name") or "").strip()
    if not class_name:
        return jsonify({"error": "请输入类别名称"}), 400

    # 安全校验：类别名只能包含字母、数字、下划线、中文
    if not class_name.replace("_", "").replace("-", "").isalnum():
        return jsonify({"error": "类别名只能包含字母、数字、下划线或中文"}), 400

    files = request.files.getlist("images")
    if not files or len(files) == 0:
        return jsonify({"error": "请选择至少一张图片"}), 400

    # 保存到临时目录
    import uuid
    batch_id = uuid.uuid4().hex[:8]
    temp_class_dir = os.path.join(TEMP_IMPORT_DIR, batch_id, class_name)
    os.makedirs(temp_class_dir, exist_ok=True)

    saved_count = 0
    errors = []
    for f in files:
        if f.filename == "":
            continue
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            errors.append(f"跳过非图片文件: {f.filename}")
            continue
        # 保留原文件名，加随机前缀防重名
        safe_name = f"{batch_id}_{saved_count:04d}{ext}"
        f.save(os.path.join(temp_class_dir, safe_name))
        saved_count += 1

    if saved_count == 0:
        shutil.rmtree(os.path.join(TEMP_IMPORT_DIR, batch_id), ignore_errors=True)
        return jsonify({"error": "没有有效的图片文件，" + "; ".join(errors)}), 400

    # 调用 split_to_splits 自动分配到 train/val/test
    try:
        src_dir = os.path.join(TEMP_IMPORT_DIR, batch_id)
        split_to_splits(src_dir=src_dir, move=True)
    except Exception as e:
        return jsonify({"error": f"分配失败: {str(e)}"}), 500

    # 清理临时目录（图片已 move 走，只删空文件夹）
    shutil.rmtree(os.path.join(TEMP_IMPORT_DIR, batch_id), ignore_errors=True)

    # 统计分配后的数量
    train_count = len(os.listdir(os.path.join(Config.TRAIN_DIR, class_name)))
    val_count = len(os.listdir(os.path.join(Config.VAL_DIR, class_name)))
    test_count = len(os.listdir(os.path.join(Config.TEST_DIR, class_name)))

    return jsonify({
        "success": True,
        "message": f"类别「{class_name}」已导入！共 {saved_count} 张图片",
        "class_name": class_name,
        "total": saved_count,
        "distribution": {"train": train_count, "val": val_count, "test": test_count},
        "warnings": errors if errors else None,
    })


# ==================== 启动 ====================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Image Classification System - Web UI")
    print("=" * 50)
    print()
    print("  Starting server...")
    print()
    print("  Press Ctrl+C to stop")
    print("=" * 50 + "\n")

    # 自动打开浏览器
    url = "http://localhost:5000"
    try:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        print(f"  >> 正在打开浏览器: {url}\n")
    except Exception:
        print(f"  >> 请手动打开浏览器访问: {url}\n")

    try:
        app.run(host="0.0.0.0", port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n\n  系统已关闭\n")
    except OSError as e:
        if "Address already in use" in str(e) or "10048" in str(e):
            print(f"\n  [X] 端口 5000 已被占用，请先关闭其他程序")
            print("     或运行: netstat -ano | findstr :5000")
        else:
            print(f"\n  [X] 启动失败: {e}")
        print("\n  按任意键退出...")
        input()
