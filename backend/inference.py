# -*- coding: utf-8 -*-
"""
推理模块（对应实验指导书 第五步）

使用 OpenCV 完成图像读取/尺寸缩放/BGR2RGB 预处理，再用 PyTorch 模型预测：
  解码 -> cv2.resize(INTER_AREA) -> BGR2RGB -> ToTensor -> Normalize -> softmax

支持从 best_model.pth 加载训练好的模型，默认架构为 resnet18。
"""
import io
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms

from common import IMAGENET_MEAN, IMAGENET_STD, CLASSES

# 推理时与训练时保持一致的数据预处理
INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# 支持的模型架构（与训练脚本对应）
_ARCH_MAP = {
    "resnet18": lambda num_classes: _resnet18(num_classes),
    "mobilenet_v3_small": lambda num_classes: _mobilenet(num_classes),
    "shufflenet_v2_x1_0": lambda num_classes: _shufflenet(num_classes),
}


def _resnet18(num_classes):
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def _mobilenet(num_classes):
    net = models.mobilenet_v3_small(weights=None)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, num_classes)
    return net


def _shufflenet(num_classes):
    net = models.shufflenet_v2_x1_0(weights=None)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def load_model(model_name="resnet18", weight_path="best_model.pth", num_classes=2, device="cpu"):
    """构建同结构模型并加载权重（结构一致即可，无需下载权重）"""
    if model_name not in _ARCH_MAP:
        raise ValueError(f"不支持的模型架构: {model_name}，可选: {list(_ARCH_MAP)}")
    net = _ARCH_MAP[model_name](num_classes)
    net.load_state_dict(torch.load(weight_path, map_location=device))
    net.to(device)
    net.eval()
    return net


def predict_bgr(bgr_image: np.ndarray, net, device="cpu"):
    """输入 OpenCV BGR 图像（numpy 数组），返回 (类别名, 置信度, 各类别概率列表)"""
    if bgr_image is None:
        raise ValueError("图像解码失败")
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)   # OpenCV BGR -> RGB
    img = transforms.functional.to_pil_image(rgb)
    x = INFER_TRANSFORM(img).unsqueeze(0)              # [1, 3, 224, 224]
    with torch.no_grad():
        probs = torch.softmax(net(x.to(device)), dim=1)[0].cpu()
    conf, pred = probs.max(0)
    probabilities = [round(float(p), 4) for p in probs]
    return {
        "label": CLASSES[int(pred)],
        "class_index": int(pred),
        "confidence": round(float(conf), 4),
        "probabilities": probabilities,
        "classes": CLASSES,
    }


def predict_bytes(image_bytes: bytes, net, device="cpu"):
    """从二进制图片数据直接预测（后端 API 使用）"""
    arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # 用 OpenCV 解码
    return predict_bgr(bgr, net, device=device)
