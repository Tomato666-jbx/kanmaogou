# -*- coding: utf-8 -*-
"""离线评估：用训练好的 best_model.pth 对全部数据集图片做分类统计"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from common import FileNameDataset, IMAGENET_MEAN, IMAGENET_STD
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix
from inference import load_model

DATA = os.path.join(os.path.dirname(__file__), "..", "datasets")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

ds = FileNameDataset(image_dir=DATA, transform=transform)
net = load_model("resnet18", os.path.join(os.path.dirname(__file__), "best_model.pth"))

import torch
preds, labels = [], []
for img, label in ds:
    with torch.no_grad():
        p = net(img.unsqueeze(0)).argmax(1).item()
    preds.append(p)
    labels.append(label)

acc = accuracy_score(labels, preds)
cm = confusion_matrix(labels, preds)
print(f"全部 {len(labels)} 张图片评估:")
print(f"准确率: {acc:.4f} ({acc*len(labels):.0f}/{len(labels)})")
print(f"混淆矩阵 (行=真实, 列=预测):")
print(f"          预测猫   预测狗")
print(f"真实猫    {cm[0][0]:5d}   {cm[0][1]:5d}")
print(f"真实狗    {cm[1][0]:5d}   {cm[1][1]:5d}")
