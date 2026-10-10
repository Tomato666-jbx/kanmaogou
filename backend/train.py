# -*- coding: utf-8 -*-
"""
训练脚本（对应实验指导书 第一步~第五步）

用法：
  python train.py --dataset ../datasets --model resnet18 --epochs 8 --augment --save best_model.pth
  python train.py --experiments all            # 一键跑完六步横向对比

说明：
  - 数据集目录下的图片文件名形如 cat.1.jpg / dog.1.jpg（标签藏在文件名里）
  - 默认保存验证集最佳模型到 best_model.pth（第五步：训练与推理分离）
"""
import argparse
import os
import time
import torch
import torch.nn as nn
from torchvision import transforms

from common import get_loaders, train_and_eval, IMAGENET_MEAN, IMAGENET_STD
from models import AlexNet, AlexNetManual, ResNet18Manual, build_model

DEVICE = "cpu"  # 本实验全程 CPU

# 数据增强（第三步）：只作用于训练集
AUGMENT_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # 随机裁剪缩放
    transforms.RandomHorizontalFlip(),                    # 随机水平翻转
    transforms.ColorJitter(0.2, 0.2, 0.2),                # 随机亮度/对比度/饱和度
    transforms.RandomRotation(10),                        # 随机小角度旋转
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def init_weight(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)


# ---------------- 各步骤实验 ----------------

def step1_alexnet_scratch(data_dir, epochs, batch_size, lr, save_dir):
    """第一步：手动搭建 AlexNet，从零训练"""
    print("\n========== 第一步：AlexNet 从零训练 ==========")
    train_loader, val_loader = get_loaders(data_dir, batch_size=batch_size)
    net = AlexNet()
    net.apply(init_weight)
    acc = train_and_eval(net, train_loader, val_loader, epochs=epochs, lr=lr,
                         device=DEVICE, save_path=os.path.join(save_dir, "model_alexnet_scratch.pth"))
    print(f"【第一步】AlexNet 从零训练 最终准确率: {acc:.4f}")
    return acc


def step2_alexnet_pretrained(data_dir, epochs, batch_size, lr, save_dir):
    """第二步：加载 ImageNet 预训练参数，冻结卷积只训练分类头"""
    print("\n========== 第二步：AlexNet + ImageNet 预训练（冻结卷积） ==========")
    train_loader, val_loader = get_loaders(data_dir, batch_size=batch_size)
    official = torchvision_models().alexnet(weights=torchvision_models().AlexNet_Weights.DEFAULT)
    net = AlexNetManual(num_classes=1000)          # 1. 先建 1000 类的完整结构
    net.load_state_dict(official.state_dict())     # 2. 形状完全一致，strict=True 也能加载
    net.classifier[6] = nn.Linear(4096, 2)         # 3. 分类头换成 2 类（新头自动随机初始化）

    # 冻结：只训练 classifier
    for name, param in net.named_parameters():
        param.requires_grad = name.startswith("classifier")

    acc = train_and_eval(net, train_loader, val_loader, epochs=epochs, lr=lr,
                         device=DEVICE, save_path=os.path.join(save_dir, "model_alexnet_pretrained.pth"))
    print(f"【第二步】AlexNet 预训练 最终准确率: {acc:.4f}")
    return acc


def step3_augment_resnet18(data_dir, epochs, batch_size, lr, save_dir):
    """第三步：预训练 ResNet18 + 数据增强"""
    print("\n========== 第三步：ResNet18 预训练 + 数据增强 ==========")
    train_loader, val_loader = get_loaders(data_dir, batch_size=batch_size,
                                           train_transform=AUGMENT_TRANSFORM)
    net = ResNet18Manual(num_classes=2)
    official = torchvision_models().resnet18(weights=torchvision_models().ResNet18_Weights.DEFAULT)
    net.load_state_dict(official.state_dict(), strict=False)
    for name, param in net.named_parameters():
        param.requires_grad = 'fc' in name

    acc = train_and_eval(net, train_loader, val_loader, epochs=epochs, lr=lr,
                         device=DEVICE, save_path=os.path.join(save_dir, "model_resnet18_aug.pth"))
    print(f"【第三步】预训练+数据增强 最终准确率: {acc:.4f}")
    return acc


def step4_compare_models(data_dir, epochs, batch_size, lr, save_dir):
    """第四步：torchvision 成熟模型横向对比"""
    print("\n========== 第四步：多模型横向对比 ==========")
    train_loader, val_loader = get_loaders(data_dir, batch_size=batch_size)
    results = {}
    for name in ["resnet18", "mobilenet_v3_small", "shufflenet_v2_x1_0"]:
        print(f"\n----- 模型: {name} -----")
        net = build_model(name)
        t0 = time.time()
        acc = train_and_eval(net, train_loader, val_loader, epochs=epochs, lr=lr,
                             device=DEVICE, save_path=os.path.join(save_dir, f"model_{name}.pth"))
        cost = time.time() - t0
        results[name] = (acc, cost)
        print(f"模型 {name}: 准确率 {acc:.4f}, 训练耗时 {cost:.1f}s")

    print("\n===== 横向对比结果（按准确率排序） =====")
    for name, (acc, cost) in sorted(results.items(), key=lambda x: -x[1][0]):
        print(f"{name:22s} 准确率: {acc:.4f} | 耗时: {cost:.1f}s")
    return results


def torchvision_models():
    from torchvision import models as m
    return m


# ---------------- 主入口 ----------------

def main():
    parser = argparse.ArgumentParser(description="猫狗分类六步实验训练脚本")
    parser.add_argument("--dataset", default="../datasets",
                        help="数据集目录（文件名形如 cat.1.jpg / dog.1.jpg）")
    parser.add_argument("--model", default="resnet18",
                        choices=["resnet18", "mobilenet_v3_small", "shufflenet_v2_x1_0",
                                 "alexnet_scratch", "alexnet_pretrained"],
                        help="要训练的模型")
    parser.add_argument("--experiments", default=None,
                        help="'all' 一键跑完第一步到第四步的横向对比")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--augment", action="store_true",
                        help="是否启用数据增强（训练集随机翻转/裁剪/色彩）")
    parser.add_argument("--freeze", action="store_true",
                        help="冻结特征层只训练分类头（默认仅对 resnet18 等封装模型生效）")
    parser.add_argument("--save", default="best_model.pth",
                        help="最佳模型保存路径（第五步：训练时记录最佳验证准确率）")
    parser.add_argument("--save_dir", default=".",
                        help="各实验模型保存目录")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.experiments == "all":
        # 一键横向对比：固定同一份数据划分(seed=42) + 同样的评价指标，每次只改一个变量
        step1_alexnet_scratch(args.dataset, args.epochs, args.batch_size, 1e-4, args.save_dir)
        step2_alexnet_pretrained(args.dataset, args.epochs, args.batch_size, 1e-4, args.save_dir)
        step3_augment_resnet18(args.dataset, args.epochs, args.batch_size, 1e-3, args.save_dir)
        step4_compare_models(args.dataset, args.epochs, args.batch_size, 1e-3, args.save_dir)
        print("\n全部实验完成！")
        return

    # 单模型训练（默认入口：resnet18 预训练 + 最佳模型保存，供后端推理使用）
    if args.model in ("resnet18", "mobilenet_v3_small", "shufflenet_v2_x1_0"):
        train_transform = AUGMENT_TRANSFORM if args.augment else None
        train_loader, val_loader = get_loaders(args.dataset, batch_size=args.batch_size,
                                               train_transform=train_transform)
        net = build_model(args.model, freeze=args.freeze)
    elif args.model == "alexnet_scratch":
        train_loader, val_loader = get_loaders(args.dataset, batch_size=args.batch_size)
        net = AlexNet()
        net.apply(init_weight)
    elif args.model == "alexnet_pretrained":
        train_loader, val_loader = get_loaders(args.dataset, batch_size=args.batch_size)
        official = torchvision_models().alexnet(weights=torchvision_models().AlexNet_Weights.DEFAULT)
        net = AlexNetManual(num_classes=1000)
        net.load_state_dict(official.state_dict())
        net.classifier[6] = nn.Linear(4096, 2)
        for name, param in net.named_parameters():
            param.requires_grad = name.startswith("classifier")

    print(f"开始训练模型: {args.model}（epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, device={DEVICE}）")
    acc = train_and_eval(net, train_loader, val_loader, epochs=args.epochs, lr=args.lr,
                         device=DEVICE, save_path=args.save)
    print(f"训练完成，最终验证准确率: {acc:.4f}，最佳模型已保存到 {args.save}")


if __name__ == '__main__':
    main()
