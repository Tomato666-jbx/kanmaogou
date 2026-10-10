# 🐱🐶 猫狗图像分类系统

基于 OpenCV + PyTorch 迁移学习的猫狗图像分类项目，带完整的前后端：

- **前端**：原生 HTML/CSS/JS，支持拖拽上传、摄像头拍照（getUserMedia）、实时结果与概率柱状图展示
- **后端**：FastAPI + OpenCV（图像解码/缩放/BGR2RGB 预处理）+ PyTorch（ResNet18 推理）
- **训练**：完整实现实验指导书六步流程（手动 AlexNet → 迁移学习 → 数据增强 → 多模型对比 → 最佳模型保存）

```
项目结构
├── backend/
│   ├── app.py           # FastAPI Web 服务（预测 API + 托管前端）
│   ├── common.py        # 数据加载 / 训练评估公共函数（指导书 0.2/0.3）
│   ├── models.py        # AlexNet / AlexNetManual / ResNet18Manual / build_model
│   ├── train.py         # 六步实验训练脚本，自动保存 best_model.pth
│   ├── inference.py     # OpenCV 预处理 + 模型推理
│   └── best_model.pth   # 训练产出的最佳模型（训练后生成）
├── frontend/
│   ├── index.html       # 主页面
│   ├── css/style.css
│   └── js/app.js
├── datasets/            # 数据集放这里（文件名形如 cat.1.jpg / dog.1.jpg）
└── requirements.txt
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 把数据集放入 datasets/ 目录（约 100 张图，文件名形如 cat.1.jpg / dog.1.jpg）

# 3. 训练模型（默认 ResNet18 预训练 + 数据增强，自动保存验证集最佳模型）
cd backend
python train.py --dataset ../datasets --model resnet18 --epochs 5 --augment --save best_model.pth

# 4. 启动服务
python app.py
# 浏览器访问 http://127.0.0.1:8000
# 手机（同一局域网）访问 http://<电脑IP>:8000
```

## 训练脚本用法（对应指导书六步）

| 实验 | 命令 |
|---|---|
| 第一步：AlexNet 从零训练 | `python train.py --model alexnet_scratch --epochs 30 --lr 1e-4` |
| 第二步：AlexNet 预训练（冻结卷积） | `python train.py --model alexnet_pretrained --epochs 30 --lr 1e-4` |
| 第三步：ResNet18 预训练 + 数据增强 | `python train.py --model resnet18 --augment` |
| 第四步：多模型横向对比 | `python train.py --experiments all` |
| 一键跑完前四步对比 | `python train.py --experiments all` |

> 全程固定同一份数据划分（seed=42）+ 同样的评价指标，每次只改一个变量，保证横向可比。

## API 接口

| 接口 | 方法 | 说明 |
|---|---|---|
| `/api/health` | GET | 服务与模型状态 |
| `/api/predict` | POST | 图片文件上传预测（multipart，字段名 `file`） |
| `/api/predict_base64` | POST | base64 图片预测（JSON `{"image": "..."}`） |

响应示例：

```json
{
  "label": "猫",
  "class_index": 0,
  "confidence": 0.9821,
  "probabilities": [0.9821, 0.0179],
  "classes": ["猫", "狗"]
}
```

## 摄像头说明

前端通过浏览器 `getUserMedia` 调用摄像头，拍照后作为图片发送到后端识别。
首次使用需在浏览器中允许摄像头权限；若无法调用（如无摄像头环境），可直接使用上传图片功能。

## 常见问题

- **提示"模型尚未训练"**：运行 `python backend/train.py` 生成 `best_model.pth` 后重启服务。
- **加载权重报错**：`best_model.pth` 需由本项目的 `train.py` 产出，或与所选模型架构一致（默认 resnet18）。
- **手机无法访问**：确保电脑与手机在同一 Wi-Fi，关闭防火墙或放行 8000 端口。
