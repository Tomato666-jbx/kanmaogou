# -*- coding: utf-8 -*-
"""
猫狗分类 Web 服务（FastAPI + OpenCV + PyTorch）
- 提供 /api/predict 预测接口（图片上传 / base64 两种方式）
- 提供 /api/health 健康检查（含模型加载状态）
- 托管前端静态页面（frontend 目录）

启动：
  python app.py
  浏览器访问 http://127.0.0.1:8000
  手机（同一局域网）访问 http://<电脑IP>:8000
"""
import os
import base64
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from inference import load_model, predict_bytes

# ---------------- 路径配置 ----------------
BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"
MODEL_PATH = BACKEND_DIR / "best_model.pth"
DEFAULT_MODEL_NAME = "resnet18"  # 与训练脚本默认一致

app = FastAPI(title="猫狗分类 Web 服务", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 托管前端
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ---------------- 模型加载（启动时只做一次） ----------------
class ModelHolder:
    net = None
    model_name = DEFAULT_MODEL_NAME
    trained = False


def load_saved_model():
    if MODEL_PATH.exists():
        try:
            ModelHolder.net = load_model(DEFAULT_MODEL_NAME, str(MODEL_PATH))
            ModelHolder.trained = True
            return True
        except Exception as e:  # 权重与架构不匹配等情况
            print(f"[警告] 模型加载失败: {e}")
            ModelHolder.net = None
            ModelHolder.trained = False
    return False


load_saved_model()


def require_model():
    if not ModelHolder.trained or ModelHolder.net is None:
        raise HTTPException(
            status_code=503,
            detail="模型尚未就绪：请先训练模型。"
                   "1) 将数据集放入 datasets 目录（文件名形如 cat.1.jpg / dog.1.jpg）；"
                   "2) 运行 python backend/train.py 生成 best_model.pth；"
                   "3) 重启本服务。")
    return ModelHolder.net


# ---------------- 请求/响应模型 ----------------
class Base64Request(BaseModel):
    image: str  # base64 编码的图片数据（可带 data:image/...;base64, 前缀）


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_trained": ModelHolder.trained,
        "model_name": ModelHolder.model_name if ModelHolder.trained else None,
        "model_path": str(MODEL_PATH) if MODEL_PATH.exists() else None,
        "classes": ["猫", "狗"],
        "api": "猫狗分类 Web 服务 v1.0.0",
    }


@app.post("/api/predict")
async def predict_upload(file: UploadFile = File(...)):
    """方式一：multipart/form-data 文件上传"""
    net = require_model()
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="上传的文件为空")
    try:
        result = predict_bytes(data, net)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"图像解析失败: {e}")
    result["source"] = file.filename
    return JSONResponse(result)


@app.post("/api/predict_base64")
async def predict_base64(req: Base64Request):
    """方式二：JSON body 传 base64 图片"""
    net = require_model()
    b64 = req.image.split(",", 1)[-1] if "," in req.image else req.image
    try:
        data = base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=400, detail="base64 解码失败")
    try:
        result = predict_bytes(data, net)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"图像解析失败: {e}")
    result["source"] = "base64"
    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn
    print("猫狗分类服务启动中...")
    if ModelHolder.trained:
        print(f"模型已加载: {MODEL_PATH}")
    else:
        print("[提示] 未找到 best_model.pth，请先训练模型（python backend/train.py）")
    print("本机访问: http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
