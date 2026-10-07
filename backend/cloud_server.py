#!/usr/bin/env python3
"""
AvocadoCheck API for the cloud (Render free plan, 512 MB of RAM).

Same endpoints as server.py, but without PyTorch:
  - the CNN runs with onnxruntime (models/cnn_final.onnx, made by export_onnx.py);
  - the class activation map is computed from the last feature map and the
    weights of the stage head (models/cnn_head.npz). For a ResNet this is
    exactly Grad-CAM, because the logit is linear in the pooled features;
  - the fruit detector is the colour threshold detector of the pipeline
    (YOLO-World needs PyTorch and does not fit in the free plan).

Run locally
    uvicorn cloud_server:api --port 8000
Render
    startCommand: uvicorn cloud_server:api --host 0.0.0.0 --port $PORT

Environment variables
    MODELS_DIR       default ./models
    ALLOWED_ORIGINS  comma separated list for CORS, default "*"
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import app as core  # noqa: E402  detection, crop and feature helpers (no Streamlit import at load time)
import pipeline as P  # noqa: E402

VERSION = "1.0.0"
MAX_SIDE = 1600
MAX_FRUITS = 12
MODELS_DIR = Path(os.environ.get("MODELS_DIR", HERE / "models"))
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

api = FastAPI(title="AvocadoCheck API", version=VERSION)
origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
api.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class OnnxCNN:
    def __init__(self, models_dir: Path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        # BASIC keeps the same numbers and halves the memory of the session (fits the 512 MB free plan)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        so.enable_cpu_mem_arena = False
        self.sess = ort.InferenceSession(str(models_dir / "cnn_final.onnx"), so, providers=["CPUExecutionProvider"])
        head = np.load(models_dir / "cnn_head.npz")
        self.W = head["W"]
        self.meta = json.loads(str(head["meta"]))
        self.inputs = [i.name for i in self.sess.get_inputs()]

    def preprocess(self, crop: Image.Image) -> np.ndarray:
        s = self.meta["img_size"]
        arr = np.asarray(crop.convert("RGB").resize((s, s), Image.BILINEAR), dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return arr.transpose(2, 0, 1)[None].astype(np.float32)

    def run(self, crop: Image.Image, condition: str):
        feeds = {"image": self.preprocess(crop)}
        if "cond" in self.inputs:
            conds = self.meta["conditions"] if self.meta.get("use_condition") else []
            vec = np.zeros((1, len(conds)), dtype=np.float32)
            if condition in conds:
                vec[0, conds.index(condition)] = 1.0
            feeds["cond"] = vec
        logits, reg, fmap = self.sess.run(None, feeds)
        z = logits[0] - logits[0].max()
        probs = np.exp(z) / np.exp(z).sum()
        days = max(0.0, float(np.ravel(reg)[0]) * float(self.meta["y_scale"]))
        return probs, days, fmap[0]

    def cam(self, fmap: np.ndarray, cls: int) -> np.ndarray:
        c = fmap.shape[0]
        cam = np.maximum(np.tensordot(self.W[cls, :c], fmap, axes=(0, 0)), 0)
        size = self.meta["img_size"]
        img = Image.fromarray(cam.astype(np.float32), mode="F").resize((size, size), Image.BILINEAR)
        cam = np.asarray(img)
        cam = cam - cam.min()
        return cam / max(float(cam.max()), 1e-8)


def load_all() -> Dict:
    m: Dict = {"cnn": None, "stage_sk": None, "days_sk": None}
    if (MODELS_DIR / "cnn_final.onnx").exists():
        m["cnn"] = OnnxCNN(MODELS_DIR)
    for key, name in (("stage_sk", "final_model_stage.joblib"), ("days_sk", "final_model_days.joblib")):
        if (MODELS_DIR / name).exists():
            m[key] = joblib.load(MODELS_DIR / name)

    cnn_days_ok = m["cnn"] is not None and m["cnn"].meta.get("mode") == "multitask"
    choice, quality = {}, {}
    for task, sk_key, cnn_ok in (("stage", "stage_sk", m["cnn"] is not None), ("days", "days_sk", cnn_days_ok)):
        overall = core.best_row(MODELS_DIR, task)
        prefer_cnn = cnn_ok and (overall is None or str(overall["model"]).startswith("cnn_") or m[sk_key] is None)
        if prefer_cnn:
            choice[task], quality[task] = "cnn", core.best_row(MODELS_DIR, task, cnn_only=True)
        elif m[sk_key] is not None:
            choice[task], quality[task] = "sk", core.best_row(MODELS_DIR, task, cnn_only=False)
        else:
            raise FileNotFoundError(f"No model for {task} in {MODELS_DIR}")
    m["choice"], m["quality"] = choice, quality
    meta = m["cnn"].meta if m["cnn"] else {}
    m["crop_pad"] = float(meta.get("crop_pad", 0.08))
    m["crop_size"] = int(meta.get("crop_size", 320))
    m["conditions"] = list(meta.get("conditions", ["T10", "T20", "Tam"]))
    m["ready_stage"] = int(meta.get("ready_stage", 4))
    return m


def describe(m: Dict, task: str) -> str:
    if m["choice"][task] == "cnn":
        return f"rede neural {m['cnn'].meta['backbone']} (ONNX)"
    est = m[f"{task}_sk"]["model"].steps[-1][1].__class__.__name__
    return {"RandomForestRegressor": "Random Forest", "RandomForestClassifier": "Random Forest",
            "Ridge": "regressão linear", "LogisticRegression": "regressão logística"}.get(est, est) + " sobre cor e textura"


M = load_all()


def to_b64(img: Image.Image, size: int = 320, quality: int = 85) -> str:
    img = img.convert("RGB").copy()
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@api.get("/")
def root() -> Dict:
    return {"name": "AvocadoCheck API", "version": VERSION, "docs": "/docs", "info": "/api/info"}


@api.get("/api/info")
def info() -> Dict:
    q_stage, q_days = M["quality"].get("stage") or {}, M["quality"].get("days") or {}
    return {
        "version": VERSION,
        "conditions": [{"code": c, "label": core.CONDITION_LABELS.get(c, c)} for c in M["conditions"]],
        "ready_stage": M["ready_stage"],
        "stage_names": core.STAGE_NAMES,
        "models": {"stage": describe(M, "stage"), "days": describe(M, "days")},
        "quality": {
            "stage_accuracy": q_stage.get("pooled_accuracy"),
            "stage_within_1": q_stage.get("pooled_within_1_stage"),
            "days_mae": q_days.get("pooled_MAE"),
            "days_within_2": q_days.get("pooled_within_2_days"),
        },
        "detector": "threshold",
    }


@api.post("/api/analyze")
async def analyze(file: UploadFile = File(...), condition: str = Form("T10")) -> Dict:
    if condition not in M["conditions"]:
        condition = M["conditions"][0]
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(await file.read()))).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Não consegui abrir a imagem. Envie uma foto JPG ou PNG.")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    t0 = time.time()

    boxes, method = core.detect_fruits(img, None, 0.05, 0.01)
    boxes = sorted(boxes, key=lambda b: (b[0][2] - b[0][0]) * (b[0][3] - b[0][1]), reverse=True)[:MAX_FRUITS]
    boxes = sorted(boxes, key=lambda b: (round(b[0][1] / max(img.size[1] / 4, 1)), b[0][0]))
    w, h = img.size
    fruits: List[Dict] = []
    for box, _ in boxes:
        crop = P.square_crop(img, box, M["crop_pad"], M["crop_size"])
        probs: Optional[np.ndarray] = None
        days: Optional[float] = None
        cam_img = None
        if M["cnn"] is not None and "cnn" in M["choice"].values():
            p_cnn, d_cnn, fmap = M["cnn"].run(crop, condition)
            if M["choice"]["stage"] == "cnn":
                probs = p_cnn
                cam_img = core.overlay_cam(crop, M["cnn"].cam(fmap, int(p_cnn.argmax())))
            if M["choice"]["days"] == "cnn":
                days = d_cnn
        if probs is None:
            sk = M["stage_sk"]
            probs = P.predict_proba_aligned(sk["model"], core.features_for_crop(crop, sk["columns"], condition))[0]
        if days is None:
            sk = M["days_sk"]
            days = max(0.0, float(sk["model"].predict(core.features_for_crop(crop, sk["columns"], condition))[0]))
        x1, y1, x2, y2 = [float(v) for v in box]
        fruits.append({
            "crop": to_b64(crop),
            "cam": to_b64(cam_img) if cam_img is not None else None,
            "stage": int(np.argmax(probs)) + 1,
            "probs": [round(float(p), 4) for p in probs],
            "confidence": round(float(np.max(probs)), 4),
            "days": round(float(days), 2),
            "box": [x1 / w, y1 / h, x2 / w, y2 / h],
            "detector": method,
        })
    if not fruits:
        raise HTTPException(status_code=422, detail="Nenhum abacate encontrado na foto.")
    return {"condition": condition, "fruits": fruits, "seconds": round(time.time() - t0, 2)}


# Optional: serve the web app too, so the API alone is enough for a quick test.
WEB = HERE.parent / "web"
if WEB.exists():
    from fastapi.staticfiles import StaticFiles

    api.mount("/app", StaticFiles(directory=WEB, html=True), name="web")
