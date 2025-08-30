# app/main.py
import time
import io
import base64
from typing import List, Dict, Any

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from transformers import AutoImageProcessor, DFineForObjectDetection

# ---------- FastAPI app ----------
app = FastAPI(title="DFINE Backend", version="0.1")

# Allow your dev phone & emulators; widen origins as you need
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # for now during dev; tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Health ----------
@app.get("/health")
async def health():
    return {"status": "ok"}

# ---------- Load model once ----------
processor = AutoImageProcessor.from_pretrained("ustc-community/dfine-xlarge-coco")
model = DFineForObjectDetection.from_pretrained("ustc-community/dfine-xlarge-coco")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------- Helpers ----------
MAX_SIDE = 800  # downscale to speed up inference

def pil_from_bytes(b: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(b)).convert("RGB")
    # downscale keeping aspect ratio (server-side safety)
    w, h = img.size
    m = max(w, h)
    if m > MAX_SIDE:
        scale = MAX_SIDE / m
        img = img.resize((int(w * scale), int(h * scale)))
    return img

def run_inference(img: Image.Image) -> Dict[str, Any]:
    t0 = time.time()
    inputs = processor(images=img, return_tensors="pt").to(device)
    t_pre = time.time()

    with torch.no_grad():
        outputs = model(**inputs)
    t_inf = time.time()

    results = processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([img.size[::-1]]).to(device),
        threshold=0.4,
    )
    t_post = time.time()

    detections = []
    for r in results:
        for score, label, box in zip(r["scores"], r["labels"], r["boxes"]):
            if score > 0.4:
                label_name = model.config.id2label[label.item()]
                box = box.to("cpu").numpy().astype(int).tolist()
                detections.append({"label": label_name, "score": float(score), "bbox": box})

    return {
        "detections": detections,
        "timings_ms": {
            "preprocess": round((t_pre - t0) * 1000),
            "inference": round((t_inf - t_pre) * 1000),
            "postprocess": round((t_post - t_inf) * 1000),
            "total": round((t_post - t0) * 1000),
        },
        "image_size": {"width": img.size[0], "height": img.size[1]},
    }

# ---------- JSON base64 endpoint ----------
class ImageRequest(BaseModel):
    image_data: str  # base64 JPEG/PNG

@app.post("/detect_objects")
async def detect_objects(request: ImageRequest):
    try:
        img_bytes = base64.b64decode(request.image_data)
        img = pil_from_bytes(img_bytes)
        return run_inference(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Multipart endpoint (optional, useful for Postman) ----------
@app.post("/detect_objects_multipart")
async def detect_objects_multipart(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = pil_from_bytes(img_bytes)
        return run_inference(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
