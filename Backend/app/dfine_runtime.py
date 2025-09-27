# dfine_runtime.py
import os, io, base64
from typing import Tuple, List, Dict
from PIL import Image
import torch
from transformers import AutoImageProcessor, DFineForObjectDetection

# ---- knobs via env / .env ----
MODEL_DIR  = os.getenv("MODEL_DIR", "/home/ubuntu/assistnav_models/assistnav_dfine_v2")
DETECT_THR = float(os.getenv("DETECT_THRESHOLD", "0.15"))
SPEAK_THR  = float(os.getenv("SPEAK_THRESHOLD",  "0.35"))
MAX_SIDE   = int(os.getenv("MAX_SIDE", "1280"))
FP16       = os.getenv("FP16", "0") == "1"

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_dtype  = torch.float16 if (FP16 and _device.type == "cuda") else torch.float32

# Load once
image_processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
model = DFineForObjectDetection.from_pretrained(MODEL_DIR).to(_device, dtype=_dtype).eval()
id2label = model.config.id2label  # {id: name}

def _resize_max_side(pil: Image.Image, max_side: int) -> Image.Image:
    w, h = pil.size
    m = max(w, h)
    if m <= max_side: return pil
    s = max_side / float(m)
    return pil.resize((int(w*s), int(h*s)), Image.Resampling.LANCZOS)

def _b64_to_rgb(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def predict_pil(img: Image.Image, threshold: float | None = None) -> tuple[list[dict], tuple[int,int]]:
    thr = float(DETECT_THR if threshold is None else threshold)
    img = _resize_max_side(img, MAX_SIDE)
    W, H = img.size
    inputs = image_processor(images=img, return_tensors="pt")
    inputs = {k: (v.to(_device, dtype=_dtype) if isinstance(v, torch.Tensor) else v) for k,v in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    res = image_processor.post_process_object_detection(
        outputs, target_sizes=torch.tensor([[H, W]], device=_device), threshold=thr
    )[0]

    dets: list[Dict] = []
    for s, lab, box in zip(res["scores"], res["labels"], res["boxes"]):
        score = float(s.detach().cpu())
        x1, y1, x2, y2 = [int(v) for v in box.detach().cpu().tolist()]
        dets.append({"label": id2label[int(lab)], "score": score, "bbox": [x1,y1,x2,y2]})
    return dets, (W, H)

def predict_b64(b64: str, threshold: float | None = None) -> dict:
    img = _b64_to_rgb(b64)
    dets, (W, H) = predict_pil(img, threshold)
    speakable = [d for d in dets if d["score"] >= SPEAK_THR]
    return {
        "image_size": {"width": W, "height": H},
        "detections": dets,
        "speakable": speakable,
        "detect_threshold": float(DETECT_THR if threshold is None else threshold),
        "speak_threshold": SPEAK_THR,
    }
