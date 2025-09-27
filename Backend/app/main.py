# main.py
import json
import base64
from io import BytesIO
from typing import List, Dict, Any, Tuple
from PIL import Image
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel
from transformers import AutoImageProcessor, DFineForObjectDetection
import time

app = FastAPI(title="D-FINE Object Detection (EN only)")

@app.get("/")
def root():
    return {"message": "D-FINE backend is running (EN-only guidance)."}

@app.get("/ping")
def ping():
    return {"ok": True}

# -------- Load model once --------
print("[DFINE] Loading processor/model…")
processor = AutoImageProcessor.from_pretrained("ustc-community/dfine-xlarge-coco")
model = DFineForObjectDetection.from_pretrained("ustc-community/dfine-xlarge-coco")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"[DFINE] Model ready on device: {device}")

# -------- Heuristics / Config --------
FOV_DEFAULT_DEG = 60.0   # assumed horizontal FOV
NEAR_H = 0.45            # bbox height / H >= NEAR_H -> "near"
MID_H  = 0.25            # bbox height / H >= MID_H  -> "mid" else "far"
MIN_SCORE_DEFAULT = 0.40
TOPK_DEFAULT = 3

# Classes with higher risk to navigation (COCO names)
HAZARD_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "dog", "cat"
}
CLASS_PRIORITY = {
    "person": 3, "car": 3, "bus": 3, "truck": 3, "motorcycle": 3,
    "bicycle": 2, "dog": 2, "cat": 2,
    "traffic light": 1, "stop sign": 1, "fire hydrant": 1
}

def zone_from_cx(cx: float, W: int) -> str:
    x = cx / max(W, 1)
    if x < 1/3: return "left"
    if x < 2/3: return "center"
    return "right"

def angle_from_cx(cx: float, W: int, fov_deg: float) -> float:
    # ~ -fov/2 at left edge, +fov/2 at right edge
    return ((cx / max(W,1)) - 0.5) * fov_deg

def range_bucket(y1: int, y2: int, H: int) -> str:
    h_ratio = (y2 - y1) / max(H, 1)
    if h_ratio >= NEAR_H: return "near"
    if h_ratio >= MID_H:  return "mid"
    return "far"

def severity_for(label: str, rng: str) -> str:
    prio = CLASS_PRIORITY.get(label, 1)
    if prio == 3 and rng in ("near", "mid"): return "high"
    if prio >= 2 and rng == "near": return "high"
    if rng == "mid": return "medium"
    return "low"

def action_suggestion(zone: str, rng: str, label: str) -> str:
    # Simple guidance rules
    hazard = label in HAZARD_CLASSES
    if hazard:
        if rng == "near":
            if zone == "center": return "stop"
            if zone == "left":   return "step_right"
            if zone == "right":  return "step_left"
        if rng == "mid":
            if zone == "center": return "slow"
            if zone == "left":   return "veer_right"
            if zone == "right":  return "veer_left"
    # non-hazard or far
    if zone == "left":  return "slight_right"
    if zone == "right": return "slight_left"
    return "forward"

def message_for(label: str, zone: str, rng: str, mode: str) -> str:
    ztxt = {"left":"on the left", "center":"ahead", "right":"on the right"}[zone]
    rtxt = {"near":"near", "mid":"medium distance", "far":"far"}[rng]
    if mode == "concis":
        return f"{label} {ztxt}, {rtxt}."
    return f"Caution: {label} {ztxt}, {rtxt}."

def enrich_detections(
    dets: List[Dict[str, Any]], W: int, H: int, *, mode: str, fov_deg: float
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    enriched = []
    announcements = []

    # sort by priority, score, and box size
    def det_key(d):
        label = d["label"]
        score = d["score"]
        x1,y1,x2,y2 = d["bbox"]
        size = (x2-x1)*(y2-y1)
        return (CLASS_PRIORITY.get(label, 0), score, size)

    for d in sorted(dets, key=det_key, reverse=True):
        x1,y1,x2,y2 = d["bbox"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        zone = zone_from_cx(cx, W)
        ang  = angle_from_cx(cx, W, fov_deg)
        rng  = range_bucket(y1, y2, H)
        sev  = severity_for(d["label"], rng)
        act  = action_suggestion(zone, rng, d["label"])
        msg  = message_for(d["label"], zone, rng, mode)
        enriched.append({
            **d,
            "position": {"cx": cx, "cy": cy, "zone": zone, "angle_deg": ang},
            "range": rng,
            "severity": sev,
            "action": act,
            "message": msg
        })

    # announcements: top-K by (priority, severity, score)
    def ann_key(ed):
        pr = CLASS_PRIORITY.get(ed["label"], 0)
        sev = {"high":2, "medium":1, "low":0}[ed["severity"]]
        return (pr, sev, ed["score"])

    top = sorted(enriched, key=ann_key, reverse=True)[:TOPK_DEFAULT]
    for ed in top:
        announcements.append({
            "label": ed["label"],
            "zone": ed["position"]["zone"],
            "range": ed["range"],
            "severity": ed["severity"],
            "action": ed["action"],
            "message": ed["message"],
        })

    # one fused directive
    global_guidance = {}
    if announcements:
        lead = announcements[0]
        action_texts = {
            "stop": "Stop.",
            "step_left": "Step left.",
            "step_right": "Step right.",
            "veer_left": "Veer left.",
            "veer_right": "Veer right.",
            "slow": "Slow down.",
            "slight_left": "Slight left.",
            "slight_right": "Slight right.",
            "forward": "Go forward."
        }
        global_guidance = {
            "action": lead["action"],
            "message": f"{lead['message']} {action_texts.get(lead['action'], '')}".strip()
        }
    return enriched, announcements, global_guidance

# -------- REST (multipart) --------
class ImageRequest(BaseModel):
    image_data: str  # kept for compatibility (unused)

@app.post("/detect_objects_multipart")
async def detect_objects_multipart(
    file: UploadFile = File(...),
    mode: str = Query(default="verbose", pattern="^(verbose|concis)$"),
    topk: int = Query(default=TOPK_DEFAULT, ge=1, le=10),
    min_score: float = Query(default=MIN_SCORE_DEFAULT, ge=0.0, le=1.0),
    fov_deg: float = Query(default=FOV_DEFAULT_DEG, ge=30.0, le=120.0),
    include_boxes: bool = Query(default=True)
):
    try:
        content = await file.read()
        img = Image.open(BytesIO(content)).convert("RGB")
        w, h = img.size

        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor([[h, w]]).to(device), threshold=min_score
        )

        detections = []
        for result in results:
            for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                if float(score) >= min_score:
                    x1, y1, x2, y2 = box.to("cpu").numpy().astype(int).tolist()
                    detections.append({
                        "label": model.config.id2label[label.item()],
                        "score": float(score),
                        "bbox": [x1, y1, x2, y2],
                    })

        enriched, announcements, global_guidance = enrich_detections(
            detections, w, h, mode=mode, fov_deg=fov_deg
        )

        resp = {
            "width": w, "height": h,
            "announcements": announcements[:topk],
            "global_guidance": global_guidance
        }
        if include_boxes:
            resp["detections"] = enriched
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------- WebSocket real-time --------
# Client -> Server JSON:
# {
#   "image_b64": "<base64 JPEG>",
#   "req_id": "<optional>",
#   "mode": "verbose|concis",
#   "min_score": 0.4,
#   "topk": 3,
#   "fov_deg": 60.0,
#   "include_boxes": true
# }
# Server -> Client JSON:
# {
#   "width": W, "height": H,
#   "announcements": [...], "global_guidance": {...},
#   "detections": [...], "req_id": "<same if sent>"
# }
@app.websocket("/ws_detect")
async def ws_detect(websocket: WebSocket):
    await websocket.accept()
    # per-connection cooldown so we don't repeat the exact same phrase too fast
    last_said = {}  # key: (label, zone, range) -> last_time
    COOLDOWN_SEC = 2.5
    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)

            b64 = data.get("image_b64")
            req_id = data.get("req_id")
            mode = data.get("mode", "concis")
            min_score = float(data.get("min_score", MIN_SCORE_DEFAULT))
            topk = int(data.get("topk", TOPK_DEFAULT))
            fov_deg = float(data.get("fov_deg", FOV_DEFAULT_DEG))
            include_boxes = bool(data.get("include_boxes", True))

            if not b64:
                await websocket.send_text(json.dumps({"error": "missing image_b64", "req_id": req_id}))
                continue

            # decode image
            try:
                img_bytes = base64.b64decode(b64)
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                await websocket.send_text(json.dumps({"error": f"bad image: {e}", "req_id": req_id}))
                continue

            w, h = img.size

            # run model
            try:
                inputs = processor(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                results = processor.post_process_object_detection(
                    outputs, target_sizes=torch.tensor([[h, w]]).to(device), threshold=min_score
                )
            except Exception as e:
                await websocket.send_text(json.dumps({"error": f"inference failed: {e}", "req_id": req_id}))
                continue

            dets = []
            for result in results:
                for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                    if float(score) >= min_score:
                        x1, y1, x2, y2 = box.to("cpu").numpy().astype(int).tolist()
                        dets.append({
                            "label": model.config.id2label[label.item()],
                            "score": float(score),
                            "bbox": [x1, y1, x2, y2],
                        })

            enriched, announcements, global_guidance = enrich_detections(
                dets, w, h, mode=mode, fov_deg=fov_deg
            )

            # cooldown filter
            now = time.monotonic()
            cooled = []
            for ann in announcements:
                key = (ann["label"], ann["zone"], ann["range"])
                t0 = last_said.get(key, 0.0)
                if now - t0 >= COOLDOWN_SEC:
                    cooled.append(ann)
                    last_said[key] = now
            final_anns = cooled[:topk] if cooled else (announcements[:1] if announcements else [])

            resp = {
                "width": w, "height": h,
                "announcements": final_anns,
                "global_guidance": global_guidance
            }
            if include_boxes:
                resp["detections"] = enriched
            if req_id is not None:
                resp["req_id"] = req_id
            await websocket.send_text(json.dumps(resp))

    except WebSocketDisconnect:
        print("[WS] client disconnected")
    except Exception as e:
        print(f"[WS] error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
