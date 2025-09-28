# main.py
import json
import base64
import time
from io import BytesIO
from typing import List, Dict, Any, Tuple, Optional

from PIL import Image
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel
from transformers import AutoImageProcessor, DFineForObjectDetection

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
    # base boundaries
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
    if zone == "left":  return "slight_right"
    if zone == "right": return "slight_left"
    return "forward"

def message_for(label: str, zone: str, rng: str, mode: str) -> str:
    ztxt = {"left":"on the left", "center":"ahead", "right":"on the right"}[zone]
    rtxt = {"near":"near", "mid":"medium distance", "far":"far"}[rng]
    if mode == "concis":
        return f"{label} {ztxt}, {rtxt}."
    return f"Caution: {label} {ztxt}, {rtxt}."

def iou(boxA, boxB) -> float:
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0: return 0.0
    areaA = (ax2-ax1)*(ay2-ay1)
    areaB = (bx2-bx1)*(by2-by1)
    return inter / max(1.0, areaA + areaB - inter)

class TrackerLite:
    """
    Very small per-connection tracker:
    - Associate by IoU + same label
    - Maintain stability count (frames seen)
    - Zone hysteresis (avoid left/center/right flapping near boundaries)
    """
    def __init__(self):
        self.tracks: List[Dict[str, Any]] = []  # each: {id,label,bbox,zone,range,score,seen,ts}
        self.next_id = 1

    def _smooth_zone(self, cx: float, W: int, prev_zone: Optional[str]) -> str:
        # hysteresis: widen the previous zone by ±6% of width to resist jitter
        x = cx / max(W, 1)
        if prev_zone == "left":
            if x < 0.36: return "left"
            return "center" if x < 0.66 else "right"
        if prev_zone == "center":
            if x < 0.30: return "left"
            if x < 0.70: return "center"
            return "right"
        if prev_zone == "right":
            if x > 0.64: return "right"
            return "center" if x >= 0.34 else "left"
        # no previous -> default zoning
        return zone_from_cx(cx, W)

    def update(self, dets: List[Dict[str, Any]], W: int, H: int) -> List[Dict[str, Any]]:
        now = time.monotonic()
        assigned = set()
        # match
        for d in dets:
            best_iou, best_idx = 0.0, -1
            for idx, tr in enumerate(self.tracks):
                if idx in assigned: continue
                if tr["label"] != d["label"]: continue
                i = iou(tr["bbox"], d["bbox"])
                if i > best_iou:
                    best_iou, best_idx = i, idx
            if best_iou >= 0.5 and best_idx >= 0:
                tr = self.tracks[best_idx]
                assigned.add(best_idx)
                # update track
                x1,y1,x2,y2 = d["bbox"]
                cx = (x1+x2)/2
                zone = self._smooth_zone(cx, W, tr.get("zone"))
                rng = range_bucket(y1, y2, H)
                tr.update({
                    "bbox": d["bbox"], "score": d["score"],
                    "zone": zone, "range": rng,
                    "seen": tr.get("seen", 0) + 1, "ts": now
                })
                d["track_id"] = tr["id"]
                d["stable"] = tr["seen"] >= 2
                d["zone_override"] = zone
            else:
                # new track
                x1,y1,x2,y2 = d["bbox"]
                cx = (x1+x2)/2
                zone = zone_from_cx(cx, W)
                rng = range_bucket(y1, y2, H)
                tr = {
                    "id": self.next_id, "label": d["label"],
                    "bbox": d["bbox"], "zone": zone, "range": rng,
                    "score": d["score"], "seen": 1, "ts": now
                }
                self.next_id += 1
                self.tracks.append(tr)
                d["track_id"] = tr["id"]
                d["stable"] = False
                d["zone_override"] = zone

        # prune stale (not matched this frame)
        keep = []
        for idx, tr in enumerate(self.tracks):
            if idx in assigned:
                keep.append(tr)
            else:
                # retain briefly (0.6s) to survive single-frame drops
                if now - tr.get("ts", now) < 0.6:
                    keep.append(tr)
        self.tracks = keep
        return dets

def enrich_detections(
    dets: List[Dict[str, Any]], W: int, H: int, *, mode: str, fov_deg: float, only_stable: bool = True
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    enriched = []
    announcements = []

    def det_key(d):
        label = d["label"]
        score = d["score"]
        x1,y1,x2,y2 = d["bbox"]
        size = (x2-x1)*(y2-y1)
        return (CLASS_PRIORITY.get(label, 0), score, size)

    for d in sorted(dets, key=det_key, reverse=True):
        x1,y1,x2,y2 = d["bbox"]
        cx = (x1 + x2) / 2
        zone_raw = d.get("zone_override")
        zone = zone_raw if isinstance(zone_raw, str) else zone_from_cx(cx, W)
        ang  = angle_from_cx(cx, W, fov_deg)
        rng  = range_bucket(y1, y2, H)
        sev  = severity_for(d["label"], rng)
        act  = action_suggestion(zone, rng, d["label"])
        msg  = message_for(d["label"], zone, rng, mode)
        enriched.append({
            **d,
            "position": {"cx": cx, "cy": (y1 + y2)/2, "zone": zone, "angle_deg": ang},
            "range": rng,
            "severity": sev,
            "action": act,
            "message": msg
        })

    # announcements prefer stable first
    def ann_key(ed):
        pr = CLASS_PRIORITY.get(ed["label"], 0)
        sev = {"high":2, "medium":1, "low":0}[ed["severity"]]
        st  = 1 if ed.get("stable") else 0
        return (st, pr, sev, ed["score"])

    sorted_enriched = sorted(enriched, key=ann_key, reverse=True)

    if only_stable:
        stable_list = [e for e in sorted_enriched if e.get("stable")]
        if stable_list:
            top = stable_list[:TOPK_DEFAULT]
        else:
            # fallback to highest priority if nothing stable yet
            top = sorted_enriched[:1]
    else:
        top = sorted_enriched[:TOPK_DEFAULT]

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

# -------- REST (kept simple; stateless so no tracking here) --------
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

        # no temporal tracker for REST
        enriched, announcements, global_guidance = enrich_detections(
            detections, w, h, mode=mode, fov_deg=fov_deg, only_stable=False
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

# -------- WebSocket with tracker-lite --------
@app.websocket("/ws_detect")
async def ws_detect(websocket: WebSocket):
    await websocket.accept()
    last_said: Dict[Tuple[str,str,str], float] = {}  # (label, zone, range) -> ts
    COOLDOWN_SEC = 2.5
    stop_hold_until = 0.0

    tracker = TrackerLite()

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

            # decode
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

            # ---- tracker-lite pass (stability + zone hysteresis)
            dets = tracker.update(dets, w, h)

            enriched, announcements, global_guidance = enrich_detections(
                dets, w, h, mode=mode, fov_deg=fov_deg, only_stable=True
            )

            # stop hold (keep "stop" for 1.0s)
            now = time.monotonic()
            lead_action = None
            if announcements:
                lead_action = announcements[0]["action"]
                if lead_action == "stop":
                    stop_hold_until = now + 1.0
            if now < stop_hold_until:
                # enforce stop in global guidance
                if global_guidance:
                    global_guidance["action"] = "stop"
                    if "message" in global_guidance and isinstance(global_guidance["message"], str):
                        if not global_guidance["message"].lower().endswith("stop."):
                            global_guidance["message"] = global_guidance["message"].rstrip() + " Stop."

            # cooldown filter
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
