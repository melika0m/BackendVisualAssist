# main.py
import json
import base64
from io import BytesIO
from typing import List
from PIL import Image
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from transformers import AutoImageProcessor, DFineForObjectDetection

app = FastAPI(title="D-FINE Object Detection")

@app.get("/")
def root():
    return {"message": "Hello, World!"}

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

# -------- Optional REST (still available) --------
class ImageRequest(BaseModel):
    image_data: str

@app.post("/detect_objects_multipart")
async def detect_objects_multipart(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = Image.open(BytesIO(content)).convert("RGB")
        w, h = img.size

        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor([[h, w]]).to(device), threshold=0.4
        )

        detections = []
        for result in results:
            for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                if score > 0.4:
                    x1, y1, x2, y2 = box.to("cpu").numpy().astype(int).tolist()
                    detections.append({
                        "label": model.config.id2label[label.item()],
                        "score": float(score),
                        "bbox": [x1, y1, x2, y2],
                    })
        return {"width": w, "height": h, "detections": detections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------- WebSocket real-time --------
# Protocol:
#   Client -> Server: JSON {"image_b64": "<base64 of JPEG>", "req_id": "<optional id>"}
#   Server -> Client: JSON {"width": W, "height": H, "detections":[...], "req_id": "<same if sent>"}
@app.websocket("/ws_detect")
async def ws_detect(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)

            b64 = data.get("image_b64")
            req_id = data.get("req_id")

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
                    outputs, target_sizes=torch.tensor([[h, w]]).to(device), threshold=0.4
                )
            except Exception as e:
                await websocket.send_text(json.dumps({"error": f"inference failed: {e}", "req_id": req_id}))
                continue

            dets = []
            for result in results:
                for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                    if score > 0.4:
                        x1, y1, x2, y2 = box.to("cpu").numpy().astype(int).tolist()
                        dets.append({
                            "label": model.config.id2label[label.item()],
                            "score": float(score),
                            "bbox": [x1, y1, x2, y2],
                        })

            resp = {"width": w, "height": h, "detections": dets}
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
