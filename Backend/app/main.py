# main.py
import time
import base64
from io import BytesIO

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, DFineForObjectDetection


# -----------------------------
# FastAPI app + simple health
# -----------------------------
app = FastAPI(title="D-FINE Object Detection")

@app.get("/")
def root():
    return {"message": "Hello, World!"}

@app.get("/ping")
def ping():
    return {"ok": True}


# -----------------------------
# Load model & processor once
# -----------------------------
try:
    print("[DFINE] Loading processor/model…")
    processor = AutoImageProcessor.from_pretrained("ustc-community/dfine-xlarge-coco")
    model = DFineForObjectDetection.from_pretrained("ustc-community/dfine-xlarge-coco")
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[DFINE] Model ready on device: {device}")
except Exception as e:
    print(f"[DFINE][FATAL] Failed to load model: {e}")
    raise


# -----------------------------
# Request schema
# -----------------------------
class ImageRequest(BaseModel):
    image_data: str  # Base64-encoded image (no data URL prefix)


# -----------------------------
# Inference endpoint (base64)
# -----------------------------
@app.post("/detect_objects")
async def detect_objects(request: ImageRequest):
    try:
        t0 = time.time()

        # 1) Decode base64 -> PIL, fix orientation, force RGB
        img_bytes = base64.b64decode(request.image_data)
        img = Image.open(BytesIO(img_bytes))
        img = ImageOps.exif_transpose(img).convert("RGB")

        # 2) Preprocess
        inputs = processor(images=img, return_tensors="pt").to(device)

        # 3) Forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # 4) Post-process with a relatively low threshold (tune as needed)
        #    NOTE: target_sizes expects (height, width)
        results = processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([img.size[::-1]]).to(device),
            threshold=0.20,   # try 0.10–0.40 depending on what you see
        )

        detections = []
        for result in results:
            for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                if score > 0.20:
                    label_name = model.config.id2label[label.item()]
                    x1, y1, x2, y2 = box.to("cpu").numpy().astype(int).tolist()
                    detections.append({
                        "label": label_name,
                        "score": float(score),
                        "bbox": [x1, y1, x2, y2],
                    })

        t1 = time.time()
        print(f"[DFINE] detections={len(detections)} latency={(t1 - t0):.2f}s size={img.size}")

        return {"detections": detections}

    except Exception as e:
        print(f"[DFINE][ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Optional: run with `python main.py`
# (you can still use `uvicorn main:app --host 0.0.0.0 --port 8000`)
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # set True only for local dev with file watching
    )
