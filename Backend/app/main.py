# main.py
import time
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile, File
from PIL import Image
import torch
import base64
from io import BytesIO
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

        # Return original image size so the app can scale boxes correctly
        return {"width": w, "height": h, "detections": detections}
    except Exception as e:
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
