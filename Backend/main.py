# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import cv2
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, DFineForObjectDetection
from io import BytesIO
import base64

# FastAPI app
app = FastAPI()

# Load model and processor
processor = AutoImageProcessor.from_pretrained("ustc-community/dfine-xlarge-coco")
model = DFineForObjectDetection.from_pretrained("ustc-community/dfine-xlarge-coco")
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Request model input format
class ImageRequest(BaseModel):
    image_data: str  # Base64-encoded image data

@app.post("/detect_objects")
async def detect_objects(request: ImageRequest):
    try:
        # Decode the base64 image data
        img_data = base64.b64decode(request.image_data)
        img = Image.open(BytesIO(img_data))
        
        # Preprocess the image
        inputs = processor(images=img, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process the results
        results = processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor([img.size[::-1]]).to(device), threshold=0.4
        )

        detections = []
        for result in results:
            for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                if score > 0.4:
                    label_name = model.config.id2label[label.item()]
                    box = box.to("cpu").numpy().astype(int)
                    detections.append({
                        "label": label_name,
                        "score": float(score),
                        "bbox": box.tolist()
                    })
        
        return {"detections": detections}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

