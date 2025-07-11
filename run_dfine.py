import torch
import cv2
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, DFineForObjectDetection

# Load model and processor
processor = AutoImageProcessor.from_pretrained("ustc-community/dfine-xlarge-coco")
model = DFineForObjectDetection.from_pretrained("ustc-community/dfine-xlarge-coco")
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Start webcam
cap = cv2.VideoCapture(0)  # 0 = default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame (OpenCV uses BGR, we convert to RGB)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Preprocess
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_object_detection(
        outputs, target_sizes=torch.tensor([image.size[::-1]]).to(device), threshold=0.4
    )

    for result in results:
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            if score > 0.4:
                label_name = model.config.id2label[label.item()]
                box = box.to("cpu").numpy().astype(int)
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label_name}: {score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display frame
    cv2.imshow("D-FINE Real-Time Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
