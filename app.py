from fastapi import FastAPI, File, UploadFile
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
import requests
import os
app = FastAPI()

# Ensure YOLO model is downloaded before starting
MODEL_PATH = "yolov8n.pt"
YOLO_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"

# Download the YOLO model if it does not exist
def download_yolo_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading YOLO model...")
        response = requests.get(YOLO_MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("YOLO model downloaded successfully!")

# Load YOLO model
download_yolo_model()
model = YOLO(MODEL_PATH)

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Run YOLOv8 model
    results = model(image)

    # Parse detections
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = result.names[int(box.cls[0])]
            distance = round(10 / box.conf[0].item(), 2)  # Mock distance calculation
            detections.append({"bbox": [x1, y1, x2, y2], "class": class_name, "distance_m": distance})

    return {"detections": detections}
