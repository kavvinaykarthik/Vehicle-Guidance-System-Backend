from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from ultralytics import YOLO
import io

app = FastAPI()

# Load YOLO model
model = YOLO("weights/yolov8n.pt")

# Constants for distance estimation
KNOWN_WIDTH = 2.0  # Approximate object width in meters
FOCAL_LENGTH = 1000  # Adjust based on your camera

def estimate_distance(bbox_width):
    return (KNOWN_WIDTH * FOCAL_LENGTH) / bbox_width if bbox_width else 0

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model(frame)
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            class_name = model.names[cls]

            bbox_width = x2 - x1
            distance = estimate_distance(bbox_width)

            detections.append({
                "class": class_name,
                "confidence": float(conf),
                "distance_m": round(distance, 2),
                "bbox": [x1, y1, x2, y2]
            })

    return {"detections": detections}
