from fastapi import FastAPI, File, UploadFile
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import os
import shutil
import tempfile
from fastapi.responses import FileResponse

app = FastAPI()

# Directories and Model Path
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n.pt")

# Load YOLO model
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Downloading YOLO model...")
    torch.hub.download_url_to_file(
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        MODEL_PATH
    )
    print("YOLO model downloaded successfully!")

model = YOLO(MODEL_PATH)

@app.post("/detect/video/")
async def detect_objects_in_video(file: UploadFile = File(...)):
    # Save the uploaded video
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, file.filename)
    
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Cannot open video file"}

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Output video file
    output_path = os.path.join(temp_dir, "processed_" + file.filename)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for result in results:
            if result.boxes:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_name = result.names[int(box.cls[0])]
                    confidence = box.conf[0].item()
                    distance = round(10 / confidence, 2) if confidence > 0 else "Unknown"

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name}: {distance}m"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    # Release resources
    cap.release()
    out.release()

    return FileResponse(output_path, filename="processed_" + file.filename, media_type="video/mp4")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
