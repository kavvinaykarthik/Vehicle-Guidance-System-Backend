from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import pyttsx3
from ultralytics import YOLO
import uvicorn
from io import BytesIO

# Initialize FastAPI app
app = FastAPI()

# Load YOLO model
model = YOLO("weights/yolov8n.pt")

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    results = model(frame)
    alerts = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = box.conf[0]
            class_name = model.names[cls]
            
            if conf >= 0.5:  # Confidence threshold
                alerts.append(f"Warning! {class_name} detected.")
                engine.say(f"Warning! {class_name} detected.")
    
    engine.runAndWait()
    return {"alerts": alerts}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
