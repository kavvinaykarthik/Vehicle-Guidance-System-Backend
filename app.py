# Backend (FastAPI) for Vehicle Guidance System
type: code/python

from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import pyttsx3
from ultralytics import YOLO
import uvicorn

app = FastAPI()

# Initialize YOLO model
model = YOLO("weights/yolov8n.pt")

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = model(frame)
    alert = ""
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = box.conf[0]
            cls = int(box.cls[0])
            class_name = model.names[cls]
            
            if conf > 0.5 and class_name.lower() in ["car", "bus", "traffic light"]:
                alert = f"Warning! {class_name} detected."
                engine.say(alert)
                engine.runAndWait()
                break
    
    return {"alert": alert}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
