# Backend (FastAPI) for Vehicle Guidance System
from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from ultralytics import YOLO
import uvicorn
from gtts import gTTS
import os

app = FastAPI()

# Initialize YOLO model
model = YOLO("yolov8n.pt")  # Use downloaded model

def voice_alert(text):
    tts = gTTS(text=text, lang="en")
    tts.save("alert.mp3")
    os.system("mpg321 alert.mp3")  # Use appropriate player for your OS

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
                voice_alert(alert)
                break
    
    return {"alert": alert}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
