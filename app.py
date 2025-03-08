import cv2
import numpy as np
import base64
import io
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
from gtts import gTTS  # Google Text-to-Speech
from playsound import playsound  # Import playsound

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load YOLOv8 model
model = YOLO("weights/yolov8n.pt")

# Configuration
KNOWN_WIDTH = 2.0  # Approximate width of car in meters
FOCAL_LENGTH = 1000  # Camera focal length for distance estimation

# Function to estimate distance
def estimate_distance(bbox_width):
    return (KNOWN_WIDTH * FOCAL_LENGTH) / bbox_width if bbox_width else 0

# Function to generate voice alerts
def voice_alert(message):
    print(message)  # Print alert in console

    # Use Google Text-to-Speech (gTTS) and save as MP3
    tts = gTTS(text=message, lang="en")
    tts.save("alert.mp3")
    playsound("alert.mp3")  # Play the MP3

# Function to detect objects and process frame
def process_frame(frame):
    results = model(frame)
    detected_objects = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]

            if conf >= 0.5:
                bbox_width = x2 - x1
                distance = estimate_distance(bbox_width)
                distance = round(distance, 2)

                detected_objects.append({"object": class_name, "distance": distance})

                if distance <= 5:
                    voice_alert(f"Warning! {class_name} detected within {distance} meters.")

    return detected_objects

@app.route("/detect-video", methods=["POST"])
def detect_video():
    try:
        video_file = request.files["video"]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        video_file.save(temp_file.name)

        cap = cv2.VideoCapture(temp_file.name)
        all_detections = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detections = process_frame(frame)
            all_detections.extend(detections)

        cap.release()

        return jsonify({"status": "success", "detections": all_detections})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/")
def home():
    return "Flask API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
