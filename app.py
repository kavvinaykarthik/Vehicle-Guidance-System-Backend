import cv2
import numpy as np
import pyttsx3
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO  # YOLOv8 model
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React Native

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

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
    engine.say(message)
    engine.runAndWait()

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

                # Generate voice alerts for close objects
                if distance <= 5:
                    voice_alert(f"Warning! {class_name} detected within {distance} meters.")

    return detected_objects

# API Route to process images from mobile
@app.route("/detect", methods=["POST"])
def detect_objects():
    try:
        data = request.json
        image_data = data["image"]  # Get base64 image data

        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        frame = np.array(image)

        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Process the frame
        detected_objects = process_frame(frame)

        return jsonify({"status": "success", "detections": detected_objects})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Home route
@app.route("/")
def home():
    return "Flask API is running!"

# Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
