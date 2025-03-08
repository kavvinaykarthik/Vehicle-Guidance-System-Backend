import cv2
import numpy as np
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load YOLOv8 model (make sure you have the correct model weights)
model = YOLO("weights/yolov8n.pt")

# Configuration for distance estimation
KNOWN_WIDTHS = {
    "car": 2.0,  # Approximate width in meters
    "person": 0.5,
    "truck": 2.5,
    "bus": 2.8,
    "traffic light": 0.3,
    "stop sign": 0.5
}
FOCAL_LENGTH = 1000  # Camera focal length for distance estimation

# Function to estimate distance based on bounding box width
def estimate_distance(class_name, bbox_width):
    if class_name in KNOWN_WIDTHS and bbox_width > 0:
        return round((KNOWN_WIDTHS[class_name] * FOCAL_LENGTH) / bbox_width, 2)
    return None  # Return None if the object is not in the list

# Function to process the video frame and detect objects
def process_frame(frame):
    results = model(frame)
    detected_objects = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]  # Object name

            if conf >= 0.5:  # Only consider objects with high confidence
                bbox_width = x2 - x1
                distance = estimate_distance(class_name, bbox_width)

                if distance is not None:
                    detected_objects.append({"object": class_name, "distance": distance})

    return detected_objects

# API endpoint to process video and detect objects
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

# Home route
@app.route("/")
def home():
    return "Flask API is running!"

# Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
