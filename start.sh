#!/bin/bash
if [ ! -f "yolov8n.pt" ]; then
    wget -O yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
fi
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
