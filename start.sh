#!/bin/bash
wget -O yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
