from ultralytics import YOLO
import os
import torch


device = "cpu"
# check if GPU is available
if torch.cuda.is_available():
    device = "1"


# Load YOLO model
model = YOLO("yolov8m.yaml")  # Replace with YOLOv10 pretrained weights if available

# Train the model
model.train(
    data="/opt/ml/input/data/all/data.yaml",  # SageMaker path for input data
    epochs=100,
    imgsz=640,
    batch=16,
    workers=8,
    project="/opt/ml/model",                      # Root directory for saving results
    device=device  # GPU index
)