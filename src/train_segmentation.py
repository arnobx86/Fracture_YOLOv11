# YOLOv11 Segmentation Training
from ultralytics import YOLO

model = YOLO("yolo11m-seg.pt")

model.train(
    data="../fracture-seg.yaml",
    imgsz=640,
    epochs=200,
    batch=8,
    project="../runs",
    name="segmentation"
)
