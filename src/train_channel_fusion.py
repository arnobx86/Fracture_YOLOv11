# X-ray + MRI Channel Fusion Training
from ultralytics import YOLO

model = YOLO("yolo11m.pt")

model.train(
    data="../fracture.yaml",
    imgsz=640,
    epochs=200,
    batch=16,
    optimizer="AdamW",
    lr0=1e-4,
    project="../runs",
    name="channel_fusion"
)
