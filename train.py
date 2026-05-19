from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Train model
model.train(
    data="data.yaml",
    epochs=20
)