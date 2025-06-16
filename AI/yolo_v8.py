from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # You can change to yolov8s.pt if needed

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="monuments_yolov8",
    project="runs/detect"
)
