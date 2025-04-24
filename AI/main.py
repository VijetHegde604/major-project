from ultralytics import YOLO

# Load model
model = YOLO('yolov8n')


results = model.train(
    data="config.yaml",
    epochs=100,
    patience = 50,
    augment = True,
    name = 'run2'
)