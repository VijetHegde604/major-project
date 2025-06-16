from ultralytics import YOLO

model = YOLO("/home/vijeth/major-project/AI/runs/detect/monuments_yolov8/weights/best.pt")

metrics = model.val(
    data="/home/vijeth/major-project/AI/data.yaml",
    split="test",
    project="/home/vijeth/major-project/AI/runs/val_2",          # 📁 main output folder
    name="monuments_test_eval",  # 📁 subfolder inside project
    save=True                    # ✅ saves confusion matrix, predictions, etc.
)

print(metrics)
