from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("/home/vijeth/major-project/AI/runs/detect/monuments_yolov8/weights/best.pt")
results = model('/home/vijeth/major-project/AI/dataset/test/images/25_jpg.rf.40db2b9c0c33953e6f5319413ce7a02f.jpg')[0]

# Extract boxes
crops = []
for box in results.boxes.xyxy.cpu().numpy():
    x1, y1, x2, y2 = map(int, box)
    cropped = cv2.imread('/home/vijeth/major-project/AI/dataset/test/images/25_jpg.rf.40db2b9c0c33953e6f5319413ce7a02f.jpg')[y1:y2, x1:x2]
    crops.append(cropped)
