import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from dinov2_FE import extract_dino_feature
from ultralytics import YOLO

# Update these paths
img_dir = '/home/vijeth/major-project/AI/dataset/valid/images'
label_dir = '/home/vijeth/major-project/AI/dataset/valid/labels'
feature_bank = np.load('/home/vijeth/major-project/AI/hybrid_model/ref_features.npy', allow_pickle=True).item()
yolo_model = YOLO("/home/vijeth/major-project/AI/runs/detect/monuments_yolov8/weights/best.pt")

# Class ID to name mapping
id2label = {
    0: "charminar",
    1: "gateway-of-india",
    2: "mysore-palace",
    3: "qutub-minar",
    4: "taj-mahal"
}

def match_with_feature_bank(feat, bank):
    best_class, best_score = None, -1
    for class_name, class_feat in bank.items():
        score = np.dot(feat, class_feat) / (np.linalg.norm(feat) * np.linalg.norm(class_feat))
        if score > best_score:
            best_score, best_class = score, class_name
    return best_class

def get_gt_label(label_file):
    if not os.path.exists(label_file):
        return None
    with open(label_file, 'r') as f:
        first_line = f.readline().strip()
        if first_line:
            class_id = int(first_line.split()[0])
            return id2label[class_id]
    return None

# Start evaluating
y_true, y_pred = [], []

for file in tqdm(os.listdir(img_dir)):
    if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(img_dir, file)
    label_path = os.path.join(label_dir, os.path.splitext(file)[0] + '.txt')
    gt_label = get_gt_label(label_path)
    if gt_label is None:
        print(f"⚠️ No label for {file}, skipping...")
        continue

    img = cv2.imread(img_path)
    results = yolo_model(img)
    detections = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []

    if len(detections) > 0:
        # Use first detected box
        x1, y1, x2, y2 = map(int, detections[0])
        crop = img[y1:y2, x1:x2]
    else:
        crop = img  # fallback to full image

    feat = extract_dino_feature(crop)
    pred_label = match_with_feature_bank(feat, feature_bank)

    y_true.append(gt_label)
    y_pred.append(pred_label)

# Results
print("\n✅ Accuracy:", f"{accuracy_score(y_true, y_pred) * 100:.2f}%")
print("\n📊 Classification Report:\n", classification_report(y_true, y_pred))
