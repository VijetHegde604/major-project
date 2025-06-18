from ultralytics import YOLO
import cv2
import numpy as np
import os
import pandas as pd
from dinov2_FE import extract_dino_feature

# Load YOLO and feature bank
yolo_model = YOLO("/home/vijeth/major-project/AI/runs/detect/monuments_yolov8/weights/best.pt")
feature_bank = np.load('/home/vijeth/major-project/AI/hybrid_model/ref_features.npy', allow_pickle=True).item()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def match_with_feature_bank(feat, feature_bank):
    best_class = None
    best_score = -1
    for class_name, class_feat in feature_bank.items():
        score = cosine_similarity(feat, class_feat)
        if score > best_score:
            best_score = score
            best_class = class_name
    return best_class, best_score

def classify_image_with_fallback(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Couldn't read {image_path}")
        return None

    results = yolo_model(img)
    detections = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []

    if len(detections) > 0:
        x1, y1, x2, y2 = map(int, detections[0])  # Only use first detection for simplicity
        crop = img[y1:y2, x1:x2]
        feat = extract_dino_feature(crop)
        pred_class, score = match_with_feature_bank(feat, feature_bank)
        return os.path.basename(image_path), pred_class, score, False
    else:
        feat = extract_dino_feature(img)
        pred_class, score = match_with_feature_bank(feat, feature_bank)
        return os.path.basename(image_path), pred_class, score, True

# Loop over test folder
test_dir = "/home/vijeth/major-project/AI/hybrid_model/sample_data"
results = []

for filename in os.listdir(test_dir):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    full_path = os.path.join(test_dir, filename)
    result = classify_image_with_fallback(full_path)
    if result:
        results.append(result)

# Save results to CSV
df = pd.DataFrame(results, columns=["image_name", "predicted_label", "confidence", "used_fallback"])
df.to_csv("predictions2.csv", index=False)
print("✅ Saved predictions to predictions.csv")

# Optional fallback usage info
fallback_used = df["used_fallback"].mean()
print(f"📉 Fallback (full image) used for {fallback_used * 100:.2f}% of images")
