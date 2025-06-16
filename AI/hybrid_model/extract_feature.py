import os
import cv2
import numpy as np
from dinov2_FE import extract_dino_feature  # from your working extractor

ref_root = '/home/vijeth/major-project/AI/hybrid_model/feature_bank'
feature_bank = {}

for class_name in os.listdir(ref_root):
    class_dir = os.path.join(ref_root, class_name)
    if not os.path.isdir(class_dir):
        continue
    
    class_features = []
    for img_file in os.listdir(class_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Could not read {img_path}")
                continue
            feat = extract_dino_feature(img)
            class_features.append(feat)

    if class_features:
        mean_feat = np.mean(class_features, axis=0)
        feature_bank[class_name] = mean_feat
        print(f"✅ Added class '{class_name}' with {len(class_features)} samples.")

np.save('ref_features.npy', feature_bank)
print("🎉 Saved feature bank to 'ref_features.npy'")
