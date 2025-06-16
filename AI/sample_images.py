import os
import shutil
import random

SRC_ROOT = "dataset/train"            # Your original dataset path
DEST_ROOT = "dataset_sampled/train"   # New path where selected images will be copied
NUM_IMAGES = 20                       # Number of images per class

os.makedirs(DEST_ROOT, exist_ok=True)

for class_name in os.listdir(SRC_ROOT):
    class_path = os.path.join(SRC_ROOT, class_name)
    if not os.path.isdir(class_path):
        continue

    # Get all image files
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(images) < NUM_IMAGES:
        print(f"⚠️ Warning: {class_name} only has {len(images)} images.")
        selected = images
    else:
        selected = random.sample(images, NUM_IMAGES)

    # Create destination subfolder
    dest_class_path = os.path.join(DEST_ROOT, class_name)
    os.makedirs(dest_class_path, exist_ok=True)

    # Copy selected files
    for image in selected:
        src_img = os.path.join(class_path, image)
        dest_img = os.path.join(dest_class_path, image)
        shutil.copyfile(src_img, dest_img)

    print(f"✅ Copied {len(selected)} images from '{class_name}'")

print("🎉 Sampling complete!")
