import os
import xml.etree.ElementTree as ET
from glob import glob
from tqdm import tqdm

# Paths
IMAGE_DIR = "./dataset/images"
ANNOTATION_DIR = "./dataset/annotation"
YOLO_LABELS_DIR = "./dataset/converted_labels"
os.makedirs(YOLO_LABELS_DIR, exist_ok=True)

# Class list will be built dynamically
class_names = []

def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh

# Loop through all XML files
xml_files = glob(os.path.join(ANNOTATION_DIR, "*.xml"))

for xml_file in tqdm(xml_files, desc="Converting XML to YOLO format"):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_file = root.find("filename").text
    image_path = os.path.join(IMAGE_DIR, image_file)
    if not os.path.exists(image_path):
        print(f"⚠️ Skipping missing image: {image_path}")
        continue

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    label_lines = []
    for obj in root.iter("object"):
        cls = obj.find("name").text
        if cls not in class_names:
            class_names.append(cls)
        cls_id = class_names.index(cls)

        xmlbox = obj.find("bndbox")
        b = (
            float(xmlbox.find("xmin").text),
            float(xmlbox.find("xmax").text),
            float(xmlbox.find("ymin").text),
            float(xmlbox.find("ymax").text)
        )
        bb = convert_bbox((w, h), b)
        label_lines.append(f"{cls_id} {' '.join([f'{x:.6f}' for x in bb])}")

    label_filename = os.path.splitext(image_file)[0] + ".txt"
    with open(os.path.join(YOLO_LABELS_DIR, label_filename), "w") as f:
        f.write("\n".join(label_lines))

# Optional: Print all detected class names
print("\n✅ Conversion complete. Classes detected:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")
