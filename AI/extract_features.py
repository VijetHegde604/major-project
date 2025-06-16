import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as T
from torchvision.models.vision_transformer import vit_b_16

model = vit_b_16(pretrained=True)
model.eval()

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

def extract_feature(img_path):
    image = Image.open(img_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        feature = model(tensor)
    return feature.squeeze().numpy()

def extract_from_dir(root, out_prefix):
    vectors, labels = [], []
    for cls in tqdm(os.listdir(root)):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir): continue
        for file in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, file)
            try:
                vec = extract_feature(fpath)
                vectors.append(vec)
                labels.append(cls)
            except:
                continue
    np.save(f"{out_prefix}_vectors.npy", np.stack(vectors))
    np.save(f"{out_prefix}_labels.npy", np.array(labels))

# Run both sets
extract_from_dir("dataset/train", "gallery")
extract_from_dir("dataset/test", "query")
