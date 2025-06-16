import torch
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
from PIL import Image

# Select device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ✅ Load DINOv2 ViT-S/14 from official FacebookResearch repo
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).eval()

# ✅ Use standard ImageNet normalization
transform = T.Compose([
    T.Resize((224, 224)),  # You can use (518, 518) for ViT-G/14
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# ✅ Feature extraction function
def extract_dino_feature(image_cv2):
    image_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(image_tensor)  # Output shape: (1, 768) for vits14
        feature = F.normalize(feature, dim=-1)  # Normalize for cosine similarity
    return feature.cpu().numpy().flatten()
