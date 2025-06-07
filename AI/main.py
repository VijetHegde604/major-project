import torch
from torchvision import transforms
from PIL import Image

# Load lighter model → better for CPU
dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load image
image = Image.open('MainBefore.jpg').convert('RGB')
image_tensor = transform(image).unsqueeze(0)  # (1, 3, 518, 518)

# Extract features on CPU
with torch.no_grad():
    features = dinov2_model(image_tensor)

# Convert to numpy
features_np = features.cpu().numpy().flatten()
print("Feature vector shape:", features_np.shape)
