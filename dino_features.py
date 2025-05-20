
import torch
from transformers import AutoFeatureExtractor, AutoModel
from torchvision import transforms
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "facebook/dino-vitb16"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_patch_features(frames_folder):
    features = []
    for fname in sorted(os.listdir(frames_folder)):
        img = Image.open(os.path.join(frames_folder, fname)).convert("RGB")
        inputs = feature_extractor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state.squeeze(0)  # [197, 768]
            patch_tokens = outputs[1:]  # Remove CLS
            features.append(patch_tokens.cpu())
    return torch.stack(features)  # [T, P, D]
