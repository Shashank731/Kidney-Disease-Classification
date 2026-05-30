import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import models, transforms
import os


class PredictionPipeline:
    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def load_model(self, model_path: str):
        model = models.densenet121(weights=False)
        model.classifier = nn.Linear(model.classifier.in_features, 4)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model

    def predict(self, image) -> int:
        transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output, 1)
        
        class_names = {
            0: "Cyst",
            1: "Normal",
            2: "Stone",
            3: "Tumor"
        }
        return class_names[predicted.item()]