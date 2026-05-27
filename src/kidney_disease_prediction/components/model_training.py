import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from kidney_disease_prediction import logger
from kidney_disease_prediction.entity.config_entity import TrainingConfig


class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config

    # 🔹 Step 1: Data Loader
    def get_data_loader(self):
        transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
          )
        ])

        dataset = datasets.ImageFolder(
            root=self.config.training_data,
            transform=transform
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True
        )

        logger.info(f"Loaded {len(dataset)} training images")

        return dataloader

    # 🔹 Step 2: Load Model
    def load_model(self):
        model = models.densenet121(weights=None)

        # Replace classifier (same as stage 2)
        model.classifier = nn.Linear(
            model.classifier.in_features,
            self.config.CLASSES
        )

        # Load weights from stage 2
        model.load_state_dict(torch.load(self.config.base_model_path))

        logger.info("Base model loaded for training")

        return model

    # 🔹 Step 3: Training loop
    def train(self):
        train_loader = self.get_data_loader()
        model = self.load_model()

        for param in model.parameters():
                param.requires_grad = False
            
        #unfreeze last dense block
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True
            
        # unfreeze final norm layer
        for param in model.features.norm5.parameters():
            param.requires_grad = True
            
        # unfreeze classifier layer
        for param in model.classifier.parameters():
            param.requires_grad = True
        logger.info("Feature layers frozen")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4
      )

        logger.info("Starting training")

        correct = 0
        total = 0

        for epoch in range(self.config.EPOCHS):
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _,predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            final_loss = running_loss / len(train_loader)
            accuracy = 100 * correct / total
            logger.info(
                f"Epoch [{epoch+1}/{self.config.EPOCHS}] "
                f"- Loss: {final_loss:.4f} "
                f"- Accuracy: {accuracy:.2f}%"
           )

        # 🔹 Step 4: Save trained model
        os.makedirs(os.path.dirname(self.config.trained_model_path), exist_ok=True)
        torch.save(model.state_dict(), self.config.trained_model_path)

        logger.info(f"Training completed. Model saved at {self.config.trained_model_path}")

