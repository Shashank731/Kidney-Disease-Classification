from pathlib import Path
from urllib.parse import urlparse

import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from torch.utils.data import Subset
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from kidney_disease_prediction.utils.common import save_json
from kidney_disease_prediction import logger
from kidney_disease_prediction.entity.config_entity import EvaluationConfig

import dagshub

dagshub.init(
    repo_owner="Shashank731",
    repo_name="Kidney-Disease-Classification",
    mlflow=True
)


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config


    # 🔹 Validation Data Loader
    def valid_dataloader(self):

        transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


        dataset = datasets.ImageFolder(
            root=self.config.test_data_path,
            transform=transform
         )

        self.valid_loader = DataLoader(
            dataset,
            batch_size=self.config.param_batch_size,
            shuffle=False
        )

        logger.info(f"Loaded {len(dataset)} validation images")


    # 🔹 Load trained model
    def load_model(self):

        model = models.densenet121(weights=False)


        model.classifier = nn.Linear(
            model.classifier.in_features,
            self.config.all_params.CLASSES
        )
        

        model.load_state_dict(
            torch.load(self.config.path_to_model)
        )

        logger.info("Trained model loaded successfully")

        return model


    # 🔹 Evaluation Loop
    def evaluation(self):

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = self.load_model()
        self.model.to(device)

        self.valid_dataloader()

        criterion = nn.CrossEntropyLoss()

        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in self.valid_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = self.model(images)

                loss = criterion(outputs, labels)

                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)

                correct += (predicted == labels).sum().item()

        final_loss = running_loss / len(self.valid_loader)

        accuracy = 100 * correct / total

        self.score = {
            "loss": final_loss,
            "accuracy": accuracy
        }

        logger.info(
            f"Validation Loss: {final_loss:.4f}, "
            f"Accuracy: {accuracy:.2f}%"
        )

        self.save_score()


    # 🔹 Save scores
    def save_score(self):

        save_json(
            path=Path("scores.json"),
            data=self.score
        )


    # 🔹 MLflow Logging
    def log_into_mlflow(self):

        tracking_url_type_store = urlparse(
            mlflow.get_tracking_uri()
        ).scheme

        mlflow.set_experiment(
            "Kidney-Disease-Classification"
        )

        with mlflow.start_run():

            mlflow.log_params(
                dict(self.config.all_params)
            )

            mlflow.log_metrics(
                self.score
            )

            if tracking_url_type_store != "file":

                mlflow.pytorch.log_model(
                    self.model,
                    "model",
                    registered_model_name="densenet121Model"
                )

            else:

                mlflow.pytorch.log_model(
                    self.model,
                    "model"
                )

        logger.info("MLflow logging completed")