from kidney_disease_prediction.entity.config_entity import PrepareBaseModelConfig

import torch
import torch.nn as nn
from torchvision import models

from kidney_disease_prediction import logger


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model = None

    def get_base_model(self):
        try:
            logger.info("Loading model")

            # Load pretrained Densenet121 model
            self.model = models.densenet121(weights="IMAGENET1K_V1",progress=True)

            logger.info("Base Densenet121 loaded")

            # Replace classifier if needed
            if not self.config.INCLUDING_TOP:
                self.model.classifier = nn.Linear(
                    self.model.classifier.in_features,
                    self.config.CLASSES
                )

                logger.info(f"Modified classifier for {self.config.CLASSES} classes")


            # Save model weights
            torch.save(self.model.state_dict(), self.config.base_model_path)

            logger.info(f"Model saved at {self.config.base_model_path}")

        except Exception as e:
            logger.exception(e)
            raise