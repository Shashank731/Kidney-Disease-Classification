from sklearn.model_selection import train_test_split
from kidney_disease_prediction.entity.config_entity import data_splittingConfig
from kidney_disease_prediction import logger

import os
import shutil


class DataSplitting:
    def __init__(self, config: data_splittingConfig):
        self.config = config

    def split_data(self):

        classes = os.listdir(self.config.root_dir)

        for class_name in classes:

            class_path = os.path.join(self.config.root_dir, class_name)

            if not os.path.isdir(class_path):
                continue

            images = os.listdir(class_path)

            train_imgs, test_imgs = train_test_split(
                images,
                test_size=self.config.split_ratio,
                random_state=42,
                shuffle=True
            )

            train_class_dir = os.path.join(
                self.config.train_data_dir,
                class_name
            )

            test_class_dir = os.path.join(
                self.config.test_data_dir,
                class_name
            )

            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)

            for img in train_imgs:
                shutil.copy2(
                    os.path.join(class_path, img),
                    os.path.join(train_class_dir, img)
                )

            for img in test_imgs:
                shutil.copy2(
                    os.path.join(class_path, img),
                    os.path.join(test_class_dir, img)
                )

            logger.info(
                f"{class_name}: "
                f"{len(train_imgs)} train, "
                f"{len(test_imgs)} test"
            )

        logger.info("Data splitting completed successfully")