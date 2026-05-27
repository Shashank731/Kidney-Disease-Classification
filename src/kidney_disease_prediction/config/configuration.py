from pathlib import Path
import os 
from kidney_disease_prediction.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from kidney_disease_prediction.utils.common import read_yaml, create_directories,save_json
from kidney_disease_prediction.entity.config_entity import *
from kidney_disease_prediction import logger


class ConfigurationManager:
    def __init__(
        self,
        config_file_path: Path = CONFIG_FILE_PATH,
        params_file_path: Path = PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)

        # Create root artifacts directory
        create_directories([self.config.artifacts_root])

        logger.info("Configuration and params loaded successfully")
        logger.info("artifacts_root directory created successfully")
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        # Create data ingestion directory
        create_directories([config.root_dir])

        # Basic validation
        if not config.source_URL:
            raise ValueError("source_URL is missing in config")

        return DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file
        )
    
    def get_data_splitting_config(self) -> data_splittingConfig:
        config = self.config.data_splitting


        return data_splittingConfig(
            root_dir=config.root_dir,
            train_data_dir=config.train_data_dir,
            test_data_dir=config.test_data_dir,
            split_ratio=config.split_ratio
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        # Create prepare base model directory
        create_directories([config.root_dir])

        # Basic validation
        if not config.root_dir:
            raise ValueError(f"in config. yaml prepare_base_model.root_dir is empty.")

        return PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            INCLUDING_TOP=self.params.INCLUDING_TOP,
            CLASSES=self.params.CLASSES
        )
    
    def get_training_config(self) -> TrainingConfig:
        config = self.config.training
        params = self.params

        training_data = self.config.data_splitting.train_data_dir

        create_directories([config.root_dir])

        return TrainingConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            base_model_path=Path(config.base_model_path),
            training_data=Path(training_data),
            EPOCHS=params.EPOCHS,
            BATCH_SIZE=params.BATCH_SIZE,
            IS_AUGMENTATION=params.AUGMENTATION,
            IMAGE_SIZE=params.IMAGE_SIZE,
            CLASSES=params.CLASSES
        )
    def get_evaluation_config(self) -> EvaluationConfig:

        eval_config = EvaluationConfig(

            path_to_model=Path(
                "artifacts/training/trained_model.pth"
            ),

            test_data_path=Path(
                "artifacts/data_splitting/test"
            ),

            all_params=self.params,

            mlflow_uri=(
                "https://dagshub.com/"
                "Shashank731/"
                "Kidney-Disease-Classification.mlflow"
            ),

            param_image_size=self.params.IMAGE_SIZE,

            param_batch_size=self.params.BATCH_SIZE
        )

        return eval_config