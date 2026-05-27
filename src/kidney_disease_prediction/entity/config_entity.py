from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path

@dataclass
class data_splittingConfig:
    root_dir: Path
    train_data_dir: Path
    test_data_dir: Path
    split_ratio: float
    
@dataclass
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    INCLUDING_TOP: bool
    CLASSES: int

@dataclass  
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    base_model_path: Path
    training_data: Path
    EPOCHS: int
    BATCH_SIZE: int
    IS_AUGMENTATION: bool
    IMAGE_SIZE: list
    CLASSES: int

@dataclass
class EvaluationConfig:
    path_to_model: Path
    test_data_path: Path
    all_params: dict
    mlflow_uri: str
    param_image_size: list
    param_batch_size: int