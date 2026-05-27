from kidney_disease_prediction.config.configuration import ConfigurationManager
from kidney_disease_prediction.components.model_training import ModelTrainer
from kidney_disease_prediction import logger



STAGE_NAME = "Model Training Stage"

class ModelTrainingPipeline:
    def __init__(self):
        self.config = ConfigurationManager()

    def main(self):
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        training_config = self.config.get_training_config()
        model_trainer = ModelTrainer(config=training_config)
        logger.info("Model Trainer initialized")
        model_trainer.load_model()
        model_trainer.get_data_loader()
        logger.info("Starting model training")
        model_trainer.train()
        logger.info("model training completed")
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\n")


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        model_training_pipeline = ModelTrainingPipeline()
        model_training_pipeline.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e