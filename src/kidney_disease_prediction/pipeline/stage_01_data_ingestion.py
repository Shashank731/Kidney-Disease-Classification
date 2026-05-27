from kidney_disease_prediction.config.configuration import ConfigurationManager
from kidney_disease_prediction.components.data_ingestion import DataIngestion
from kidney_disease_prediction import logger

STAGE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        self.config = ConfigurationManager()

    def main(self):
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        data_ingestion_config = self.config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\n")



if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        data_ingestion_training_pipeline = DataIngestionTrainingPipeline()
        data_ingestion_training_pipeline.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e