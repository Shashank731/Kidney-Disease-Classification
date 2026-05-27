from kidney_disease_prediction import logger
from kidney_disease_prediction.config.configuration import ConfigurationManager 
from kidney_disease_prediction.components.data_splitting import DataSplitting

STAGE_NAME = "Data Splitting Stage"

class DataSplittingPipeline:
    def __init__(self):
        pass

    def main(self):
        logger.info("Starting data splitting stage")

        try:
            config = ConfigurationManager()

            data_splitting_config = config.get_data_splitting_config()

            data_splitting = DataSplitting(config=data_splitting_config)

            data_splitting.split_data()

            logger.info("Data splitting stage completed successfully")

        except Exception as e:
            logger.exception(e)
            raise e
        
if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        data_splitting_pipeline = DataSplittingPipeline()
        data_splitting_pipeline.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e