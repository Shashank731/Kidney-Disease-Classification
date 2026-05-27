from kidney_disease_prediction import logger
from kidney_disease_prediction.config.configuration import ConfigurationManager
from kidney_disease_prediction.components.prepare_base_model import PrepareBaseModel


STAGE_NAME = "Prepare Base Model Stage"

class PrepareBaseModelPipeline:
    def __init__(self):
        pass
    
    def main(self):
        logger.info("Starting prepare base model stage")

        try:
            config = ConfigurationManager()

            prepare_base_model_config = config.get_prepare_base_model_config()

            prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)

            prepare_base_model.get_base_model()

            logger.info("Prepare base model stage completed successfully")

        except Exception as e:
            logger.exception(e)
            raise e 




if __name__ == "__main__":
    try : 
        logger.info(f">>>>> stage 2 prepare base model started <<<<<")
        prepare_base_model_pipeline = PrepareBaseModelPipeline()
        prepare_base_model_pipeline.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\n")

    except Exception as e:
        logger.exception(e)
        raise e













