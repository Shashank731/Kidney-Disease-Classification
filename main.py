from kidney_disease_prediction import logger
from kidney_disease_prediction.components import data_splitting
from kidney_disease_prediction.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from kidney_disease_prediction.pipeline.stage_011_data_splitting import DataSplittingPipeline
from kidney_disease_prediction.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline
from kidney_disease_prediction.pipeline.stage_03_model_training import ModelTrainingPipeline
from kidney_disease_prediction.pipeline.stage_04_model_evaluation import EvaluationPipeline


STAGE_NAME = "Data Ingestion Stage"
if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        data_ingestion_training_pipeline = DataIngestionTrainingPipeline()
        data_ingestion_training_pipeline.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Data Splitting Stage"
if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        data_splitting_pipeline = DataSplittingPipeline()
        data_splitting_pipeline.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Prepare Base Model Stage"
if __name__ == "__main__":
    try : 
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        prepare_base_model_pipeline = PrepareBaseModelPipeline()
        prepare_base_model_pipeline.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\n")

    except Exception as e:
        logger.exception(e)
        raise e
    

STAGE_NAME = "Model traning"
if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        model_training_pipeline = ModelTrainingPipeline()
        model_training_pipeline.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
    

STAGE_NAME = "Model Evaluation Stage"

if __name__ == "__main__":
    try:
        evaluation_pipeline = EvaluationPipeline()
        evaluation_pipeline.main()
    except Exception as e:
        logger.exception(e)
        raise e
