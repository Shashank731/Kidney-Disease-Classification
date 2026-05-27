from kidney_disease_prediction.components.model_evaluation_mlflow import Evaluation
from kidney_disease_prediction.config.configuration import ConfigurationManager
from kidney_disease_prediction import logger

STAGE_NAME = "Model Evaluation Stage"


class EvaluationPipeline:

    def __init__(self):
        self.config = ConfigurationManager()

    def main(self):
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")

        eval_config = self.config.get_evaluation_config()

        evaluation = Evaluation(config=eval_config)

        logger.info("Starting model evaluation")

        evaluation.evaluation()

        evaluation.log_into_mlflow()

        logger.info("Model evaluation completed")
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\n")

if __name__ == "__main__":
    try:
        evaluation_pipeline = EvaluationPipeline()
        evaluation_pipeline.main()
    except Exception as e:
        logger.exception(e)
        raise e