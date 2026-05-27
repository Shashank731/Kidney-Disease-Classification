import os
import zipfile
from kidney_disease_prediction import logger
from kidney_disease_prediction.utils.common import get_size
from kidney_disease_prediction.entity.config_entity import DataIngestionConfig
from kaggle.api.kaggle_api_extended import KaggleApi


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) :
        """
        Fetch data from Kaggle
        """

        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file

            os.makedirs(zip_download_dir, exist_ok=True)

            logger.info(f"Downloading dataset from {dataset_url} into file {zip_download_dir}")

            api = KaggleApi()
            api.authenticate()

    
            api.dataset_download_files(
                dataset_url,
                path=zip_download_dir,
                unzip=True
            )


            logger.info(f"Download file from{dataset_url} into file {zip_download_dir} completed")


        except Exception as e:
            logger.exception(e)
            raise e