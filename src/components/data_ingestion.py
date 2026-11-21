import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.components import data_transformation
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_path : str=os.path.join("Artifacts","train.csv")
    test_path : str=os.path.join("Artifacts","test.csv")
    raw_path : str=os.path.join("Artifacts","raw.csv")
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def intiate_data(self):
        logging.info("Data Ingestion")
        try:
            df=pd.read_csv(os.path.join("notebook","data","stud.csv"))
            logging.info("Dataset Reading")
            os.makedirs(os.path.dirname(self.ingestion_config.train_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_path,index=False,header=True)
            logging.info("Train Test Split Initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=47)
            train_set.to_csv(self.ingestion_config.train_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_path,index=False,header=True)
            logging.info("Data Ingestion Complete")
            return(
                self.ingestion_config.train_path,
                self.ingestion_config.test_path  
            )
        except Exception as e:
            raise CustomException(e,sys)
