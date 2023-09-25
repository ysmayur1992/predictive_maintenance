import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.componants.data_transformation import Data_Transformation
from src.componants.data_transformation import Data_Transformation_Config

from src.componants.model_trainer import ModelTrainerConfig
from src.componants.model_trainer import ModelTrainer

@dataclass
class Data_Ingestion_Config:
    raw_data_path: str=os.path.join('artifacts',"raw_data.csv")
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")


class Data_Ingestion:
    def __init__(self):
        self.ingestion_config=Data_Ingestion_Config()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('Data\data.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.25,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
        

if __name__=="__main__":
    obj=Data_Ingestion()
    train_path,test_path=obj.initiate_data_ingestion()

    data_transformation=Data_Transformation()
    train_set,test_set,_=data_transformation.initiate_data_transformation(train_path=train_path,test_path=test_path)

    modeltrainer=ModelTrainer()
    name,score = modeltrainer.initiate_model_training(train_array=train_set,test_array=test_set)
    message = "The best model is {0} and it's score is {1}".format(name,score)
    print(message)