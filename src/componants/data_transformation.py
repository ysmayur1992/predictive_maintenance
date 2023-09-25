import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import OneHotEncoder

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class Data_Transformation_Config:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")


class Data_Transformation:
    def __init__(self):
        self.data_transformation_config=Data_Transformation_Config()

    def get_data_transformer_object(self):
        try:
            numerical = ['Air temperature [K]',
                          'Process temperature [K]',
                          'Rotational speed [rpm]',
                          'Torque [Nm]',
                          'Tool wear [min]',
                          'Power' ]
            
            categorical = ["Type"]

            num_pipeline= Pipeline(
                steps=[
                ("scaler",MaxAbsScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("One Hote Encoding",OneHotEncoder()),
                    ("Scaling",MaxAbsScaler())
                ]
            )
            #logging.info(f"Categorical columns: {categorical_feature} preprocessing done")
            logging.info("preprocessing done")

            preprocessor=ColumnTransformer(
                [
                ("Numerical Features",num_pipeline,numerical),
                ("Categorical Features",cat_pipeline,categorical)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
   
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="Machine failure"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        
