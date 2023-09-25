import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
        type: str,
        air_temperature: float,
        process_temperature: float,
        rotational_speed: float,
        torque: float,
        tool_wear: float,
        power: float):

        self.type = type
        self.air_temperature = air_temperature
        self.process_temperature = process_temperature
        self.rotational_speed = rotational_speed
        self.torque = torque
        self.tool_wear = tool_wear
        self.power = power

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Type": [self.type],
                "Air_Temperature": [self.air_temperature],
                "Process_Temperature": [self.process_temperature],
                "Rotational_Speed": [self.rotational_speed],
                "Torque": [self.torque],
                "Tool_Wear": [self.tool_wear],
                "Power": [self.power],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)