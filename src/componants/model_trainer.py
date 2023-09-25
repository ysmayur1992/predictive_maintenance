import os
import sys
from dataclasses import dataclass
from imblearn.over_sampling import SVMSMOTE

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            logging.info("Using SMOTE for sampling purposes")
            oversample = SVMSMOTE(random_state = 42)
            X_train, y_train = oversample.fit_resample(X_train, y_train)
            logging.info("Sampling of data done...")

            classifier_models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "CatBoosting Regressor": CatBoostClassifier(verbose=False),
                "AdaBoost Regressor": AdaBoostClassifier(),
                "KNN Classifier":KNeighborsClassifier(),
                "Gaussian Naive Bayes":GaussianNB()
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=classifier_models)
            
            sorted_model = dict(sorted(model_report.items(),key= lambda x:x[1],reverse=True))

            best_model_name = list(sorted_model.keys())[0]
            best_model = classifier_models[best_model_name]
            best_model_score = list(sorted_model.values())[0]
            
            if model_report[best_model_name] < 0.6:
                raise CustomException("No good model found... all the models have low accuracy")            
            

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            
            return best_model_name,best_model_score*100
            
            
        except Exception as e:
            raise CustomException(e,sys)
        

