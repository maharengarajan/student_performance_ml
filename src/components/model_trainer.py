import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor

from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("splitting train data and test data after transformation")
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "KNNeighour": KNeighborsRegressor(),
                "Catboost": CatBoostRegressor(verbose=False),
                "Xgboost": XGBRegressor(),
                "Adaboost": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
            }

            parameters = {"Linear Regression":{},
                          "Random Forest":{"n_estimators": [8,16,32,64,128,256]},
                          "Decision Tree": {},
                          "KNNeighour": {"n_neighbors": [10]},
                          "Catboost": {},
                          "Xgboost": {"n_estimators": [8,16,32,64,128,256]},
                          "Adaboost": {},
                          "Gradient Boosting": {}}
            

            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,
                                                param=parameters)

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            #if best_model < 0.6:
                #raise CustomException("No Best model found")
            logging.info(f"Best model found on both train and test dataset")

            save_object(self.model_trainer_config.trained_model_file_path,obj=best_model)

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test,predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)

