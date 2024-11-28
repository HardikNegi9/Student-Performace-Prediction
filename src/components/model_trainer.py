import os
import sys
from dataclasses import dataclass

from xgboost import XGBRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

from src.exception import CustomError
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    model_obj_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__ (self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self, train_arr, test_arr):
        try: 
            logging.info("Model Training has been initiated")
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1], # exclude last column
                train_arr[:, -1], # last column
                test_arr[:, :-1],
                test_arr[:, -1]
            )
        
            models = {
                "Linear Regression": LinearRegression(),
                "Support Vector Machine": SVR(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBoost": XGBRegressor(),
                "KNN": KNeighborsRegressor()
            }


            model_report: dict = evaluate_models(
                x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, models = models
                )
            
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                logging.error("Model Score is less than threshold")
                raise CustomError("Model Score is less than threshold", sys)
            
            logging.info(f"Best Model: {best_model_name} with R2 Score: {best_model_score}")

            save_object(
                file_path = self.model_trainer_config.model_obj_file_path,
                obj = best_model
            )

            predicted = best_model.predict(x_test)
            r2 = r2_score(y_test, predicted)
            return r2

        
        except Exception as e:
            raise CustomError(e, sys)

