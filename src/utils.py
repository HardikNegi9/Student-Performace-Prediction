import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score

from src.exception import CustomError

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file:
            dill.dump(obj, file)

            
    except Exception as e:
        raise CustomError(e, sys)
    
def evaluate_models (x_train, y_train, x_test, y_test, models):
    try:
        model_report = {}

        for model_name, model in models.items():

            model.fit(x_train, y_train)

            y_pred_test = model.predict(x_test)

            test_model_score = r2_score(y_test, y_pred_test)

            model_report[model_name] = test_model_score

        return model_report
    
    except Exception as e:
        raise CustomError(e, sys)