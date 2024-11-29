import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomError

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file:
            dill.dump(obj, file)

            
    except Exception as e:
        raise CustomError(e, sys)
    
def evaluate_models(x_train, y_train, x_test, y_test, models, params):
    try:
        model_report = {}

        for model_name, model in models.items():
            try:
                # Get parameters for this specific model
                model_params = params[model_name]

                # Perform GridSearchCV
                grid_search = GridSearchCV(model, model_params, cv=3, scoring='r2', n_jobs=-1)
                grid_search.fit(x_train, y_train)

                # Use the best model found by GridSearchCV
                best_model = grid_search.best_estimator_

                # Make predictions using the best model
                y_test_pred = best_model.predict(x_test)

                # Calculate R2 scores
                test_model_score = r2_score(y_test, y_test_pred)

                # Store the test model score in the report
                model_report[model_name] = test_model_score

                # Optional: print best parameters found
                print(f"Best parameters for {model_name}: {grid_search.best_params_}")

            except Exception as e:
                model_report[model_name] = np.nan
                print(f"Error in {model_name}: {str(e)}")

        return model_report
    
    except Exception as e:
        raise CustomError(e, sys)
    
    except Exception as e:
        raise CustomError(e, sys)