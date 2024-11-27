import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomError
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTranformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTranformation:
    def __init__(self):
        self.data_tranformation_config = DataTranformationConfig()
    
    def get_data_transformer_object(self):  # function responsible data transformation
        try:
            numeric_features = ['writing_score', 'reading_score']
            categorical_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder()),
                ]
            )

            logging.info(f"Numerical Features: {numeric_features}")
            logging.info(f"Categorical Features: {categorical_features}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("numeric pipeline", num_pipeline, numeric_features),
                    ("categorical pipeline", cat_pipeline, categorical_features),
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomError(e, sys)
        
    def inititate_data_tranformation(self, tain_path, test_path):
        try:
            train_df = pd.read_csv(tain_path)
            test_df = pd.read_csv(test_path)

            logging.info(f"Train Data Shape: {train_df.shape}")
            logging.info(f"Test Data Shape: {test_df.shape}")

            logging.info("Obtaining Preprocessor Object")
            preprocessor_obj = self.get_data_transformer_object()

            target_col_name = "math_score"
            numeric_col = ['writing_score', 'reading_score']

            input_features_train_df = train_df.drop(target_col_name, axis=1)
            target_feature_train_df = train_df[target_col_name]

            input_features_test_df = test_df.drop(target_col_name, axis=1)
            target_feature_test_df = test_df[target_col_name]


            logging.info("Applying Preprocessor Object on Train Data")


            input_feature_train_arr = preprocessor_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_features_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


            save_object(
                file_path=self.data_tranformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj,
            )

            logging.info("Preprocessor Object has been saved successfully")

            logging.info("Data Transformation has been completed successfully")

            return (
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_obj_file_path,
            )
        

        except Exception as e:
            raise CustomError(e, sys)