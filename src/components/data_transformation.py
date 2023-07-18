import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            numerical_columns = ['reading score', 'writing score']
            categorical_columns = ['gender','race/ethnicity','parental level of education','lunch','test preparation course']

            num_pipeline=Pipeline(steps=[('imputer',SimpleImputer(strategy='median')),
                                         ('scaler',StandardScaler())])
            
            cat_pipeline = Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),
                                           ('one_hot_encoder',OneHotEncoder()),
                                           ('scaler',StandardScaler())])
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer([("num_pipeline",num_pipeline,numerical_columns),
                                              ("cat_pipeline",cat_pipeline,categorical_columns)])
            
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)    
        
    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info('Reding dataset completed')

            preprocessing_obj = self.get_data_transformer_obj()
            logging.info('obtaining pre processing object')

            target_column_name="math score"
            numerical_columns = ["writing score", "reading score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("applying preprocessing object on train and test dataframe")

            input_feature_train_array = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessing_obj.transform(input_feature_test_df)
            


        except:
            pass

