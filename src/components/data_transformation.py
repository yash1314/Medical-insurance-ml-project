import os
import sys
from dataclasses import dataclass

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging

from src.utils import save_obj


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data transformation Initiated")

            numerical_cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
            # creating the standard scaler pipeline
            scaler_pipeline = Pipeline([('scaler', StandardScaler())])
            
            # creating the columntransformer using scaler pipeline
            # preprocessor = ColumnTransformer(transformers=[('scaler_pipeline', scaler_pipeline, numerical_cols)])

            return scaler_pipeline
        except Exception as e:
            logging.info("Error occured in data transformation")
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try : 
            # reading the train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'train dataframe head: \n{train_df.head().to_string()}')
            logging.info(f"test dataframe head: \n{test_df.head().to_string()}")

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            # Performing encodig to train and test dataframe
            train_df.replace({'region':{'southwest':0, "southeast":1, "northwest":2, "northeast":3}}, inplace=True)
            test_df.replace({'region':{'southwest':0, "southeast":1, "northwest":2, "northeast":3}}, inplace=True)

            train_df.replace({'sex': {'male':1, "female":0}, 'smoker': {'yes':1, 'no':0}},inplace = True)
            test_df.replace({'sex': {'male':1, "female":0}, 'smoker': {'yes':1, 'no':0}},inplace = True)

            # creating our final train and test split that is X_train, y_trian, X_test and y_test
            target_column_name = 'charges'
            

            input_feature_train_df = train_df.drop(columns= [target_column_name], axis = 1)
            input_feature_test_df = test_df.drop(columns= [target_column_name], axis = 1)

            target_feature_train_df = train_df[target_column_name]
            target_feature_test_df = test_df[target_column_name]

            # transformation using pickle object
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info('Applying preprocessor obj on train and test dataframe')

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_obj(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessing_obj
            )

            logging.info('Preprocess pickle is created and saved')
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            logging.info('error occured in the initiate_data_transformation')
            raise CustomException(e, sys)