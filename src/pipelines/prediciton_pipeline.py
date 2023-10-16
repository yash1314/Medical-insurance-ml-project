import sys
import os
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

import pandas as pd
import numpy as np

class PredictPipeline:

    def __init__(self):
        pass

    def predict(self,features):
        try:
            # processor and model loading 
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')
            
            #processor and model object creation 
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            
            # creating a array of input data from the use
            features = pd.DataFrame(data = np.array(features).reshape(1,6), columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
            
            data_scaled = preprocessor.transform(features)
            
            pred = model.predict(data_scaled)
            return pred
        
        except Exception as e:
            logging.info('Error occured in predict pipeline folder')
            raise CustomException(e, sys)
        