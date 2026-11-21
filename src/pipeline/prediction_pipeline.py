import sys
import os
import pandas as pd
import numpy as np
from src.utils import load_object
from src.exception import CustomException

class PredictPipeline:
    def __init__(self):
        self.model = load_object(os.path.join('Artifacts', 'model.pkl'))
        self.preprocessor = load_object(os.path.join('Artifacts', 'preprocessor.pkl'))

    def predict(self,features):
        try:
            scaled_data = self.preprocessor.transform(features)
            return self.model.predict(scaled_data)
        except Exception as e:
            raise CustomException(e, sys)
class CustomData:
    def __init__(self,gender:str,race:str,parental_level:str,lunch:str,test_preparation_course:str,
                reading_score:float,writing_score:float):
        self.gender=gender
        self.race=race
        self.parental_level=parental_level
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score
    
    
    def get_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race],
                "parental_level_of_education": [self.parental_level],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
