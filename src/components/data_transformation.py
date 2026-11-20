import sys
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import _num_features
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path=os.path.join('Artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformer(self):
        try:
            num_features=['reading_score', 'writing_score']
            cat_features=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
                ]

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("Standard_Scaling",StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("OneHotEncoding",OneHotEncoder(sparse_output=False)),
                    ("Standard_Scaling",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical Feature {num_features}")
            logging.info(f"Categorical Feature {cat_features}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_features),
                    ("cat_pipeline",cat_pipeline,cat_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def transform_initiate(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading the Training and Testing Dataset")

            preprocessing_obj=self.get_data_transformer()
            target_feature="math_score"
            
            input_train_df=train_df.drop(columns=[target_feature])
            target_train_df=train_df[target_feature]
            
            input_test_df=test_df.drop(columns=[target_feature])
            target_test_df=test_df[target_feature]

            input_train_df_arr=preprocessing_obj.fit_transform(input_train_df)
            input_test_df_arr=preprocessing_obj.transform(input_test_df)
            
            logging.info("Preprocessing Applied on Training and Testing Dataset")

            train_arr=np.c_[input_train_df_arr,np.array(target_train_df)]
            test_arr=np.c_[input_test_df_arr,np.array(target_test_df)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessing_obj
                )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path
            )
        except Exception as e:
            raise CustomException(e,sys)