#It will do feature engineering , data cleaning and converting the data etc..
#sys is mainly used for error handling and logging
#A sparse matrix is a special way of storing data where most of the values are zero. Instead of saving every value (including all the zeros), it only saves the locations and values of the non-zero elements. This saves memory and speeds up calculations when you have a lot of zeros.
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer #helps you preprocess different types of columns in your dataset with different techniques, all in one go.
from sklearn.impute import SimpleImputer #used to fill in missing values in your dataset
from sklearn.pipeline import Pipeline #used to streamline the process of applying multiple transformations to your data
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.utils import save_object #used to save the preprocessor object to a file
from src.exception import CustomException
from src.logger import logging
import os


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl') # is a saved version of your preprocessing pipeline, allowing you to load and use it later for new data or in production.

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        this function is responsible for data transformation
        """
        try:
            numerical_columns= ['writing_score','reading_score']
            categorical_columns=["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")), #fill the missing values with the median of the column
                    ("scaler",StandardScaler())#standardize the data
                ]
            )
            #target_encoding : is a technique used to encode categorical variables by replacing each category with the mean of the target variable for that category
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False)),
                ]
            )
            logging.info(f"Numerical columns :{numerical_columns}")
            logging.info(f"Categorical columns :{categorical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj= self.get_data_transformer_object()
            target_column_name="math_score"
            numerical_columns=['writing_score','reading_score']

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
