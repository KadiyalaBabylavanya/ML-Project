#data ingestion is used to gets the data, organizes it and prepare it for use

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from  sklearn.model_selection import train_test_split
from dataclasses import dataclass #creating class variables

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig



@dataclass
#decorator and that makes it easier to create classes for storing data by automatically generating common methods for you(init,repr,eq,etc.)
class DataIngestionConfig:
    train_data_path : str=os.path.join('artifacts',"train.csv")
    test_data_path:str=os.path.join('artifacts',"test.csv")
    raw_data_oath:str=os.path.join('artifacts',"data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion method or component")

        try:
            df=pd.read_csv('notebook/data/stud.csv')
            #it will chnage as per our data coming from the source(api,database,mangodb, etc)
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_oath,index=False,header=True)
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            #header=True ensures the CSV file includes column names at the top
            #index=False ensures that the index column is not included in the CSV file

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) 
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
            pass

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
