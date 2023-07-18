import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass  #available python 3.9

@dataclass
class DataIngesitionConfig:       #if we only defining variables dataclass will be the best choice
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngesition:
    def __init__(self):
        self.ingestion_config = DataIngesitionConfig

    def initiate_data_ingestion(self):   #if data reading from DB, the code should be write here
        logging.info('Entered data ingestion method')
        try:
            df = pd.read_csv('notebook\data\StudentsPerformance.csv')
            logging.info('Read the data and saving it in dataframe')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            logging.info('train test split initiated')

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('ingestion of the data completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj = DataIngesition()
    obj.initiate_data_ingestion()


