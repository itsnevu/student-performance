import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import sys

def load_data(file_path):
    try:
        logging.info("Loading dataset from path: {0}".format(file_path))
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
        
        df = pd.read_csv(file_path, sep=';')
        
        # Ensure G1, G2, G3 are numeric (sometimes quoted in CSV)
        for col in ['G1', 'G2', 'G3']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logging.info("Dataset loaded successfully with shape: {0}".format(df.shape))
        return df
    except Exception as e:
        raise CustomException(e, sys)
