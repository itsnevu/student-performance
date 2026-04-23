import sys
import joblib
import pandas as pd
from src.exception import CustomException

class Predictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, features):
        try:
            return self.model.predict(features)
        except Exception as e:
            raise CustomException(e, sys)
