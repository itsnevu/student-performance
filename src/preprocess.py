import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging

class DataPreprocess:
    def __init__(self):
        pass

    def feature_engineering(self, df):
        try:
            logging.info("Starting feature engineering")
            df['Total_Score'] = df['G1'] + df['G2'] + df['G3']
            df['Fail_History'] = df['failures'].apply(lambda x: 1 if x > 0 else 0)
            df['Study_Efficiency'] = df['studytime'] / df['freetime']
            df['Final_Grade_Class'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
            logging.info("Feature engineering completed")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def split_and_transform(self, df):
        try:
            logging.info("Cleaning data and splitting into train/test sets")
            
            # Check for missing values
            if df.isnull().sum().sum() > 0:
                df = df.dropna()
                logging.info("Dropped missing values")

            X = df.drop(['Final_Grade_Class', 'G3'], axis=1)
            y = df['Final_Grade_Class']

            binary_cols = [col for col in X.columns if X[col].nunique() == 2 and X[col].dtype == 'object']
            multi_cols = [col for col in X.columns if X[col].nunique() > 2 and X[col].dtype == 'object']
            
            # Label Encoding
            le = LabelEncoder()
            for col in binary_cols:
                X[col] = le.fit_transform(X[col])

            # Column Transformation (OneHot + Scaling)
            # Scaling will be done separately for LogReg in the train pipeline or here
            # For simplicity and model comparison, we'll return both scaled and raw X
            
            ct = ColumnTransformer(transformers=[
                ('onehot', OneHotEncoder(drop='first'), multi_cols)
            ], remainder='passthrough')

            X_transformed = ct.fit_transform(X)
            
            ohe_features = ct.named_transformers_['onehot'].get_feature_names_out(multi_cols)
            feature_names = list(ohe_features) + [c for c in X.columns if c not in multi_cols]

            X_train, X_test, y_train, y_test = train_test_split(
                X_transformed, y, test_size=0.2, random_state=42, stratify=y
            )

            # Standard Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            logging.info("Data transformation and splitting completed")
            return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, feature_names
            
        except Exception as e:
            raise CustomException(e, sys)
