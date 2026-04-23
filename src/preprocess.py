import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from src.exception import CustomException
from src.logger import logging

class DataPreprocess:
    def __init__(self):
        pass

    def initiate_preprocessing(self, df):
        try:
            logging.info("--- Fase 2: Data Preparation (7 Teknik) ---")
            
            # 1. Data Cleaning (Duplicates)
            df = df.drop_duplicates()
            logging.info("Step 1: Duplicates removed")

            # 2. Imputation (Missing Values)
            num_cols = df.select_dtypes(include=['int64', 'float64']).columns
            cat_cols = df.select_dtypes(include=['object']).columns
            
            imputer_num = SimpleImputer(strategy='median')
            df[num_cols] = imputer_num.fit_transform(df[num_cols])
            
            imputer_cat = SimpleImputer(strategy='most_frequent')
            df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
            logging.info("Step 2: Imputation completed")

            # 3. Data Transformation (Basic Categorical Encoding)
            # Binary encoding for 2-unique objects
            binary_cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtype == 'object']
            le = LabelEncoder()
            for col in binary_cols:
                df[col] = le.fit_transform(df[col])
            logging.info("Step 3: Binary transformation completed")

            # 4. Outlier Handling (IQR Clipping)
            # We only clip highly skewed numeric columns like absences
            if 'absences' in df.columns:
                Q1 = df['absences'].quantile(0.25)
                Q3 = df['absences'].quantile(0.75)
                IQR = Q3 - Q1
                df['absences'] = np.clip(df['absences'], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
            logging.info("Step 4: Outlier handling (IQR) completed")

            # Note: Normalization & Scaling (Step 5 & 6) will be done in Feature Engineering 
            # or Pipeline to avoid leakage.
            
            # 7. Data Splitting (Before Advanced Engineering & Resampling)
            # Define target (default G3 if not renamed)
            target_col = 'G3' if 'G3' in df.columns else 'Final_Grade_Class'
            X = df.drop([target_col], axis=1)
            y = df[target_col].apply(lambda x: 1 if x >= 10 else 0) if target_col == 'G3' else df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            logging.info("Step 7: Data splitting (80:20) completed")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)
