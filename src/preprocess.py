import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler
from src.exception import CustomException
from src.logger import logging

class DataPreprocess:
    def __init__(self):
        pass

    def audit_preparation(self, df):
        try:
            logging.info("--- Fase 2: Data Preparation (7 Teknik) ---")
            
            # 1. Data Cleaning (Duplicates)
            before_dup = len(df)
            df = df.drop_duplicates()
            logging.info(f"Teknik 1 (Cleaning): Removed {before_dup - len(df)} duplicates")

            # 2. Imputation (Missing Values)
            # Numeric: Mean/Median, Categorical: Most Frequent
            num_cols = df.select_dtypes(include=['int64', 'float64']).columns
            cat_cols = df.select_dtypes(include=['object']).columns
            
            imputer_num = SimpleImputer(strategy='median')
            df[num_cols] = imputer_num.fit_transform(df[num_cols])
            
            imputer_cat = SimpleImputer(strategy='most_frequent')
            df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
            logging.info("Teknik 2 (Imputation): Completed using Median/Most Frequent")

            # 3. Data Transformation (Feature Engineering: Addition & Extraction)
            logging.info("--- Fase 3: Feature Engineering ---")
            # Addition
            df['Total_Score'] = df['G1'] + df['G2'] + df['G3']
            # Extraction
            df['Study_Efficiency'] = df['studytime'] / (df['freetime'] + 0.1) # avoid div by zero
            df['Fail_History'] = df['failures'].apply(lambda x: 1 if x > 0 else 0)
            # Target Class
            df['Final_Grade_Class'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
            logging.info("Teknik 3 (Transformation): Added Total_Score, Study_Efficiency, Fail_History")

            # 4. Outlier Handling (IQR Clipping)
            for col in ['absences', 'Total_Score']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df[col] = np.clip(df[col], lower, upper)
            logging.info("Teknik 4 (Outlier): Applied IQR Clipping on absences and Total_Score")

            # 5. Normalization/Scaling & Categorical Encoding
            # Drop G1, G2, G3 to avoid leakage, Drop failures as it is now Fail_History
            X = df.drop(['Final_Grade_Class', 'G3', 'G1', 'G2', 'failures'], axis=1)
            y = df['Final_Grade_Class']

            binary_cols = [col for col in X.columns if X[col].nunique() == 2 and X[col].dtype == 'object']
            multi_cols = [col for col in X.columns if X[col].nunique() > 2 and X[col].dtype == 'object']
            
            le = LabelEncoder()
            for col in binary_cols:
                X[col] = le.fit_transform(X[col])

            ct = ColumnTransformer(transformers=[
                ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), multi_cols)
            ], remainder='passthrough')

            X_transformed = ct.fit_transform(X)
            
            # 6. Scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_transformed)
            logging.info("Teknik 5 & 6 (Normalization & Encoding): Completed")

            # 7. Data Splitting & Imbalance Handling (SMOTE/RandomOverSampler)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Handling Imbalance on Training Set
            ros = RandomOverSampler(random_state=42)
            X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
            logging.info(f"Teknik 7 (Imbalance & Splitting): Resampled train set. New size: {len(y_train_res)}")

            # Reduction (Feature Names tracking)
            ohe_features = ct.named_transformers_['onehot'].get_feature_names_out(multi_cols)
            remaining_cols = [c for c in X.columns if c not in multi_cols]
            feature_names = list(ohe_features) + remaining_cols

            return X_train_res, X_test, y_train_res, y_test, feature_names

        except Exception as e:
            raise CustomException(e, sys)
