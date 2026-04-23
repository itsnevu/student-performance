import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from src.exception import CustomException
from src.logger import logging

class FeatureEngineer:
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components)
        self.selector = SelectKBest(score_func=f_classif, k='all')

    def apply_feature_engineering(self, X_train, X_test, y_train):
        try:
            logging.info("--- Fase 3: Feature Engineering (5 Teknik) ---")
            
            # 1. Feature Addition & 2. Extraction
            for df in [X_train, X_test]:
                if 'G1' in df.columns and 'G2' in df.columns:
                    df['Midterm_Avg'] = (df['G1'] + df['G2']) / 2
                df['Study_Efficiency'] = df['studytime'] / (df['freetime'] + 0.1)
                df['Fail_History'] = df['failures'].apply(lambda x: 1 if x > 0 else 0)

            # 3. Feature Reduction (Drop irrelevant/leakage columns)
            drop_cols = ['G1', 'G2', 'failures'] # Dropped because extracted/added
            X_train = X_train.drop([c for c in drop_cols if c in X_train.columns], axis=1)
            X_test = X_test.drop([c for c in drop_cols if c in X_test.columns], axis=1)
            logging.info("Step 1-3: Addition, Extraction, Reduction completed")

            # Handling multi-label categorical before scaling (OneHot)
            multi_cols = X_train.select_dtypes(include=['object']).columns
            
            ct = ColumnTransformer(transformers=[
                ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), multi_cols)
            ], remainder='passthrough')

            X_train_transformed = ct.fit_transform(X_train)
            X_test_transformed = ct.transform(X_test)
            
            # 4. Normalization / Feature Scaling (Fase 2 - Step 5 & 6)
            # This is where we ensure no 'GP' strings reach the scaler
            X_train_scaled = self.scaler.fit_transform(X_train_transformed)
            X_test_scaled = self.scaler.transform(X_test_transformed)
            logging.info("Step 4 (Scaling): Standardization completed")

            # 5. Feature Selection (SelectKBest)
            X_train_selected = self.selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = self.selector.transform(X_test_scaled)
            
            # 6. PCA (Dimensionality Reduction)
            # Only apply if features > n_components
            if X_train_selected.shape[1] > self.n_components:
                X_train_final = self.pca.fit_transform(X_train_selected)
                X_test_final = self.pca.transform(X_test_selected)
                logging.info(f"Step 5 (PCA): Reduced to {self.n_components} components")
            else:
                X_train_final = X_train_selected
                X_test_final = X_test_selected

            return X_train_final, X_test_final

        except Exception as e:
            raise CustomException(e, sys)
