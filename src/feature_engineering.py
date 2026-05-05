import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from src.exception import CustomException
from src.logger import logging

# [C4] Guard for OHE compatibility
try:
    from sklearn.preprocessing import OneHotEncoder
    OneHotEncoder(sparse_output=False)
    OHE_KWARGS = {"sparse_output": False}
except TypeError:
    OHE_KWARGS = {"sparse": False}

class FeatureEngineer:
    def __init__(self, n_components=5):
        self.n_components = n_components

    def _apply_features(self, data):
        """
        Helper: Addition + Extraction + Reduction.
        Independent logic to prevent leakage.
        """
        data = data.copy()
        if 'G1' in data.columns and 'G2' in data.columns:
            data['Midterm_Avg'] = (data['G1'] + data['G2']) / 2
        if 'studytime' in data.columns and 'freetime' in data.columns:
            data['Study_Efficiency'] = data['studytime'] / (data['freetime'] + 0.1)
        # M10: Keep 'failures' as it is highly predictive
        return data.drop(columns=[c for c in ['G1', 'G2'] if c in data.columns])

    def apply_feature_engineering(self, X_train, X_test, y_train):
        """
        Fase 3: University Feature Engineering (Generic OHE + Scaling + Selection + PCA).
        """
        try:
            print("\n--- FASE 3: UNIVERSITY FEATURE ENGINEERING ---")
            
            # Step 1: Normalization (OHE for categorical + Scaling)
            cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
            
            ct = ColumnTransformer([
                ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', **OHE_KWARGS), cat_cols)
            ], remainder='passthrough')

            # Fit on train, transform both
            X_train_t = ct.fit_transform(X_train)
            X_test_t  = ct.transform(X_test)
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train_t)
            X_test_s  = scaler.transform(X_test_t)
            
            # Step 2: Selection (Top features - TRANSPARENT)
            k_best = min(30, X_train_s.shape[1])
            selector = SelectKBest(f_classif, k=k_best)
            X_train_final = selector.fit_transform(X_train_s, y_train)
            X_test_final  = selector.transform(X_test_s)
            
            print(f"[✓] Feature Selection Selesai ({k_best} Fitur Utama Terpilih)")

            preprocessors = {
                'column_transformer': ct,
                'scaler': scaler,
                'selector': selector
            }


            return X_train_final, X_test_final, preprocessors

        except Exception as e:
            raise CustomException(e, sys)

