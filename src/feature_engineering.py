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
        [M1] Returns X_train_final, X_test_final AND preprocessors dict.
        """
        try:
            print("\n--- FASE 3: FEATURE ENGINEERING (5 Teknik) ---")
            
            # Step 1-3: Addition, Extraction, Reduction
            X_train_f = self._apply_features(X_train)
            X_test_f  = self._apply_features(X_test)
            print("[✓] Step 1-3: Add/Extract/Reduce Selesai (failures DIPERTAHANKAN)")

            # Step 4: Normalization (OHE + Scaling)
            multi_cat_cols = X_train_f.select_dtypes(include=['object']).columns.tolist()
            
            ct = ColumnTransformer([
                ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', **OHE_KWARGS), multi_cat_cols)
            ], remainder='passthrough')

            X_train_t = ct.fit_transform(X_train_f)
            X_test_t  = ct.transform(X_test_f)
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train_t)
            X_test_s  = scaler.transform(X_test_t)
            print(f"[✓] Step 4: Normalization ({X_train_s.shape[1]} fitur setelah OHE+Scaling)")

            # Step 5: Selection + PCA
            k_best = min(20, X_train_s.shape[1])
            selector = SelectKBest(f_classif, k=k_best)
            X_train_sel = selector.fit_transform(X_train_s, y_train)
            X_test_sel  = selector.transform(X_test_s)
            
            n_comp = min(self.n_components, X_train_sel.shape[1])
            pca = PCA(n_components=n_comp, random_state=42)
            X_train_final = pca.fit_transform(X_train_sel)
            X_test_final  = pca.transform(X_test_sel)
            
            explained = pca.explained_variance_ratio_.sum() * 100
            print(f"[✓] Step 5: Selection & PCA Selesai ({n_comp} komponen | Explained: {explained:.1f}%)")

            # [M1] Preprocessor dictionary for persistence
            preprocessors = {
                'column_transformer': ct,
                'scaler': scaler,
                'selector': selector,
                'pca': pca
            }

            return X_train_final, X_test_final, preprocessors

        except Exception as e:
            raise CustomException(e, sys)
