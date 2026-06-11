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
        self.label_encoders = {}

    def initiate_preprocessing(self, df):
        try:
            print("\n--- FASE 2: UNIVERSITY DATA PREPARATION ---")
            
            # 1. Cleaning
            df = df.drop_duplicates().reset_index(drop=True)

            # Drop student_id or similar if exists early
            id_cols = ['student_id', 'id', 'STUDENT_ID']
            for id_c in id_cols:
                if id_c in df.columns:
                    df = df.drop(id_c, axis=1)

            # 2. Auto-detect Target Column
            target_candidates = ['Target', 'grade', 'Grade', 'G3', 'status']
            target_col = None
            for cand in target_candidates:
                if cand in df.columns:
                    target_col = cand
                    break
            
            if not target_col:
                target_col = df.columns[-1]
                print(f"[WARN] Target tidak terdeteksi, menggunakan kolom terakhir: {target_col}")

            # 3. Handle specific University Target Logic (Multi-class Label Encoding)
            self.target_encoder = LabelEncoder()
            df[target_col] = self.target_encoder.fit_transform(df[target_col])
            print(f"[✓] Target Encoded: {dict(zip(self.target_encoder.classes_, self.target_encoder.transform(self.target_encoder.classes_)))}")
            
            # 4. Imputation
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()

            if num_cols:
                imp_num = SimpleImputer(strategy='median')
                df[num_cols] = imp_num.fit_transform(df[num_cols])

            if cat_cols:
                imp_cat = SimpleImputer(strategy='most_frequent')
                df[cat_cols] = imp_cat.fit_transform(df[cat_cols])

            # 5. Transformation (Label Encoding for Binary Categorical)
            for col in cat_cols:
                if df[col].nunique() == 2:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    self.label_encoders[col] = le
            
            # 6. Outlier Handling (Numerical only, exclude target)
            num_features = [c for c in num_cols if c != target_col]
            for col in num_features:
                Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                IQR = Q3 - Q1
                df[col] = np.clip(df[col], Q1 - 1.5*IQR, Q3 + 1.5*IQR)

            # 7. Splitting
            X = df.drop([target_col], axis=1)
            y = df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=(y if y.value_counts().min() >= 2 else None)
            )
            print(f"[✓] Data Splitting: Train {len(X_train)}, Test {len(X_test)}")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)

