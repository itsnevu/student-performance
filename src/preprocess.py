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
            print("\n--- FASE 2: DATA PREPARATION (7 Teknik) ---")
            
            # 1. Cleaning
            before = len(df)
            df = df.drop_duplicates().reset_index(drop=True)
            print(f"[✓] Step 1: Cleaning ({before - len(df)} duplikat dibuang)")

            # 2. Imputation (Fixed: Numerical & Categorical)
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()

            if num_cols:
                imp_num = SimpleImputer(strategy='median')
                df[num_cols] = imp_num.fit_transform(df[num_cols])

            if cat_cols:
                imp_cat = SimpleImputer(strategy='most_frequent')
                df[cat_cols] = imp_cat.fit_transform(df[cat_cols])
            print(f"[✓] Step 2: Imputation Selesai")

            # 3. Transformation (C7: LabelEncoder per kolom)
            binary_cols = [c for c in df.columns if df[c].nunique() == 2 and df[c].dtype == 'object']
            for col in binary_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
            print(f"[✓] Step 3: Transformation ({len(binary_cols)} kolom binary di-encode)")

            # 4. Outlier Handling (Dinamis IQR)
            num_cols_now = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'G3']
            total_clipped = 0
            for col in num_cols_now:
                Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                total_clipped += ((df[col] < lower) | (df[col] > upper)).sum()
                df[col] = np.clip(df[col], lower, upper)
            print(f"[✓] Step 4: Outlier Handling ({total_clipped} nilai di-clip)")

            # 7. Splitting (C3: Guard for small classes)
            target_col = 'G3' if 'G3' in df.columns else (df.columns[-1] if 'Final_Grade_Class' not in df.columns else 'Final_Grade_Class')
            X = df.drop([target_col], axis=1)
            y = df[target_col].apply(lambda x: 1 if x >= 10 else 0) if target_col == 'G3' else df[target_col]

            min_class = y.value_counts().min()
            stratify_param = y if min_class >= 2 else None
            if stratify_param is None:
                print("[WARN] Stratified split dinonaktifkan (kelas terlalu kecil)")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify_param
            )
            print(f"[✓] Step 7: Data Splitting Selesai (Train: {len(X_train)}, Test: {len(X_test)})")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)
