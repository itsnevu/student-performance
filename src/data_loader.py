import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.exception import CustomException
from src.logger import logging

def load_data(file_path: str) -> pd.DataFrame:
    """
    Fase 1: Load University Student Dataset with Generic EDA.
    """
    try:
        logging.info(f"Loading data from: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

        # Auto-detect separator
        try:
            df = pd.read_csv(file_path, sep=';')
            if len(df.columns) < 2:
                df = pd.read_csv(file_path, sep=',')
        except Exception:
            df = pd.read_csv(file_path)
            
        # Generic Cleaning: Remove whitespace from column names
        df.columns = [c.strip() for c in df.columns]

        # Force conversion to numeric for potential numeric columns (messy data handling)
        for col in df.columns:
            if col.lower() not in ['target', 'status', 'grade']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        print("\n" + "="*60)
        print(f"  AUDIT REPORT FASE 1 (UNIVERSITY DATA): {os.path.basename(file_path)}")
        print("="*60)
        print(f"[*] Shape Dataset    : {df.shape[0]} baris x {df.shape[1]} kolom")
        print(f"[*] Duplikat         : {df.duplicated().sum()}")
        
        missing = df.isnull().sum()
        missing_count = missing.sum()
        print(f"[*] Missing Values   : {missing_count}")

        # Auto-detect Target Column
        target_candidates = ['Target', 'grade', 'Grade', 'G3', 'status']
        target_col = None
        for cand in target_candidates:
            if cand in df.columns:
                target_col = cand
                break
        
        if target_col:
            dist = df[target_col].value_counts()
            print(f"[*] Distribusi Target ({target_col}):")
            print(dist.to_string())
            
            if len(dist) > 0:
                ratio = dist.min() / dist.max()
                status = '⚠ Imbalanced' if ratio < 0.8 else '✓ Balanced'
                print(f"[*] Imbalance Ratio  : {ratio:.2f} | {status}")

            # --- [SCIENCE ADDITION] Correlation Analysis ---
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                corr = numeric_df.corr()
                plt.figure(figsize=(12, 8))
                sns.heatmap(corr, annot=False, cmap='coolwarm', fmt='.2f')
                plt.title(f"Correlation Analysis — {os.path.basename(file_path)}")
                
                # Save plot to outputs
                os.makedirs("outputs/eda", exist_ok=True)
                plt.savefig(f"outputs/eda/correlation_{os.path.basename(file_path)}.png")
                plt.close()
                print(f"[*] Science Audit: Korelasi fitur disimpan di outputs/eda/")
        
        print("-" * 60)
        return df

    except Exception as e:
        raise CustomException(e, sys)

