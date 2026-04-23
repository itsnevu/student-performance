import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging

def load_data(file_path: str) -> pd.DataFrame:
    """
    [C1-m] Fase 1: Load Dataset with Error Handling & EDA.
    """
    try:
        logging.info(f"Loading data from: {file_path}")
        
        # [C1] FileNotFoundError guard
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File tidak ditemukan di path: {file_path}")

        # Auto-detect separator
        try:
            df = pd.read_csv(file_path, sep=';')
            if len(df.columns) < 2:
                df = pd.read_csv(file_path, sep=',')
        except pd.errors.ParserError:
            df = pd.read_csv(file_path)
            
        # Clean numeric columns
        for col in ['G1', 'G2', 'G3']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- VERBOSE EDA REPORT (m1) ---
        print("\n" + "="*60)
        print(f"  REPORT FASE 1: {os.path.basename(file_path)}")
        print("="*60)
        print(f"[*] Shape Dataset    : {df.shape[0]} baris x {df.shape[1]} kolom")
        print(f"[*] Duplikat         : {df.duplicated().sum()}")
        
        missing = df.isnull().sum()
        missing_count = missing.sum()
        print(f"[*] Missing Values   : {missing_count}")
        if missing_count > 0:
            print(missing[missing > 0].to_string())

        if 'G3' in df.columns:
            target = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
            dist = target.value_counts()
            ratio = dist.min() / dist.max()
            print(f"[*] Distribusi Target (0: Gagal, 1: Lulus):")
            print(dist.to_string())
            status = '⚠ Imbalanced' if ratio < 0.8 else '✓ Balanced'
            print(f"[*] Imbalance Ratio  : {ratio:.2f} | {status}")
        
        print("-" * 60)

        return df
    except Exception as e:
        raise CustomException(e, sys)
