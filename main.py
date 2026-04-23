import sys
import os
import pandas as pd
from src.data_loader import load_data
from src.preprocess import DataPreprocess
from src.train import ModelTrainer
from src.predict import ModelEvaluator
from src.logger import logging
from src.exception import CustomException

def run_audit_pipeline():
    try:
        logging.info("==========================================")
        logging.info("      STARTING TOTAL AUDIT ML PIPELINE    ")
        logging.info("==========================================")
        
        raw_data_dir = os.path.join("data", "raw")
        all_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
        
        if not all_files:
            logging.error("No CSV files found in data/raw/. Audit aborted.")
            return

        print(f"Ditemukan {len(all_files)} dataset: {all_files}")
        
        for file_name in all_files:
            logging.info(f"--- MENGODIT DATASET: {file_name} ---")
            print(f"\n> Sedang mengaudit: {file_name}...")
            
            data_path = os.path.join(raw_data_dir, file_name)
            df = load_data(data_path)
            
            if df is not None:
                # Validasi minimal: pastikan ada kolom target (G3 atau G3-like)
                # Jika namanya beda di Kaggle, kita coba ganti secara dinamis
                if 'G3' not in df.columns and 'GRADE' in df.columns:
                    df.rename(columns={'GRADE': 'G3'}, inplace=True)
                
                if 'G3' in df.columns:
                    # Phase 2 & 3: Data Preparation (7 Techniques) & Feature Engineering
                    preprocessor = DataPreprocess()
                    X_train, X_test, y_train, y_test, feature_names = preprocessor.audit_preparation(df)
                    
                    # Phase 4: Modeling (10 Models)
                    trainer = ModelTrainer()
                    trained_models, cv_results = trainer.initiate_model_trainer(X_train, y_train)
                    
                    # Phase 5: Evaluation
                    evaluator = ModelEvaluator()
                    performance_table = evaluator.evaluate_all(trained_models, X_test, y_test)
                    
                    logging.info(f"Audit untuk {file_name} Selesai!")
                    print(f"Selesai! Model terbaik untuk {file_name} adalah: {performance_table.iloc[0]['Model']}")
                else:
                    logging.warning(f"File {file_name} tidak memiliki kolom 'G3' atau 'GRADE'. Melewati file ini.")
            
        print("\nAudit Total Selesai untuk semua dataset! Cek folder 'outputs/' dan 'artifacts/logs/'.")

    except Exception as e:
        logging.error(f"Audit Pipeline Failed: {str(e)}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    run_audit_pipeline()
