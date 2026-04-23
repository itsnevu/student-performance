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
        
        # Phase 1: Load Dataset
        data_path = os.path.join("data", "raw", "student-mat.csv")
        df = load_data(data_path)
        
        if df is not None:
            # Phase 2 & 3: Data Preparation (7 Techniques) & Feature Engineering
            preprocessor = DataPreprocess()
            X_train, X_test, y_train, y_test, feature_names = preprocessor.audit_preparation(df)
            
            # Phase 4: Modeling (10 Models)
            trainer = ModelTrainer()
            trained_models, cv_results = trainer.initiate_model_trainer(X_train, y_train)
            
            # Phase 5: Evaluation
            evaluator = ModelEvaluator()
            performance_table = evaluator.evaluate_all(trained_models, X_test, y_test)
            
            logging.info("Audit Pipeline Completed Successfully!")
            print("\nAudit Selesai! Cek folder 'outputs/' untuk hasil perbandingan model.")
            print("Model terbaik adalah:", performance_table.iloc[0]['Model'])
            
        else:
            logging.error("Data not found. Audit aborted.")

    except Exception as e:
        logging.error(f"Audit Pipeline Failed: {str(e)}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    run_audit_pipeline()
