import sys
import os
from src.data_loader import load_data
from src.preprocess import DataPreprocess
from src.feature_engineering import FeatureEngineer
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src.logger import logging
from src.exception import CustomException

def run_audit_pipeline():
    try:
        logging.info("==========================================")
        logging.info("      STARTING IMPROVED AUDIT PIPELINE    ")
        logging.info("==========================================")
        
        raw_data_dir = os.path.join("data", "raw")
        all_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
        
        for file_name in all_files:
            logging.info(f"--- MENGODIT DATASET: {file_name} ---")
            data_path = os.path.join(raw_data_dir, file_name)
            df = load_data(data_path)
            
            if df is not None:
                # Rename target if needed
                if 'G3' not in df.columns and 'GRADE' in df.columns:
                    df.rename(columns={'GRADE': 'G3'}, inplace=True)
                
                if 'G3' in df.columns or 'Final_Grade_Class' in df.columns:
                    # 1. Phase 2: Preprocessing (7 Steps)
                    preprocessor = DataPreprocess()
                    X_train, X_test, y_train, y_test = preprocessor.initiate_preprocessing(df)
                    
                    # 2. Phase 3: Feature Engineering (5 Techniques)
                    engineer = FeatureEngineer(n_components=5)
                    X_train_final, X_test_final = engineer.apply_feature_engineering(X_train, X_test, y_train)
                    
                    # 3. Phase 4: Modeling (Tuning & SMOTE)
                    trainer = ModelTrainer()
                    best_models = trainer.initiate_model_trainer(X_train_final, y_train)
                    
                    # 4. Phase 5: Evaluation
                    evaluator = ModelEvaluator()
                    performance_table = evaluator.evaluate_all(best_models, X_test_final, y_test)
                    
                    print(f"Audit Selesai untuk {file_name}. Best Model: {performance_table.iloc[0]['Model']}")
                else:
                    logging.warning(f"File {file_name} tidak memiliki target. Skip.")

    except Exception as e:
        logging.error(f"Audit Pipeline Failed: {str(e)}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    run_audit_pipeline()
