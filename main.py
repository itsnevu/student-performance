import os
import sys
import joblib
from src.data_loader import load_data
from src.preprocess import DataPreprocess
from src.feature_engineering import FeatureEngineer
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src.exception import CustomException
from src.logger import logging

def _configure_utf8_console():
    """Avoid Windows cp1252 print errors for unicode logs/symbols."""
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        # Non-fatal: keep default streams if reconfigure is unavailable.
        pass

def run_audit_pipeline():
    try:
        raw_data_path = "data/raw"
        output_dir = "outputs"
        os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)

        if not os.path.exists(raw_data_path):
            os.makedirs(raw_data_path)
            print(f"Folder {raw_data_path} dibuat. Silakan masukkan file CSV Anda.")
            return

        datasets = [f for f in os.listdir(raw_data_path) if f.endswith('.csv')]
        
        if not datasets:
            print("Tidak ada dataset .csv ditemukan di data/raw/")
            return

        print(f"Ditemukan {len(datasets)} dataset: {datasets}")

        for file_name in datasets:
            file_path = os.path.join(raw_data_path, file_name)
            logging.info(f"Memulai Audit untuk {file_name}")
            
            # 1. Phase 1: Data Loading & EDA
            df = load_data(file_path)
            
            # 2. Phase 2: Preprocessing
            preprocessor = DataPreprocess()
            X_train, X_test, y_train, y_test = preprocessor.initiate_preprocessing(df)
            
            # 3. Phase 3: Feature Engineering
            fe = FeatureEngineer()
            # [M1] Return preprocessors for persistence
            X_train_final, X_test_final, pre_objs = fe.apply_feature_engineering(X_train, X_test, y_train)
            
            # Save preprocessors for this specific run
            pp_path = os.path.join(output_dir, "models", "preprocessors.pkl")
            joblib.dump(pre_objs, pp_path)
            print(f"[✓] Preprocessor Objs (ct, scaler, pca) Tersimpan di {pp_path}")
            
            # 4. Phase 4: Modeling
            trainer = ModelTrainer()
            best_models = trainer.initiate_model_trainer(X_train_final, y_train)
            
            # 5. Phase 5: Evaluation
            evaluator = ModelEvaluator()
            evaluator.evaluate_all(
                best_models, X_test_final, y_test, 
                X_train_final, y_train, dataset_name=file_name
            )
            
            print(f"\n✅ AUDIT BERHASIL UNTUK: {file_name}")
            print("-" * 70)

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    _configure_utf8_console()
    run_audit_pipeline()
