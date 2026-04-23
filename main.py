import sys
import os
from src.data_loader import load_data
from src.preprocess import DataPreprocess
from src.train import ModelTrainer
from src.predict import ModelEvaluator
from src.logger import logging
from src.exception import CustomException

def main():
    try:
        logging.info("Starting the ML Pipeline")
        
        # Path to data
        data_path = os.path.join("data", "raw", "student-mat.csv")
        
        # 1. Load Data
        df = load_data(data_path)
        
        if df is not None:
            # 2. Preprocessing
            preprocessor = DataPreprocess()
            df = preprocessor.feature_engineering(df)
            X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, features = preprocessor.split_and_transform(df)
            
            # 3. Training
            trainer = ModelTrainer()
            lr, dt = trainer.initiate_model_trainer(X_train_sc, X_train, y_train)
            
            # 4. Evaluation
            evaluator = ModelEvaluator()
            evaluator.evaluate((lr, dt), (X_test_sc, X_test, y_test), features)
            
            logging.info("Pipeline executed successfully!")
        else:
            logging.warning("No data found. Please place 'student-mat.csv' in 'data/raw/'.")

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
