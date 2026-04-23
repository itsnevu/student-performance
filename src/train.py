import sys
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.exception import CustomException
from src.logger import logging

class ModelTrainer:
    def __init__(self):
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)

    def initiate_model_trainer(self, X_train_scaled, X_train_raw, y_train):
        try:
            logging.info("Starting model training")

            # 1. Logistic Regression
            lr_model = LogisticRegression(max_iter=2000, solver='lbfgs', random_state=42)
            lr_model.fit(X_train_scaled, y_train)
            
            # 2. Decision Tree
            dt_model = DecisionTreeClassifier(
                max_depth=5, 
                min_samples_split=10, 
                criterion='entropy', 
                random_state=42
            )
            dt_model.fit(X_train_raw, y_train)

            # Cross-Validation
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            lr_cv = cross_val_score(lr_model, X_train_scaled, y_train, cv=skf, scoring='accuracy')
            dt_cv = cross_val_score(dt_model, X_train_raw, y_train, cv=skf, scoring='accuracy')

            logging.info(f"Logistic Regression CV Score: {lr_cv.mean()}")
            logging.info(f"Decision Tree CV Score: {dt_cv.mean()}")

            # Save Models
            joblib.dump(lr_model, os.path.join(self.models_dir, "logistic_regression.pkl"))
            joblib.dump(dt_model, os.path.join(self.models_dir, "decision_tree.pkl"))

            logging.info("Models saved successfully in 'models/' directory")
            return lr_model, dt_model

        except Exception as e:
            raise CustomException(e, sys)
