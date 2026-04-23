import sys
import os
import joblib
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from src.exception import CustomException
from src.logger import logging

class ModelTrainer:
    def __init__(self):
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
        self.models = {
            "Random Forest": (RandomForestClassifier(), {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5, 10]
            }),
            "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7]
            }),
            "Logistic Regression": (LogisticRegression(max_iter=2000), {
                "C": [0.1, 1, 10]
            }),
            "SVM": (SVC(probability=True), {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"]
            })
        }

    def initiate_model_trainer(self, X_train, y_train):
        try:
            logging.info("--- Fase 4: Modeling (Tuning & Imbalance Handling) ---")
            
            # Handling Imbalance ONLY on Train Set
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            logging.info(f"SMOTE applied: Resampled train set size {len(y_train_res)}")

            best_models = {}
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for name, (model, params) in self.models.items():
                logging.info(f"Tuning {name}...")
                search = RandomizedSearchCV(
                    model, param_distributions=params, n_iter=5, 
                    cv=skf, scoring='f1', n_jobs=-1, random_state=42
                )
                search.fit(X_train_res, y_train_res)
                
                best_model = search.best_estimator_
                best_models[name] = best_model
                logging.info(f"{name} best F1: {search.best_score_:.4f}")

                joblib.dump(best_model, os.path.join(self.models_dir, f"{name.lower().replace(' ', '_')}.pkl"))

            return best_models

        except Exception as e:
            raise CustomException(e, sys)
