import sys
import os
import joblib
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from src.exception import CustomException
from src.logger import logging

class ModelTrainer:
    def __init__(self):
        # [C4-m] Guard for output directory
        self.models_dir = os.path.join("outputs", "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # [C-m12] All models with random_state=42
        self.models = {
            "Random Forest"       : RandomForestClassifier(random_state=42),
            "XGBoost"             : XGBClassifier(eval_metric='logloss', verbosity=0, random_state=42),
            "Gradient Boosting"   : GradientBoostingClassifier(random_state=42)
        }

        
        # [M4] n_iter=10 support | [M7] KNN tuning
        self.tuning_params = {
            "Random Forest": {
                'n_estimators': [100, 200, 300], 
                'max_depth': [None, 5, 10, 15], 
                'min_samples_split': [2, 5]
            },
            "XGBoost": {
                'n_estimators' : [100, 200], 
                'max_depth': [3, 5, 7], 
                'learning_rate': [0.01, 0.1, 0.2]
            },
            "Gradient Boosting": {
                'n_estimators' : [100, 200], 
                'max_depth': [3, 5], 
                'learning_rate': [0.05, 0.1]
            },
            "Naive Bayes": {
                'var_smoothing': [1e-9, 1e-8, 1e-7]
            },
            "KNN": {
                'n_neighbors': [3, 5, 7, 9], 
                'weights': ['uniform', 'distance'], 
                'metric': ['euclidean', 'manhattan']
            }
        }

    def initiate_model_trainer(self, X_train, y_train):
        try:
            print("\n--- FASE 4: MODELING (TOP 3 MODELS + SMOTE) ---")
            
            # 1. SMOTE (Otomatis & Eksplisit di dalam engine)
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_train, y_train)
            print(f"[✓] SMOTE Applied. Balanced dataset: {len(y_res)} samples")
            print(f"[*] Metodologi: SMOTE hanya pada Training Set (No Leakage).")


            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            trained_models = {}

            for name, model in self.models.items():
                if name in self.tuning_params:
                    print(f"[*] Tuning {name} (n_iter=10)...")
                    search = RandomizedSearchCV(
                        model, self.tuning_params[name], 
                        n_iter=10, cv=cv, scoring='f1_weighted', n_jobs=1, random_state=42
                    )
                    search.fit(X_res, y_res)
                    best_model = search.best_estimator_
                else:
                    print(f"[*] Training {name} (Default Params)...")
                    model.fit(X_res, y_res)
                    best_model = model
                
                trained_models[name] = best_model
                # [M3] Save model
                pkl_path = os.path.join(self.models_dir, f"{name.lower().replace(' ', '_')}.pkl")
                joblib.dump(best_model, pkl_path)

            print(f"[✓] Semua 3 Model Tersimpan di {self.models_dir}")
            return trained_models

        except Exception as e:
            raise CustomException(e, sys)
