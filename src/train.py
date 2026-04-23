import sys
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.exception import CustomException
from src.logger import logging

class ModelTrainer:
    def __init__(self):
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, criterion='entropy'),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5),
            "Naive Bayes": GaussianNB(),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "SVM": SVC(probability=True),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            "Gradient Boosting": GradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "ANN": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000)
        }

    def initiate_model_trainer(self, X_train, y_train):
        try:
            logging.info("--- Fase 4: Modeling (10 Models Comparison) ---")
            trained_models = {}
            cv_results = {}

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for name, model in self.models.items():
                logging.info(f"Training {name}...")
                model.fit(X_train, y_train)
                trained_models[name] = model
                
                # Cross Validation
                cv_score = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
                cv_results[name] = cv_score.mean()
                logging.info(f"{name} CV Accuracy: {cv_score.mean():.4f}")

                # Save each model
                model_filename = name.lower().replace(" ", "_") + ".pkl"
                joblib.dump(model, os.path.join(self.models_dir, model_filename))

            logging.info("All 10 models trained and saved successfully")
            return trained_models, cv_results

        except Exception as e:
            raise CustomException(e, sys)
