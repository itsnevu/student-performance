import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_curve, auc
)
from src.exception import CustomException
from src.logger import logging

class ModelEvaluator:
    def __init__(self):
        self.outputs_dir = "outputs"
        os.makedirs(self.outputs_dir, exist_ok=True)

    def evaluate_all(self, trained_models, X_test, y_test):
        try:
            logging.info("--- Fase 5: Model Evaluation (Matrix Comparison) ---")
            performance_list = []
            
            plt.figure(figsize=(10, 8))
            
            for name, model in trained_models.items():
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                performance_list.append({
                    "Model": name,
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1-Score": f1
                })
                
                # ROC Curve for each model
                if y_prob is not None:
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.2f})")

            # Finalize ROC Plot
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve Comparison (All Models)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(self.outputs_dir, "roc_comparison.png"))
            
            # Performance Table
            perf_df = pd.DataFrame(performance_list).sort_values(by="F1-Score", ascending=False)
            perf_df.to_csv(os.path.join(self.outputs_dir, "model_comparison.csv"), index=False)
            
            logging.info("\nPerformance Comparison Table:")
            logging.info("\n" + perf_df.to_string())
            
            # Save the best model name
            best_model_name = perf_df.iloc[0]['Model']
            logging.info(f"--- BEST MODEL: {best_model_name} ---")
            
            return perf_df

        except Exception as e:
            raise CustomException(e, sys)
