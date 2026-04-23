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

    def evaluate(self, models, data, feature_names):
        try:
            lr_model, dt_model = models
            X_test_scaled, X_test_raw, y_test = data
            
            logging.info("Evaluating models")
            
            y_pred_lr = lr_model.predict(X_test_scaled)
            y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
            
            y_pred_dt = dt_model.predict(X_test_raw)
            y_prob_dt = dt_model.predict_proba(X_test_raw)[:, 1]
            
            # Confusion Matrix
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues', ax=ax[0])
            ax[0].set_title('Logistic Regression')
            sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', cmap='Greens', ax=ax[1])
            ax[1].set_title('Decision Tree')
            plt.savefig(os.path.join(self.outputs_dir, "confusion_matrix.png"))
            
            # ROC Curve
            fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
            fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr_lr, tpr_lr, label=f'LR (AUC = {auc(fpr_lr, tpr_lr):.2f})')
            plt.plot(fpr_dt, tpr_dt, label=f'DT (AUC = {auc(fpr_dt, tpr_dt):.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.legend()
            plt.savefig(os.path.join(self.outputs_dir, "roc_curve.png"))
            
            # Feature Importance
            feat_imp = pd.Series(dt_model.feature_importances_, index=feature_names).sort_values(ascending=False).head(5)
            plt.figure(figsize=(8, 6))
            sns.barplot(x=feat_imp.values, y=feat_imp.index)
            plt.savefig(os.path.join(self.outputs_dir, "feature_importance.png"))
            
            # Save Metrics to CSV
            metrics = []
            for name, y_p in [('LR', y_pred_lr), ('DT', y_pred_dt)]:
                metrics.append({
                    'Model': name,
                    'Accuracy': accuracy_score(y_test, y_p),
                    'Precision': precision_score(y_test, y_p),
                    'Recall': recall_score(y_test, y_p),
                    'F1': f1_score(y_test, y_p)
                })
            pd.DataFrame(metrics).to_csv(os.path.join(self.outputs_dir, "metrics.csv"), index=False)
            
            logging.info("Evaluation results saved in 'outputs/'")
            
        except Exception as e:
            raise CustomException(e, sys)
