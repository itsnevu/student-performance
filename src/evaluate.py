import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, roc_auc_score, confusion_matrix, classification_report
)
from src.exception import CustomException

class ModelEvaluator:
    def __init__(self):
        self.outputs_dir = "outputs"
        os.makedirs(self.outputs_dir, exist_ok=True)

    def evaluate_all(self, trained_models, X_test, y_test, X_train, y_train, dataset_name="Default"):
        try:
            print(f"\n--- FASE 5: EVALUATION (Perfect Dashboard Sync) ---")
            results = []
            # [C5] StratifiedKFold for stable CV
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            for name, model in trained_models.items():
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # [C6] ROC-AUC Guard
                try:
                    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
                except ValueError:
                    auc = np.nan

                # [M6] Cross Validation (Mean & Std)
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')

                results.append({
                    'Model': name,
                    'Accuracy': round(accuracy_score(y_test, y_pred), 4),
                    'F1-Score': round(f1_score(y_test, y_pred, zero_division=0), 4),
                    'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
                    'Recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
                    'AUC-ROC': round(auc, 4) if not np.isnan(auc) else np.nan,
                    'CV F1 Mean': round(cv_scores.mean(), 4),
                    'CV F1 Std': round(cv_scores.std(), 4)
                })

            results_df = pd.DataFrame(results).sort_values('F1-Score', ascending=False).reset_index(drop=True)
            results_df.index += 1
            
            # Save results CSV
            csv_name = f"model_comparison_{dataset_name.replace('.csv', '')}.csv"
            results_df.to_csv(os.path.join(self.outputs_dir, csv_name), index=False)
            
            # --- [M2/M8] Visualisasi Dashboard (Bar Chart + CM) ---
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            fig.suptitle(f'ML Audit Report — {dataset_name}', fontsize=16, fontweight='bold')

            # Plot 1: Bar Chart Comparison
            metrics_cols = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
            plot_df = results_df.set_index('Model')[metrics_cols]
            plot_df.plot(kind='bar', ax=axes[0], colormap='Set2', edgecolor='black')
            axes[0].set_title('Performance Comparison (All Metrics)', fontweight='bold')
            axes[0].set_ylim(0, 1.2)
            axes[0].tick_params(axis='x', rotation=40, labelsize=9)
            axes[0].grid(axis='y', alpha=0.3)

            # Plot 2: Confusion Matrix (Rank #1)
            best_name = results_df.iloc[0]['Model']
            best_model = trained_models[best_name]
            cm = confusion_matrix(y_test, best_model.predict(X_test))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1], annot_kws={"size": 15})
            axes[1].set_title(f'Confusion Matrix — {best_name} (Best Model)', fontweight='bold')
            axes[1].set_xlabel('Predicted Label')
            axes[1].set_ylabel('True Label')
            
            plt.tight_layout()
            chart_name = f"eval_report_{dataset_name.replace('.csv', '')}.png"
            plt.savefig(os.path.join(self.outputs_dir, chart_name), dpi=150)
            plt.close()

            print(f"[✓] Evaluasi Selesai. Laporan Tabel & Dashboard (.png) disimpan di {self.outputs_dir}/")
            print(results_df.to_string())
            
            return results_df

        except Exception as e:
            raise CustomException(e, sys)
