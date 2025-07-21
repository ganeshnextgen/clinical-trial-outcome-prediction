#!/usr/bin/env python3
"""
Step 8: Model Evaluation & Threshold Optimization
Comprehensive evaluation with threshold optimization for failure recall
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, fbeta_score, recall_score, precision_score
)

class ModelEvaluator:
    """Model evaluation with threshold optimization for failure recall."""
    def __init__(self, input_dir="data/processed", models_dir="models", output_dir="results"):
        self.input_dir = Path(input_dir)
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.optimal_threshold = 0.5
        
    def load_model_and_data(self):
        model_files = list(self.models_dir.glob("balanced_rf_model_*.pkl"))
        if not model_files:
            raise FileNotFoundError("No trained model found. Run Step 7 first.")
        latest_model_file = max(model_files, key=lambda x: x.stat().st_mtime)
        model = joblib.load(latest_model_file)
        splits_files = list(self.input_dir.glob("data_splits_*.pkl"))
        if not splits_files:
            raise FileNotFoundError("No test splits found. Run Step 6 first.")
        latest_splits_file = max(splits_files, key=lambda x: x.stat().st_mtime)
        splits = joblib.load(latest_splits_file)
        print(f"📂 Loaded model: {latest_model_file}")
        print(f"📂 Loaded test data: {latest_splits_file}")
        print(f"📊 Test set shape: {splits['X_test'].shape}")
        return model, splits
    
    def optimize_threshold_for_recall(self, model, X_val, y_val, beta=2.0):
        print(f"🎯 OPTIMIZING THRESHOLD FOR FAILURE RECALL (β={beta})")
        y_proba = model.predict_proba(X_val)[:, 1]
        thresholds = np.arange(0.05, 0.95, 0.01)
        best_score = 0
        best_threshold = 0.5
        scores = []
        recalls_failure = []
        precisions_failure = []
        for threshold in thresholds:
            y_pred_threshold = (y_proba >= threshold).astype(int)
            score = fbeta_score(y_val, y_pred_threshold, beta=beta, zero_division=0)
            recall_fail = recall_score(y_val, y_pred_threshold, pos_label=0, zero_division=0)
            precision_fail = precision_score(y_val, y_pred_threshold, pos_label=0, zero_division=0)
            scores.append(score)
            recalls_failure.append(recall_fail)
            precisions_failure.append(precision_fail)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        self.optimal_threshold = best_threshold
        print(f"✅ Optimal threshold: {best_threshold:.3f}")
        print(f"✅ Best F{beta}-score: {best_score:.3f}")
        return best_threshold, scores, recalls_failure, precisions_failure
    
    def evaluate_model(self, model, splits):
        X_test, y_test = splits['X_test'], splits['y_test']
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred_optimized = (y_proba >= self.optimal_threshold).astype(int)
        y_pred_standard = (y_proba >= 0.5).astype(int)
        print("🔸 STANDARD THRESHOLD (0.5) RESULTS:")
        print(classification_report(y_test, y_pred_standard, target_names=['Failure', 'Success']))
        print(f"\n🔹 OPTIMIZED THRESHOLD ({self.optimal_threshold:.3f}) RESULTS:")
        print(classification_report(y_test, y_pred_optimized, target_names=['Failure', 'Success']))
        cm_standard = confusion_matrix(y_test, y_pred_standard)
        cm_optimized = confusion_matrix(y_test, y_pred_optimized)
        roc_auc = auc(*roc_curve(y_test, y_proba)[:2])
        print(f"\n🎯 ROC-AUC: {roc_auc:.3f}")
        print(f"📊 Failure Recall Standard: {recall_score(y_test, y_pred_standard, pos_label=0):.3f}, Optimized: {recall_score(y_test, y_pred_optimized, pos_label=0):.3f}")
        return {
            'optimal_threshold': self.optimal_threshold,
            'cm_standard': cm_standard.tolist(),
            'cm_optimized': cm_optimized.tolist(),
            'roc_auc': roc_auc,
            'classification_report_standard': classification_report(y_test, y_pred_standard, output_dict=True),
            'classification_report_optimized': classification_report(y_test, y_pred_optimized, output_dict=True)
        }
    
    def plot_confusion_matrices(self, eval_results):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(eval_results['cm_standard'], annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Standard Threshold (0.5)')
        axes[0].set_ylabel('Actual')
        axes[0].set_xlabel('Predicted')
        sns.heatmap(eval_results['cm_optimized'], annot=True, fmt='d', cmap='Oranges', ax=axes[1])
        axes[1].set_title(f'Optimized ({eval_results["optimal_threshold"]:.3f})')
        axes[1].set_ylabel('Actual')
        axes[1].set_xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix_comparison.png', dpi=300)
        plt.close()
        print(f"📊 Confusion matrices plot saved.")
        
def main():
    print("🚀 STEP 8: MODEL EVALUATION & THRESHOLD OPTIMIZATION")
    evaluator = ModelEvaluator()
    model, splits = evaluator.load_model_and_data()
    best_threshold, scores, recalls, precisions = evaluator.optimize_threshold_for_recall(
        model, splits['X_val'], splits['y_val'], beta=2.0)
    eval_results = evaluator.evaluate_model(model, splits)
    evaluator.plot_confusion_matrices(eval_results)
    print("✅ Step 8 completed: Evaluation and threshold optimization complete.")
    return

if __name__ == "__main__":
    main()
