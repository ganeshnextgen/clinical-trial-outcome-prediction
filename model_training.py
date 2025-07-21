#!/usr/bin/env python3
"""
Step 7: Balanced Model Training - Improved for Failure Recall
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
from datetime import datetime
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

class BalancedModelTrainer:
    """Balanced model trainer for clinical trial prediction"""
    def __init__(self, input_dir="data/processed", output_dir="models"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = BalancedRandomForestClassifier(
            n_estimators=400,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            sampling_strategy='auto',
            bootstrap=True
        )
        
    def load_training_data(self):
        splits_files = list(self.input_dir.glob("data_splits_*.pkl"))
        if not splits_files:
            raise FileNotFoundError("No training splits found. Run Step 6 first.")
        latest_splits_file = max(splits_files, key=lambda x: x.stat().st_mtime)
        splits = joblib.load(latest_splits_file)
        print(f"📂 Loaded training splits: {latest_splits_file}")
        print(f"📊 Training set: {splits['X_train'].shape}")
        print(f"📊 Validation set: {splits['X_val'].shape}")
        print(f"📊 Test set: {splits['X_test'].shape}")
        return splits
    
    def analyze_class_distribution(self, splits):
        print("📊 ANALYZING CLASS DISTRIBUTION")
        train_failure_count = (splits['y_train'] == 0).sum()
        train_success_count = (splits['y_train'] == 1).sum()
        imbalance_ratio = train_success_count / max(train_failure_count, 1)
        print(f"   • Failures: {train_failure_count}")
        print(f"   • Successes: {train_success_count}")
        print(f"   • Imbalance Ratio: {imbalance_ratio:.2f}")
        return imbalance_ratio
    
    def train_balanced_model(self, X_train, y_train, X_val, y_val):
        print("🎯 TRAINING BALANCED RANDOM FOREST")
        self.model.fit(X_train, y_train)
        train_predictions = self.model.predict(X_train)
        print("📊 Training Performance:")
        print(classification_report(y_train, train_predictions, target_names=['Failure', 'Success']))
        val_predictions = self.model.predict(X_val)
        print("📊 Validation Performance:")
        print(classification_report(y_val, val_predictions, target_names=['Failure', 'Success']))
        
        from sklearn.metrics import roc_auc_score, recall_score, precision_score
        val_prob = self.model.predict_proba(X_val)
        val_auc = roc_auc_score(y_val, val_prob[:, 1])
        val_recall_failure = recall_score(y_val, val_predictions, pos_label=0)
        val_precision_failure = precision_score(y_val, val_predictions, pos_label=0, zero_division=0)
        training_results = {
            'val_auc': val_auc,
            'val_recall_failure': val_recall_failure,
            'val_precision_failure': val_precision_failure,
            'n_estimators': self.model.n_estimators
        }
        print(f"\n🎯 Key Validation Metrics:")
        print(f"   • ROC-AUC: {val_auc:.3f}")
        print(f"   • Failure Recall: {val_recall_failure:.3f}")
        print(f"   • Failure Precision: {val_precision_failure:.3f}")
        return training_results, val_predictions, val_prob
        
    def save_trained_model(self, training_results, feature_mapping=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file = self.output_dir / f"balanced_rf_model_{timestamp}.pkl"
        joblib.dump(self.model, model_file)
        model_metadata = {
            'model_type': 'BalancedRandomForestClassifier',
            'model_params': self.model.get_params(),
            'training_timestamp': timestamp,
            'training_results': training_results,
            'feature_mapping': feature_mapping
        }
        metadata_file = self.output_dir / f"model_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(model_metadata, f, indent=2, default=str)
        print(f"💾 Model saved: {model_file}")
        print(f"💾 Metadata: {metadata_file}")
        return model_file, metadata_file

def main():
    print("🚀 STEP 7: BALANCED MODEL TRAINING")
    trainer = BalancedModelTrainer()
    splits = trainer.load_training_data()
    trainer.analyze_class_distribution(splits)
    feature_mapping = None
    mapping_files = list(trainer.input_dir.glob("feature_mapping_*.json"))
    if mapping_files:
        latest_mapping = max(mapping_files, key=lambda x: x.stat().st_mtime)
        with open(latest_mapping, 'r') as f:
            feature_mapping = json.load(f)
    training_results, val_preds, val_prob = trainer.train_balanced_model(
        splits['X_train'], splits['y_train'], splits['X_val'], splits['y_val'])
    trainer.save_trained_model(training_results, feature_mapping)
    print("✅ Step 7 completed: Balanced model trained.")
    return

if __name__ == "__main__":
    main()
