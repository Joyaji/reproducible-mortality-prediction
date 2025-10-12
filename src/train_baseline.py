#!/usr/bin/env python3
"""
Train Baseline (Logistic Regression) and Save Results
For comparison with Deep Learning
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from pathlib import Path
from src.baseline import BaselineModel
from src.data_loader import MIMICDataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
    confusion_matrix, roc_curve, precision_recall_curve
)

print("=" * 70)
print("BASELINE TRAINING (LOGISTIC REGRESSION)")
print("=" * 70)
print()

# 1. Load data
print("1. Loading data...")
loader = MIMICDataLoader('data/in-hospital-mortality')
X_train, y_train = loader.load_data('train')
X_test, y_test = loader.load_data('test')

print(f"   Train: {X_train.shape}")
print(f"   Test:  {X_test.shape}")
print(f"   Mortality - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")
print()

# 2. Train baseline
print("2. Training Logistic Regression...")
baseline = BaselineModel(max_iter=2000, random_state=42)
baseline.fit(X_train, y_train)
print("   ✓ Model trained")
print()

# 3. Predictions
print("3. Generating predictions on test set...")
y_pred_proba = baseline.predict_proba(X_test)
y_pred = baseline.predict(X_test, threshold=0.5)
print(f"   Predictions generated: {len(y_pred)}")
print()

# 4. Calculate metrics
print("4. Calculating metrics...")
auroc = roc_auc_score(y_test, y_pred_proba)
auprc = average_precision_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"   AUROC:     {auroc:.4f}")
print(f"   AUPRC:     {auprc:.4f}")
print(f"   F1-Score:  {f1:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   Accuracy:  {accuracy:.4f}")
print()

# 5. Save model
print("5. Saving model...")
os.makedirs('models', exist_ok=True)
baseline.save('models/baseline_model.pkl')
print()

# 6. Save metrics and predictions
print("6. Saving metrics and predictions...")
os.makedirs('results/baseline', exist_ok=True)

# Metrics
metrics = {
    'model': 'Logistic Regression',
    'test_metrics': {
        'auroc': float(auroc),
        'auprc': float(auprc),
        'f1_score': float(f1),
        'recall': float(recall),
        'precision': float(precision),
        'accuracy': float(accuracy)
    },
    'confusion_matrix': {
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    },
    'threshold': 0.5,
    'probability_stats': {
        'min': float(y_pred_proba.min()),
        'max': float(y_pred_proba.max()),
        'mean': float(y_pred_proba.mean()),
        'median': float(np.median(y_pred_proba)),
        'std': float(y_pred_proba.std())
    }
}

with open('results/baseline/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("   ✓ Metrics saved: results/baseline/metrics.json")

# Predictions (for plots)
predictions = {
    'y_test': y_test.tolist(),
    'y_pred_proba': y_pred_proba.tolist(),
    'y_pred': y_pred.tolist()
}

with open('results/baseline/predictions.json', 'w') as f:
    json.dump(predictions, f)
print("   ✓ Predictions saved: results/baseline/predictions.json")

# ROC Curve data
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_data = {
    'fpr': fpr.tolist(),
    'tpr': tpr.tolist(),
    'auroc': float(auroc)
}

with open('results/baseline/roc_curve.json', 'w') as f:
    json.dump(roc_data, f)
print("   ✓ ROC curve saved: results/baseline/roc_curve.json")

# PR Curve data
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
pr_data = {
    'precision': precision_curve.tolist(),
    'recall': recall_curve.tolist(),
    'auprc': float(auprc)
}

with open('results/baseline/pr_curve.json', 'w') as f:
    json.dump(pr_data, f)
print("   ✓ PR curve saved: results/baseline/pr_curve.json")

print()
print("=" * 70)
print("✅ BASELINE TRAINED AND SAVED!")
print("=" * 70)
print()
print("Files created:")
print("  1. models/baseline_model.pkl")
print("  2. results/baseline/metrics.json")
print("  3. results/baseline/predictions.json")
print("  4. results/baseline/roc_curve.json")
print("  5. results/baseline/pr_curve.json")
print()
