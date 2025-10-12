#!/usr/bin/env python3
"""
K-Fold Cross-Validation for Robust Validation
Ensures that result is not just a "lucky split"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import json
from datetime import datetime

from src.data_loader import MIMICDataLoader
from src.calibration_utils import train_with_calibration, predict_calibrated

print("=" * 70)
print("K-FOLD CROSS-VALIDATION - MORTALITY PREDICTION")
print("=" * 70)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Configuration
N_SPLITS = 5
EPOCHS = 30  # Fewer epochs for CV (faster)
BATCH_SIZE = 64
LEARNING_RATE = 0.0003

# Load data
print("1. Loading data...")
print("-" * 70)

loader = MIMICDataLoader('data/in-hospital-mortality')
X_train, y_train = loader.load_data('train')
X_test, y_test = loader.load_data('test')

print(f"   Train: {X_train.shape}")
print(f"   Test:  {X_test.shape}")
print(f"   Train mortality: {y_train.mean():.2%}")
print(f"   Test mortality:  {y_test.mean():.2%}")
print()

# K-Fold CV
print(f"2. K-Fold Cross-Validation ({N_SPLITS} folds)...")
print("-" * 70)
print()

kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

cv_results = {
    'auroc': [],
    'auprc': [],
    'f1': [],
    'recall': [],
    'precision': [],
    'threshold': []
}

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
    print(f"\n{'='*70}")
    print(f"FOLD {fold+1}/{N_SPLITS}")
    print(f"{'='*70}\n")
    
    # Split
    X_fold_train = X_train[train_idx]
    y_fold_train = y_train[train_idx]
    X_fold_val = X_train[val_idx]
    y_fold_val = y_train[val_idx]
    
    print(f"Train: {X_fold_train.shape}, Mortality: {y_fold_train.mean():.2%}")
    print(f"Val:   {X_fold_val.shape}, Mortality: {y_fold_val.mean():.2%}")
    print()
    
    # Create model
    from tensorflow.keras import layers, regularizers
    
    inputs = layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
    mask = layers.Masking(mask_value=0.0)(inputs)
    lstm = layers.LSTM(64, dropout=0.5, recurrent_dropout=0.3,
                       kernel_regularizer=regularizers.l2(0.01),
                       recurrent_regularizer=regularizers.l2(0.01))(mask)
    dense1 = layers.Dense(32, activation='relu', 
                          kernel_regularizer=regularizers.l2(0.01))(lstm)
    dropout1 = layers.Dropout(0.5)(dense1)
    dense2 = layers.Dense(16, activation='relu',
                          kernel_regularizer=regularizers.l2(0.01))(dropout1)
    dropout2 = layers.Dropout(0.5)(dense2)
    outputs = layers.Dense(1, activation='sigmoid')(dropout2)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Train with calibration
    model, calibrator, threshold, _ = train_with_calibration(
        model,
        X_fold_train, y_fold_train,
        X_fold_val, y_fold_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        use_focal_loss=True,
        learning_rate=LEARNING_RATE
    )
    
    # Evaluate
    y_pred_proba, y_pred = predict_calibrated(
        model, calibrator, X_fold_val, threshold=threshold
    )
    
    # Metrics
    from sklearn.metrics import recall_score, precision_score
    
    auroc = roc_auc_score(y_fold_val, y_pred_proba)
    auprc = average_precision_score(y_fold_val, y_pred_proba)
    f1 = f1_score(y_fold_val, y_pred)
    recall = recall_score(y_fold_val, y_pred)
    precision = precision_score(y_fold_val, y_pred)
    
    cv_results['auroc'].append(auroc)
    cv_results['auprc'].append(auprc)
    cv_results['f1'].append(f1)
    cv_results['recall'].append(recall)
    cv_results['precision'].append(precision)
    cv_results['threshold'].append(threshold)
    
    print(f"\nFold {fold+1} Results:")
    print(f"  AUROC:     {auroc:.4f}")
    print(f"  AUPRC:     {auprc:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Threshold: {threshold:.3f}")
    
    # Clear memory
    del model
    keras.backend.clear_session()

# Final results
print("\n" + "=" * 70)
print("FINAL RESULTS - K-FOLD CROSS-VALIDATION")
print("=" * 70)

for metric_name, values in cv_results.items():
    mean = np.mean(values)
    std = np.std(values)
    print(f"\n{metric_name.upper()}:")
    print(f"  Folds: {', '.join([f'{v:.4f}' for v in values])}")
    print(f"  Mean:  {mean:.4f}")
    print(f"  Std:   {std:.4f}")
    print(f"  95% CI: [{mean - 1.96*std:.4f}, {mean + 1.96*std:.4f}]")

# Evaluate on test set (train final model)
print("\n" + "=" * 70)
print("EVALUATION ON TEST SET (Final Model)")
print("=" * 70)
print()

# Create final model
inputs = layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
mask = layers.Masking(mask_value=0.0)(inputs)
lstm = layers.LSTM(64, dropout=0.5, recurrent_dropout=0.3,
                   kernel_regularizer=regularizers.l2(0.01),
                   recurrent_regularizer=regularizers.l2(0.01))(mask)
dense1 = layers.Dense(32, activation='relu', 
                      kernel_regularizer=regularizers.l2(0.01))(lstm)
dropout1 = layers.Dropout(0.5)(dense1)
dense2 = layers.Dense(16, activation='relu',
                      kernel_regularizer=regularizers.l2(0.01))(dropout1)
dropout2 = layers.Dropout(0.5)(dense2)
outputs = layers.Dense(1, activation='sigmoid')(dropout2)

model_final = keras.Model(inputs=inputs, outputs=outputs)

# Use 20% of train as val
val_size = int(0.2 * len(X_train))
X_train_final = X_train[:-val_size]
y_train_final = y_train[:-val_size]
X_val_final = X_train[-val_size:]
y_val_final = y_train[-val_size:]

model_final, calibrator_final, threshold_final, _ = train_with_calibration(
    model_final,
    X_train_final, y_train_final,
    X_val_final, y_val_final,
    epochs=50,
    batch_size=BATCH_SIZE,
    use_focal_loss=True,
    learning_rate=LEARNING_RATE
)

# Evaluate on test
y_pred_proba_test, y_pred_test = predict_calibrated(
    model_final, calibrator_final, X_test, threshold=threshold_final
)

auroc_test = roc_auc_score(y_test, y_pred_proba_test)
auprc_test = average_precision_score(y_test, y_pred_proba_test)
f1_test = f1_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)

print(f"\nTest Set Results:")
print(f"  AUROC:     {auroc_test:.4f}")
print(f"  AUPRC:     {auprc_test:.4f}")
print(f"  F1-Score:  {f1_test:.4f}")
print(f"  Recall:    {recall_test:.4f}")
print(f"  Precision: {precision_test:.4f}")

# Compare with CV
print(f"\nComparison CV vs Test:")
print(f"  AUROC:  CV {np.mean(cv_results['auroc']):.4f} ± {np.std(cv_results['auroc']):.4f}  |  Test {auroc_test:.4f}")
print(f"  F1:     CV {np.mean(cv_results['f1']):.4f} ± {np.std(cv_results['f1']):.4f}  |  Test {f1_test:.4f}")

# Save results
os.makedirs('results', exist_ok=True)

results_summary = {
    'cv_results': {
        metric: {
            'folds': [float(v) for v in values],
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'ci_95': [
                float(np.mean(values) - 1.96*np.std(values)),
                float(np.mean(values) + 1.96*np.std(values))
            ]
        }
        for metric, values in cv_results.items()
    },
    'test_results': {
        'auroc': float(auroc_test),
        'auprc': float(auprc_test),
        'f1': float(f1_test),
        'recall': float(recall_test),
        'precision': float(precision_test),
        'threshold': float(threshold_final)
    },
    'config': {
        'n_splits': N_SPLITS,
        'epochs_cv': EPOCHS,
        'epochs_final': 50,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE
    }
}

with open('results/kfold_validation.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("\n✓ Results saved to results/kfold_validation.json")

print("\n" + "=" * 70)
print("✅ VALIDATION COMPLETE!")
print("=" * 70)
print(f"\nConsistency: {'✅ EXCELLENT' if np.std(cv_results['auroc']) < 0.01 else '⚠️ CHECK'}")
print(f"Std AUROC: {np.std(cv_results['auroc']):.4f} ({'< 0.01 = consistent' if np.std(cv_results['auroc']) < 0.01 else '>= 0.01 = variable'})")
