#!/usr/bin/env python3
"""
Training script with COMPLETE CALIBRATION
Focal Loss + Isotonic Calibration + Threshold Learning
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import argparse
from datetime import datetime
import json

from src.data_loader import MIMICDataLoader
from src.calibration_utils import (
    train_with_calibration,
    predict_calibrated
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)

print("=" * 70)
print("TRAINING WITH CALIBRATION - MORTALITY PREDICTION")
print("=" * 70)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='data/in-hospital-mortality')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--learning-rate', type=float, default=0.0003)
parser.add_argument('--lstm-units', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--recurrent-dropout', type=float, default=0.3)
parser.add_argument('--l2-reg', type=float, default=0.01)
parser.add_argument('--use-focal-loss', action='store_true', default=True)
args = parser.parse_args()

# Load data
print("1. Loading data...")
print("-" * 70)

loader = MIMICDataLoader(args.data_dir)
X_train, y_train = loader.load_data('train')
X_val, y_val = loader.load_data('val')
X_test, y_test = loader.load_data('test')

print(f"   Train shape: {X_train.shape}")
print(f"   Val shape:   {X_val.shape}")
print(f"   Test shape:  {X_test.shape}")
print(f"   Features:    {X_train.shape[2]}")
print(f"   Timesteps:   {X_train.shape[1]}")
print()
print(f"   Train mortality rate: {y_train.mean():.2%} ({y_train.sum()}/{len(y_train)})")
print(f"   Val mortality rate:   {y_val.mean():.2%} ({y_val.sum()}/{len(y_val)})")
print(f"   Test mortality rate:  {y_test.mean():.2%} ({y_test.sum()}/{len(y_test)})")
print()

# Create model
print("2. Creating model...")
print("-" * 70)
print(f"   Input shape:         ({X_train.shape[1]}, {X_train.shape[2]})")
print(f"   LSTM units:          {args.lstm_units}")
print(f"   Dropout:             {args.dropout}")
print(f"   Recurrent Dropout:   {args.recurrent_dropout}")
print(f"   L2 Regularization:   {args.l2_reg}")
print(f"   Use Focal Loss:      {args.use_focal_loss}")
print()

def create_model(input_shape, lstm_units=64, dropout=0.5,
                recurrent_dropout=0.3, l2_reg=0.01):
    """
    Simple LSTM model with regularization
    """
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Masking
    mask = layers.Masking(mask_value=0.0)(inputs)
    
    # LSTM
    lstm = layers.LSTM(
        lstm_units,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        kernel_regularizer=regularizers.l2(l2_reg),
        recurrent_regularizer=regularizers.l2(l2_reg),
        return_sequences=False,
        name='lstm'
    )(mask)
    
    # Dense layers
    dense1 = layers.Dense(
        32,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='dense1'
    )(lstm)
    
    dropout1 = layers.Dropout(dropout, name='dropout1')(dense1)
    
    dense2 = layers.Dense(
        16,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='dense2'
    )(dropout1)
    
    dropout2 = layers.Dropout(dropout, name='dropout2')(dense2)
    
    # Output
    outputs = layers.Dense(1, activation='sigmoid', name='output')(dropout2)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='lstm_calibrated')
    return model

model = create_model(
    input_shape=(X_train.shape[1], X_train.shape[2]),
    lstm_units=args.lstm_units,
    dropout=args.dropout,
    recurrent_dropout=args.recurrent_dropout,
    l2_reg=args.l2_reg
)

model.summary()
print()

# Train with calibration
print("3. Training with calibration pipeline...")
print("-" * 70)
print()

model, calibrator, optimal_threshold, history = train_with_calibration(
    model,
    X_train, y_train,
    X_val, y_val,
    epochs=args.epochs,
    batch_size=args.batch_size,
    use_focal_loss=args.use_focal_loss,
    learning_rate=args.learning_rate
)

# Evaluate on test set
print("\n" + "=" * 70)
print("EVALUATING ON TEST SET")
print("=" * 70)

y_pred_proba_test, y_pred_test = predict_calibrated(
    model, calibrator, X_test, threshold=optimal_threshold
)

# Metrics
auroc_test = roc_auc_score(y_test, y_pred_proba_test)
auprc_test = average_precision_score(y_test, y_pred_proba_test)

print(f"\nTest Set Metrics:")
print(f"  AUROC: {auroc_test:.4f}")
print(f"  AUPRC: {auprc_test:.4f}")
print(f"  Threshold: {optimal_threshold:.3f}")

print(f"\nProbability Distribution (Test):")
print(f"  Min: {y_pred_proba_test.min():.4f}")
print(f"  Max: {y_pred_proba_test.max():.4f}")
print(f"  Mean: {y_pred_proba_test.mean():.4f}")
print(f"  Median: {np.median(y_pred_proba_test):.4f}")

print(f"\nClassification Report:")
print(classification_report(
    y_test, y_pred_test,
    target_names=['Survived', 'Died'],
    digits=4
))

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_test)
print(f"  TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
print(f"  FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")

# Save results
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Save model
model.save('models/best_model_calibrated.keras')
print("✓ Model saved to models/best_model_calibrated.keras")

# Save calibrator
calibrator.save('models/calibrator.pkl')

# Save threshold
with open('models/optimal_threshold.txt', 'w') as f:
    f.write(f"{optimal_threshold:.6f}\n")
print(f"✓ Optimal threshold saved: {optimal_threshold:.3f}")

# Save test metrics
test_metrics = {
    'auroc': float(auroc_test),
    'auprc': float(auprc_test),
    'threshold': float(optimal_threshold),
    'confusion_matrix': cm.tolist(),
    'probability_stats': {
        'min': float(y_pred_proba_test.min()),
        'max': float(y_pred_proba_test.max()),
        'mean': float(y_pred_proba_test.mean()),
        'median': float(np.median(y_pred_proba_test))
    }
}

with open('results/test_metrics_calibrated.json', 'w') as f:
    json.dump(test_metrics, f, indent=2)
print("✓ Test metrics saved to results/test_metrics_calibrated.json")

# Save predictions for visualization generation (no TensorFlow needed later)
np.save('results/test_predictions.npy', y_pred_proba_test)
print("✓ Test predictions saved to results/test_predictions.npy")

# Save config
config = {
    'model': {
        'lstm_units': args.lstm_units,
        'dropout': args.dropout,
        'recurrent_dropout': args.recurrent_dropout,
        'l2_reg': args.l2_reg
    },
    'training': {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'use_focal_loss': args.use_focal_loss
    },
    'data': {
        'train_samples': int(len(X_train)),
        'val_samples': int(len(X_val)),
        'test_samples': int(len(X_test)),
        'train_mortality': float(y_train.mean()),
        'val_mortality': float(y_val.mean()),
        'test_mortality': float(y_test.mean())
    }
}

with open('results/training_config_calibrated.json', 'w') as f:
    json.dump(config, f, indent=2)
print("✓ Config saved to results/training_config_calibrated.json")

# Save training history
history_dict = {key: [float(val) for val in values] for key, values in history.history.items()}
with open('results/training_history_calibrated.json', 'w') as f:
    json.dump(history_dict, f, indent=2)
print("✓ Training history saved to results/training_history_calibrated.json")

print("\n" + "=" * 70)
print("✅ ALL DONE!")
print("=" * 70)
print(f"\nSummary:")
print(f"  Test AUROC: {auroc_test:.4f}")
print(f"  Test AUPRC: {auprc_test:.4f}")
print(f"  Optimal Threshold: {optimal_threshold:.3f}")
print(f"  Model: models/best_model_calibrated.keras")
print(f"  Calibrator: models/calibrator.pkl")
print()
