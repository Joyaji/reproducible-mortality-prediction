#!/usr/bin/env python3
"""
Model Calibration Utilities
Focal Loss + Post-Training Calibration + Threshold Learning
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
import pickle


# ============================================================================
# 1. FOCAL LOSS
# ============================================================================

def get_focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for severe class imbalance
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        gamma: Focus on hard cases (default 2.0)
        alpha: Class balancing weight (default 0.25)
    
    Returns:
        Loss function
    """
    def focal_loss(y_true, y_pred):
        # Clip for numerical stability
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        
        # Cross entropy
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        
        # Focal term
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = K.pow(1 - p_t, gamma)
        
        # Alpha balancing
        alpha_weight = y_true * alpha + (1 - y_true) * (1 - alpha)
        
        loss = alpha_weight * focal_weight * cross_entropy
        return K.mean(loss)
    
    return focal_loss


# ============================================================================
# 2. CALIBRATOR
# ============================================================================

class ModelCalibrator:
    """
    Calibrate probabilities using Isotonic Regression
    """
    def __init__(self):
        self.calibrator = None
    
    def fit(self, y_val, y_pred_proba_val):
        """
        Train calibrator on validation set
        
        Args:
            y_val: True labels (0/1)
            y_pred_proba_val: Predicted probabilities (0-1)
        """
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(y_pred_proba_val, y_val)
        
        print(f"\n✓ Calibrator trained (Isotonic Regression)")
        print(f"  Before calibration:")
        print(f"    Min: {y_pred_proba_val.min():.4f}")
        print(f"    Max: {y_pred_proba_val.max():.4f}")
        print(f"    Mean: {y_pred_proba_val.mean():.4f}")
        print(f"    Median: {np.median(y_pred_proba_val):.4f}")
        
        # Calibrate and verify
        calibrated = self.predict(y_pred_proba_val)
        print(f"  After calibration:")
        print(f"    Min: {calibrated.min():.4f}")
        print(f"    Max: {calibrated.max():.4f}")
        print(f"    Mean: {calibrated.mean():.4f}")
        print(f"    Median: {np.median(calibrated):.4f}")
    
    def predict(self, y_pred_proba):
        """
        Apply calibration to probabilities
        
        Args:
            y_pred_proba: Uncalibrated probabilities
        
        Returns:
            Calibrated probabilities
        """
        if self.calibrator is None:
            raise ValueError("Calibrator not trained! Call .fit() first.")
        
        return self.calibrator.predict(y_pred_proba)
    
    def save(self, path):
        """Save calibrator"""
        with open(path, 'wb') as f:
            pickle.dump(self.calibrator, f)
        print(f"✓ Calibrator saved to {path}")
    
    def load(self, path):
        """Load calibrator"""
        with open(path, 'rb') as f:
            self.calibrator = pickle.load(f)
        print(f"✓ Calibrator loaded from {path}")


# ============================================================================
# 3. THRESHOLD LEARNING
# ============================================================================

def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    """
    Find optimal threshold based on metric
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        metric: 'f1', 'recall', 'precision', 'balanced'
    
    Returns:
        optimal_threshold, best_score
    """
    # Test thresholds
    thresholds = np.arange(0.05, 0.95, 0.01)
    scores = []
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'balanced':
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            score = 2 * (prec * rec) / (prec + rec + 1e-8)
        
        scores.append(score)
    
    # Find best
    best_idx = np.argmax(scores)
    optimal_threshold = thresholds[best_idx]
    best_score = scores[best_idx]
    
    print(f"\n✓ Optimal threshold ({metric}): {optimal_threshold:.3f}")
    print(f"  Score: {best_score:.4f}")
    
    return optimal_threshold, best_score


# ============================================================================
# 4. COMPLETE PIPELINE
# ============================================================================

def train_with_calibration(
    model,
    X_train, y_train,
    X_val, y_val,
    epochs=50,
    batch_size=64,
    use_focal_loss=True,
    learning_rate=0.0003
):
    """
    Pipeline: Train with Focal Loss + Calibrate + Threshold Learning
    
    Args:
        model: Keras model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Number of epochs
        batch_size: Batch size
        use_focal_loss: Use Focal Loss?
        learning_rate: Learning rate
    
    Returns:
        model, calibrator, optimal_threshold
    """
    
    print("=" * 70)
    print("TRAINING PIPELINE WITH CALIBRATION")
    print("=" * 70)
    
    # 1. Class balance
    pos_count = np.sum(y_train)
    neg_count = len(y_train) - pos_count
    pos_weight = neg_count / pos_count
    
    print(f"\n1. Class Balance:")
    print(f"   Positive: {pos_count} ({pos_count/len(y_train):.2%})")
    print(f"   Negative: {neg_count} ({neg_count/len(y_train):.2%})")
    print(f"   Pos Weight: {pos_weight:.2f}")
    
    # 2. Compile
    if use_focal_loss:
        print(f"\n2. Loss Function: Focal Loss (gamma=2.0, alpha=0.25)")
        loss = get_focal_loss(gamma=2.0, alpha=0.25)
    else:
        print(f"\n2. Loss Function: Binary Crossentropy")
        loss = keras.losses.BinaryCrossentropy()
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[
            keras.metrics.AUC(name='auroc'),
            keras.metrics.AUC(curve='PR', name='auprc'),
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    # 3. Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_auroc',
            patience=15,
            restore_best_weights=True,
            mode='max'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
    ]
    
    # 4. Train
    print(f"\n3. Training model...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2
    )
    
    # 5. Evaluate pre-calibration
    print(f"\n4. Pre-Calibration Evaluation:")
    y_pred_proba_val = model.predict(X_val, verbose=0).flatten()
    
    print(f"   Probability Distribution:")
    print(f"     Min: {y_pred_proba_val.min():.4f}")
    print(f"     Max: {y_pred_proba_val.max():.4f}")
    print(f"     Mean: {y_pred_proba_val.mean():.4f}")
    print(f"     Median: {np.median(y_pred_proba_val):.4f}")
    
    auroc_pre = roc_auc_score(y_val, y_pred_proba_val)
    auprc_pre = average_precision_score(y_val, y_pred_proba_val)
    print(f"   AUROC: {auroc_pre:.4f}")
    print(f"   AUPRC: {auprc_pre:.4f}")
    
    # 6. Calibrate
    print(f"\n5. Calibrating model...")
    calibrator = ModelCalibrator()
    calibrator.fit(y_val, y_pred_proba_val)
    
    y_pred_proba_val_cal = calibrator.predict(y_pred_proba_val)
    
    auroc_post = roc_auc_score(y_val, y_pred_proba_val_cal)
    auprc_post = average_precision_score(y_val, y_pred_proba_val_cal)
    print(f"   AUROC post-calibration: {auroc_post:.4f}")
    print(f"   AUPRC post-calibration: {auprc_post:.4f}")
    
    # Check if calibration improves or worsens
    if auroc_post < auroc_pre * 0.95 or auprc_post < auprc_pre * 0.95:
        print(f"\n   ⚠️ WARNING: Calibration WORSENS metrics!")
        print(f"   → Disabling calibration and using RAW probabilities")
        calibrator = None
        y_pred_proba_val_cal = y_pred_proba_val
    
    # 7. Threshold learning
    print(f"\n6. Optimizing threshold...")
    optimal_threshold, f1_score_val = find_optimal_threshold(
        y_val, y_pred_proba_val_cal, metric='f1'
    )
    
    # 8. Final evaluation
    print(f"\n7. Final Evaluation (Val):")
    y_pred_val = (y_pred_proba_val_cal >= optimal_threshold).astype(int)
    
    print(f"\n   Classification Report:")
    print(classification_report(
        y_val, y_pred_val,
        target_names=['Survived', 'Died'],
        digits=4
    ))
    
    print(f"\n   Confusion Matrix:")
    cm = confusion_matrix(y_val, y_pred_val)
    print(f"     TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
    print(f"     FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    
    return model, calibrator, optimal_threshold, history


# ============================================================================
# 5. PREDICTION WITH CALIBRATION
# ============================================================================

def predict_calibrated(model, calibrator, X, threshold=0.5):
    """
    Predict with calibration and custom threshold
    
    Args:
        model: Trained model
        calibrator: Trained calibrator
        X: Input data
        threshold: Threshold for classification
    
    Returns:
        y_pred_proba, y_pred
    """
    # Predict probabilities
    y_pred_proba = model.predict(X, verbose=0).flatten()
    
    # Calibrate
    if calibrator is not None:
        y_pred_proba = calibrator.predict(y_pred_proba)
    
    # Classify
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    return y_pred_proba, y_pred
