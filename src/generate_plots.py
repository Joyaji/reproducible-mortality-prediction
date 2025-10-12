#!/usr/bin/env python3
"""
Visualization Generator for Reports
ROC, PR, Calibration, Confusion Matrix, etc.

This script generates all visualizations from pre-computed metrics.
No TensorFlow required - uses saved predictions and metrics.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix
)
from sklearn.calibration import calibration_curve, CalibrationDisplay
import json

# Configure style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

print("=" * 70)
print("VISUALIZATION GENERATOR")
print("=" * 70)
print()

# Create directory
os.makedirs('results/plots', exist_ok=True)

# Load pre-computed metrics and predictions
print("1. Loading saved metrics and predictions...")

# Load data from MIMICDataLoader (only labels, no model needed)
from src.data_loader import MIMICDataLoader
loader = MIMICDataLoader('data/in-hospital-mortality')
X_test, y_test = loader.load_data('test')
print(f"   ✓ Loaded {len(y_test)} test labels")

# Try to load saved predictions
predictions_file = 'results/test_predictions.npy'
if os.path.exists(predictions_file):
    print(f"   ✓ Loading saved predictions from {predictions_file}")
    y_pred_proba = np.load(predictions_file)
    
    with open('models/optimal_threshold.txt', 'r') as f:
        threshold = float(f.read().strip())
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    print(f"   ✓ Threshold: {threshold:.3f}")
    print()
    
else:
    # Fallback: Load model and generate predictions (requires TensorFlow)
    print("   ⚠ Predictions not found, loading model...")
    from tensorflow import keras
    from src.calibration_utils import predict_calibrated, get_focal_loss
    import pickle
    
    focal_loss = get_focal_loss(gamma=2.0, alpha=0.25)
    model = keras.models.load_model(
        'models/best_model_calibrated.keras',
        custom_objects={'focal_loss': focal_loss}
    )
    
    with open('models/calibrator.pkl', 'rb') as f:
        calibrator = pickle.load(f)
    
    with open('models/optimal_threshold.txt', 'r') as f:
        threshold = float(f.read().strip())
    
    y_pred_proba, y_pred = predict_calibrated(model, calibrator, X_test, threshold=threshold)
    
    # Save predictions for future use
    np.save(predictions_file, y_pred_proba)
    print(f"   ✓ Saved predictions to {predictions_file}")
    print()

# ============================================================================
# PLOT 1: ROC Curve
# ============================================================================
print("4. Generating ROC Curve...")

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/plots/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: results/plots/roc_curve.png")

# ============================================================================
# PLOT 2: Precision-Recall Curve
# ============================================================================
print("5. Generating Precision-Recall Curve...")

pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = average_precision_score(y_test, y_pred_proba)

plt.figure(figsize=(10, 8))
plt.plot(pr_recall, pr_precision, color='blue', lw=2,
         label=f'PR curve (AUC = {pr_auc:.4f})')
plt.axhline(y=y_test.mean(), color='red', linestyle='--', lw=2,
            label=f'Baseline ({y_test.mean():.2%})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
plt.legend(loc="lower left", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/plots/precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: results/plots/precision_recall_curve.png")

# ============================================================================
# PLOT 3: Calibration Curve
# ============================================================================
print("6. Generating Calibration Curve...")

prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)

plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfectly calibrated')
plt.plot(prob_pred, prob_true, 's-', lw=2, markersize=8,
         label='Model', color='darkorange')
plt.xlabel('Mean Predicted Probability', fontsize=14)
plt.ylabel('Fraction of Positives', fontsize=14)
plt.title('Calibration Curve (Reliability Diagram)', fontsize=16, fontweight='bold')
plt.legend(loc='upper left', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/plots/calibration_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: results/plots/calibration_curve.png")

# ============================================================================
# PLOT 4: Confusion Matrix
# ============================================================================
print("7. Generating Confusion Matrix...")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Survived', 'Died'],
            yticklabels=['Survived', 'Died'],
            annot_kws={'size': 16, 'weight': 'bold'})
plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=14, fontweight='bold')
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')

# Adicionar métricas
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)

plt.text(0.5, -0.15, f'Accuracy: {accuracy:.4f}  |  Precision: {precision:.4f}  |  Recall: {recall:.4f}  |  F1: {f1:.4f}',
         ha='center', transform=plt.gca().transAxes, fontsize=12)

plt.tight_layout()
plt.savefig('results/plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: results/plots/confusion_matrix.png")

# ============================================================================
# PLOT 5: Probability Distribution
# ============================================================================
print("8. Generating Probability Distribution...")

plt.figure(figsize=(12, 6))

# Histogram by class
plt.subplot(1, 2, 1)
plt.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label='Survived', color='blue')
plt.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label='Died', color='red')
plt.axvline(threshold, color='black', linestyle='--', lw=2, label=f'Threshold ({threshold:.3f})')
plt.xlabel('Predicted Probability', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Probability Distribution by True Class', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Box plot
plt.subplot(1, 2, 2)
data_box = [y_pred_proba[y_test == 0], y_pred_proba[y_test == 1]]
bp = plt.boxplot(data_box, labels=['Survived', 'Died'], patch_artist=True)
bp['boxes'][0].set_facecolor('blue')
bp['boxes'][1].set_facecolor('red')
plt.axhline(threshold, color='black', linestyle='--', lw=2, label=f'Threshold ({threshold:.3f})')
plt.ylabel('Predicted Probability', fontsize=12)
plt.title('Probability Distribution (Box Plot)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/plots/probability_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: results/plots/probability_distribution.png")

# ============================================================================
# PLOT 6: Threshold vs Metrics
# ============================================================================
print("9. Generating Threshold vs Metrics...")

from sklearn.metrics import f1_score, recall_score, precision_score

thresholds = np.arange(0.05, 0.95, 0.01)
f1_scores = []
recalls = []
precisions = []

for thresh in thresholds:
    y_pred_thresh = (y_pred_proba >= thresh).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_thresh, zero_division=0))
    recalls.append(recall_score(y_test, y_pred_thresh, zero_division=0))
    precisions.append(precision_score(y_test, y_pred_thresh, zero_division=0))

plt.figure(figsize=(12, 6))
plt.plot(thresholds, f1_scores, label='F1-Score', lw=2)
plt.plot(thresholds, recalls, label='Recall', lw=2)
plt.plot(thresholds, precisions, label='Precision', lw=2)
plt.axvline(threshold, color='black', linestyle='--', lw=2, 
            label=f'Optimal Threshold ({threshold:.3f})')
plt.xlabel('Threshold', fontsize=14)
plt.ylabel('Score', fontsize=14)
plt.title('Metrics vs Threshold', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/plots/threshold_vs_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: results/plots/threshold_vs_metrics.png")

# ============================================================================
# PLOT 6.5: F-beta Score vs Threshold for Different Beta Values
# ============================================================================
print("9.5. Generating F-beta vs Threshold...")

from sklearn.metrics import fbeta_score

# Different beta values
beta_values = [0.5, 1.0, 2.0, 3.0]
beta_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
beta_labels = ['β=0.5 (Precision focus)', 'β=1.0 (F1, Balanced)', 'β=2.0 (Recall focus)', 'β=3.0 (High Recall focus)']

plt.figure(figsize=(14, 8))

# Calculate F-beta for each beta value
for beta, color, label in zip(beta_values, beta_colors, beta_labels):
    fbeta_scores = []
    for thresh in thresholds:
        y_pred_thresh = (y_pred_proba >= thresh).astype(int)
        fbeta = fbeta_score(y_test, y_pred_thresh, beta=beta, zero_division=0)
        fbeta_scores.append(fbeta)
    
    # Find optimal threshold for this beta
    optimal_idx = np.argmax(fbeta_scores)
    optimal_thresh = thresholds[optimal_idx]
    optimal_fbeta = fbeta_scores[optimal_idx]
    
    # Plot
    plt.plot(thresholds, fbeta_scores, label=label, lw=2.5, color=color)
    
    # Mark optimal point
    plt.scatter([optimal_thresh], [optimal_fbeta], s=100, color=color, 
                marker='o', edgecolors='black', linewidths=2, zorder=5)
    plt.text(optimal_thresh, optimal_fbeta + 0.02, f'{optimal_thresh:.3f}',
             ha='center', fontsize=9, fontweight='bold')

# Mark current threshold
plt.axvline(threshold, color='black', linestyle='--', lw=2, alpha=0.7,
            label=f'Current Threshold ({threshold:.3f})')

plt.xlabel('Threshold', fontsize=14, fontweight='bold')
plt.ylabel('F-beta Score', fontsize=14, fontweight='bold')
plt.title('F-beta Score vs Threshold for Different β Values', fontsize=16, fontweight='bold')
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

# Add explanation text
explanation = (
    "β < 1: Emphasizes Precision (fewer false positives)\n"
    "β = 1: Balanced (F1-Score)\n"
    "β > 1: Emphasizes Recall (fewer false negatives)"
)
plt.text(0.98, 0.02, explanation, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('results/plots/fbeta_vs_threshold.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: results/plots/fbeta_vs_threshold.png")

# ============================================================================
# PLOT 7: Métricas Resumidas
# ============================================================================
print("10. Generating Summary Metrics...")

metrics = {
    'AUROC': roc_auc,
    'AUPRC': pr_auc,
    'F1-Score': f1,
    'Recall': recall,
    'Precision': precision,
    'Accuracy': accuracy
}

# Define colors based on performance
colors = []
for value in metrics.values():
    if value >= 0.9:
        colors.append('#2ecc71')  # Green - Excellent
    elif value >= 0.7:
        colors.append('#3498db')  # Blue - Good
    elif value >= 0.5:
        colors.append('#f39c12')  # Orange - Moderate
    else:
        colors.append('#e74c3c')  # Red - Poor

plt.figure(figsize=(12, 7))
bars = plt.bar(metrics.keys(), metrics.values(), color=colors, 
               edgecolor='black', linewidth=1.5, alpha=0.8)

# Set appropriate y-axis limits (0 to 1.1 to show all metrics)
plt.ylim([0, 1.1])
plt.ylabel('Score', fontsize=14, fontweight='bold')
plt.title('Model Performance Metrics Summary', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Add reference lines
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='Baseline (0.5)')
plt.axhline(y=0.7, color='lightblue', linestyle='--', alpha=0.3, linewidth=1.5, label='Good (0.7)')
plt.axhline(y=0.9, color='lightgreen', linestyle='--', alpha=0.3, linewidth=1.5, label='Excellent (0.9)')

# Add values on bars
for bar, (name, value) in zip(bars, metrics.items()):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.03,
             f'{value:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.legend(loc='upper right', fontsize=10)
plt.xticks(rotation=0, fontsize=12)
plt.tight_layout()
plt.savefig('results/plots/summary_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: results/plots/summary_metrics.png")

# ============================================================================
# Save metrics to JSON
# ============================================================================
print("\n11. Saving metrics...")

metrics_summary = {
    'test_metrics': {
        'auroc': float(roc_auc),
        'auprc': float(pr_auc),
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
    'threshold': float(threshold),
    'probability_stats': {
        'min': float(y_pred_proba.min()),
        'max': float(y_pred_proba.max()),
        'mean': float(y_pred_proba.mean()),
        'median': float(np.median(y_pred_proba)),
        'std': float(y_pred_proba.std())
    },
    'roc_curve': {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'auroc': float(roc_auc)
    },
    'pr_curve': {
        'precision': pr_precision.tolist(),
        'recall': pr_recall.tolist(),
        'auprc': float(pr_auc)
    }
}

with open('results/plots/metrics_summary.json', 'w') as f:
    json.dump(metrics_summary, f, indent=2)

print("   ✓ Saved: results/plots/metrics_summary.json")

# ============================================================================
# COMPARISON WITH BASELINE (if available)
# ============================================================================
baseline_available = os.path.exists('results/baseline/metrics.json')

if baseline_available:
    print("\n12. Generating comparisons with Baseline...")
    
    # Load baseline data
    with open('results/baseline/metrics.json', 'r') as f:
        baseline_metrics = json.load(f)
    
    with open('results/baseline/roc_curve.json', 'r') as f:
        baseline_roc = json.load(f)
    
    with open('results/baseline/pr_curve.json', 'r') as f:
        baseline_pr = json.load(f)
    
    # PLOT 8: ROC Comparison
    print("   Generating ROC Comparison...")
    plt.figure(figsize=(10, 8))
    
    # Deep Learning
    plt.plot(fpr, tpr, color='darkorange', lw=2.5, 
             label=f'Deep Learning (AUC = {roc_auc:.4f})')
    
    # Baseline
    plt.plot(baseline_roc['fpr'], baseline_roc['tpr'], 
             color='blue', lw=2.5, linestyle='--',
             label=f'Logistic Regression (AUC = {baseline_roc["auroc"]:.4f})')
    
    # Random
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle=':', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve Comparison: Deep Learning vs Baseline', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/plots/roc_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: results/plots/roc_comparison.png")
    
    # PLOT 9: PR Comparison
    print("   Generating PR Comparison...")
    plt.figure(figsize=(10, 8))
    
    # Deep Learning - use data saved in metrics_summary
    dl_pr_recall = metrics_summary['pr_curve']['recall']
    dl_pr_precision = metrics_summary['pr_curve']['precision']
    dl_pr_auc = metrics_summary['pr_curve']['auprc']
    
    plt.plot(dl_pr_recall, dl_pr_precision, color='darkorange', lw=2.5,
             label=f'Deep Learning (AUC = {dl_pr_auc:.4f})')
    
    # Baseline
    plt.plot(baseline_pr['recall'], baseline_pr['precision'],
             color='blue', lw=2.5, linestyle='--',
             label=f'Logistic Regression (AUC = {baseline_pr["auprc"]:.4f})')
    
    # Baseline prevalence
    plt.axhline(y=y_test.mean(), color='gray', linestyle=':', lw=2,
                label=f'Baseline Prevalence ({y_test.mean():.2%})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve Comparison: Deep Learning vs Baseline', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/plots/pr_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: results/plots/pr_comparison.png")
    
    # PLOT 10: Metrics Comparison Bar Chart
    print("   Generating Metrics Comparison...")
    
    metrics_names = ['AUROC', 'AUPRC', 'F1-Score', 'Recall', 'Precision', 'Accuracy']
    dl_values = [roc_auc, pr_auc, f1, recall, precision, accuracy]
    baseline_values = [
        baseline_metrics['test_metrics']['auroc'],
        baseline_metrics['test_metrics']['auprc'],
        baseline_metrics['test_metrics']['f1_score'],
        baseline_metrics['test_metrics']['recall'],
        baseline_metrics['test_metrics']['precision'],
        baseline_metrics['test_metrics']['accuracy']
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 7))
    bars1 = ax.bar(x - width/2, dl_values, width, label='Deep Learning', color='darkorange')
    bars2 = ax.bar(x + width/2, baseline_values, width, label='Logistic Regression', color='blue')
    
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Performance Comparison: Deep Learning vs Baseline', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    
    # Add values on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/plots/metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: results/plots/metrics_comparison.png")
    
    # PLOT 11: Improvement Percentage
    print("   Generating Improvement Chart...")
    
    improvements = []
    for dl_val, bl_val in zip(dl_values, baseline_values):
        if bl_val > 0:
            improvement = ((dl_val - bl_val) / bl_val) * 100
        else:
            improvement = 0
        improvements.append(improvement)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax.barh(metrics_names, improvements, color=colors, alpha=0.7)
    
    ax.set_xlabel('Improvement (%)', fontsize=14)
    ax.set_title('Deep Learning Improvement over Baseline (%)', fontsize=16, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add values
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        width = bar.get_width()
        ax.text(width + (1 if width > 0 else -1), bar.get_y() + bar.get_height()/2.,
               f'{imp:+.1f}%', ha='left' if width > 0 else 'right', 
               va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/plots/improvement_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: results/plots/improvement_chart.png")
    
    # Add comparison to JSON
    metrics_summary['baseline_comparison'] = {
        'baseline_metrics': baseline_metrics['test_metrics'],
        'improvements': {
            name: float(imp) for name, imp in zip(metrics_names, improvements)
        }
    }
    
    with open('results/plots/metrics_summary.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    print("   ✓ Comparisons added to metrics_summary.json")

# ============================================================================
# ADDITIONAL PLOT: Learning Curves (if history available)
# ============================================================================
# Search for history file (preference order)
history_paths = [
    'results/training_history_calibrated.json',
    'results/training_history_regularized.json',
    'results/training_history.json'
]

history_path = None
for path in history_paths:
    if os.path.exists(path):
        history_path = path
        break

if history_path:
    print("\n13. Generating Learning Curves...")
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Detect format (Keras 2 vs 3)
    loss_key = 'loss' if 'loss' in history else 'train_loss'
    auroc_key = 'auroc' if 'auroc' in history else 'train_auroc'
    
    epochs = range(1, len(history[loss_key]) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Deep Learning Model - Training History', fontsize=16, fontweight='bold')
    
    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history[loss_key], 'b-', linewidth=2, label='Train', marker='o', markersize=3)
    ax.plot(epochs, history['val_loss'], 'r--', linewidth=2, label='Validation', marker='s', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Curve', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # AUROC
    ax = axes[0, 1]
    ax.plot(epochs, history[auroc_key], 'b-', linewidth=2, label='Train', marker='o', markersize=3)
    ax.plot(epochs, history['val_auroc'], 'r--', linewidth=2, label='Validation', marker='s', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('AUROC Curve', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # AUPRC
    ax = axes[1, 0]
    auprc_key = 'auprc' if 'auprc' in history else 'train_auprc'
    ax.plot(epochs, history[auprc_key], 'b-', linewidth=2, label='Train', marker='o', markersize=3)
    ax.plot(epochs, history['val_auprc'], 'r--', linewidth=2, label='Validation', marker='s', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('AUPRC', fontsize=12)
    ax.set_title('AUPRC Curve', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Overfitting gap
    ax = axes[1, 1]
    auroc_gap = np.array(history[auroc_key]) - np.array(history['val_auroc'])
    ax.plot(epochs, auroc_gap, 'g-', linewidth=2, marker='o', markersize=3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Train - Val AUROC', fontsize=12)
    ax.set_title('Overfitting Analysis', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/learning_curves_dl.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: results/plots/learning_curves_dl.png")

# ============================================================================
# PLOT ADICIONAL: Comparison Table (se baseline disponível)
# ============================================================================
if baseline_available:
    print("\n14. Generating Comparison Table...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Table data
    table_data = [
        ['Metric', 'Logistic Regression', 'Deep Learning', 'Improvement'],
        ['AUROC', f"{baseline_metrics['test_metrics']['auroc']:.4f}", 
         f"{roc_auc:.4f}", 
         f"+{((roc_auc - baseline_metrics['test_metrics']['auroc']) / baseline_metrics['test_metrics']['auroc'] * 100):.1f}%"],
        ['AUPRC', f"{baseline_metrics['test_metrics']['auprc']:.4f}", 
         f"{pr_auc:.4f}",
         f"+{((pr_auc - baseline_metrics['test_metrics']['auprc']) / baseline_metrics['test_metrics']['auprc'] * 100):.1f}%"],
        ['F1-Score', f"{baseline_metrics['test_metrics']['f1_score']:.4f}", 
         f"{f1:.4f}",
         f"+{((f1 - baseline_metrics['test_metrics']['f1_score']) / baseline_metrics['test_metrics']['f1_score'] * 100):.1f}%"],
        ['Recall', f"{baseline_metrics['test_metrics']['recall']:.4f}", 
         f"{recall:.4f}",
         f"+{((recall - baseline_metrics['test_metrics']['recall']) / baseline_metrics['test_metrics']['recall'] * 100):.1f}%"],
        ['Precision', f"{baseline_metrics['test_metrics']['precision']:.4f}", 
         f"{precision:.4f}",
         f"+{((precision - baseline_metrics['test_metrics']['precision']) / baseline_metrics['test_metrics']['precision'] * 100):.1f}%"],
        ['Accuracy', f"{baseline_metrics['test_metrics']['accuracy']:.4f}", 
         f"{accuracy:.4f}",
         f"+{((accuracy - baseline_metrics['test_metrics']['accuracy']) / baseline_metrics['test_metrics']['accuracy'] * 100):.1f}%"]
    ]
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Header style
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Cell style
    for i in range(1, 7):
        for j in range(4):
            if j == 3:  # Coluna improvement
                table[(i, j)].set_facecolor('#E7E6E6')
    
    plt.title('Performance Comparison: Deep Learning vs Baseline', 
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig('results/plots/comparison_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: results/plots/comparison_table.png")

print("\n" + "=" * 70)
print("✅ VISUALIZATIONS GENERATED!")
print("=" * 70)
print("\nFiles created:")
print("  1. results/plots/roc_curve.png")
print("  2. results/plots/precision_recall_curve.png")
print("  3. results/plots/calibration_curve.png")
print("  4. results/plots/confusion_matrix.png")
print("  5. results/plots/probability_distribution.png")
print("  6. results/plots/threshold_vs_metrics.png")
print("  6.5. results/plots/fbeta_vs_threshold.png")
print("  7. results/plots/summary_metrics.png")
print("  8. results/plots/metrics_summary.json")

if os.path.exists('results/training_history_calibrated.json'):
    print("  9. results/plots/learning_curves_dl.png")

if baseline_available:
    print("\n  COMPARISONS WITH BASELINE:")
    print("  10. results/plots/roc_comparison.png")
    print("  11. results/plots/pr_comparison.png")
    print("  12. results/plots/metrics_comparison.png")
    print("  13. results/plots/improvement_chart.png")
    print("  14. results/plots/comparison_table.png")
else:
    print("\n  ⚠️  Baseline not found. Execute 'python src/train_baseline.py' to generate comparisons.")

print()
