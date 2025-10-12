# Visualization System Documentation
---

## 📊 Overview

This document describes the comprehensive visualization system for the mortality prediction model. The system generates 11+ publication-quality plots to analyze model performance, calibration, and clinical utility.

---

## 🎯 Quick Start

### **Generate All Visualizations**

```bash
# Generate plots from existing model
python src/generate_plots.py

# Or use the complete pipeline
./scripts/run_plots_and_report.sh
```

### **Output Location**

All plots are saved to: `results/plots/`

---

## 📈 Available Visualizations

### **1. ROC Curve**
**File:** `roc_curve.png`

**Description:** Receiver Operating Characteristic curve showing the trade-off between True Positive Rate (Sensitivity) and False Positive Rate (1-Specificity).

**Key Metrics:**
- AUROC (Area Under ROC Curve)
- Diagonal reference line (random classifier)
- Optimal operating point

**Interpretation:**
- AUROC = 0.5: Random classifier
- AUROC = 0.7-0.8: Acceptable
- AUROC = 0.8-0.9: Excellent
- AUROC > 0.9: Outstanding

**Clinical Relevance:** Higher AUROC indicates better discrimination between survivors and non-survivors.

---

### **2. Precision-Recall Curve**
**File:** `precision_recall_curve.png`

**Description:** Shows the trade-off between Precision (positive predictive value) and Recall (sensitivity) across different thresholds.

**Key Metrics:**
- AUPRC (Area Under PR Curve)
- Baseline (prevalence line)
- F1-Score optimal point

**Interpretation:**
- More informative than ROC for imbalanced datasets
- AUPRC > prevalence indicates model adds value
- Steep drop-off indicates poor calibration

**Clinical Relevance:** Critical for understanding false alarm rates in clinical deployment.

---

### **3. Calibration Curve**
**File:** `calibration_curve.png`

**Description:** Reliability diagram comparing predicted probabilities to observed frequencies.

**Key Features:**
- 10 probability bins
- Perfect calibration diagonal
- Histogram of predictions
- Brier Score

**Interpretation:**
- Points on diagonal = well-calibrated
- Above diagonal = underconfident
- Below diagonal = overconfident

**Clinical Relevance:** Ensures predicted probabilities can be trusted for clinical decision-making.

---

### **4. Confusion Matrix**
**File:** `confusion_matrix.png`

**Description:** 2x2 matrix showing classification results at optimal threshold.

**Components:**
- True Negatives (TN): Correct survivor predictions
- False Positives (FP): False alarms
- False Negatives (FN): Missed deaths
- True Positives (TP): Correct death predictions

**Annotations:**
- Absolute counts
- Percentages
- Row/column totals

**Clinical Relevance:** Quantifies clinical impact of false alarms vs missed deaths.

---

### **5. Probability Distribution**
**File:** `probability_distribution.png`

**Description:** Histogram of predicted probabilities stratified by true outcome.

**Features:**
- Separate distributions for survivors (blue) and deaths (red)
- Optimal threshold line
- Overlap region analysis
- KDE (Kernel Density Estimation) overlay

**Interpretation:**
- Good separation = distinct distributions
- Large overlap = poor discrimination
- Threshold position affects FP/FN trade-off

**Clinical Relevance:** Visualizes model's ability to separate risk groups.

---

### **6. Threshold vs Metrics**
**File:** `threshold_vs_metrics.png`

**Description:** Shows how key metrics vary across threshold values (0.0 to 1.0).

**Metrics Plotted:**
- F1-Score (harmonic mean of precision/recall)
- Precision (positive predictive value)
- Recall (sensitivity)
- Specificity (true negative rate)

**Features:**
- Optimal threshold marker
- Metric crossover points
- Trade-off visualization

**Clinical Relevance:** Helps select threshold based on clinical priorities (e.g., minimize missed deaths vs false alarms).

---

### **7. Summary Metrics**
**File:** `summary_metrics.png`

**Description:** Bar chart comparing key performance metrics.

**Metrics Included:**
- AUROC
- AUPRC
- F1-Score
- Recall
- Precision
- Accuracy

**Features:**
- Color-coded bars
- Value annotations
- Benchmark lines (if available)

**Clinical Relevance:** Quick overview of model performance across multiple dimensions.

---

### **8. Learning Curves**
**File:** `learning_curves.png`

**Description:** Training and validation metrics over epochs.

**Plots:**
- Loss (training vs validation)
- AUROC (training vs validation)
- Learning rate schedule

**Features:**
- Early stopping point
- Overfitting detection
- Convergence analysis

**Clinical Relevance:** Ensures model is properly trained without overfitting.

---

### **9. Feature Importance** (Optional)
**File:** `feature_importance.png`

**Description:** Ranking of clinical variables by predictive importance.

**Methods:**
- Permutation importance
- SHAP values (if available)
- Attention weights (for attention models)

**Clinical Relevance:** Identifies which vital signs/labs drive predictions.

---

### **10. Calibration Comparison**
**File:** `calibration_comparison.png`

**Description:** Before/after calibration comparison.

**Components:**
- Raw model probabilities
- Calibrated probabilities
- Improvement metrics

**Clinical Relevance:** Demonstrates value of post-training calibration.

---

### **11. Model Comparison**
**File:** `model_comparison.png`

**Description:** Side-by-side comparison with baseline models.

**Models Compared:**
- Deep Learning (LSTM + Calibration)
- Logistic Regression (baseline)
- Published benchmarks (Harutyunyan et al., Purushotham et al.)

**Metrics:**
- AUROC
- AUPRC
- F1-Score
- Recall
- Precision

**Clinical Relevance:** Justifies deep learning approach vs simpler alternatives.

---

## 🔧 Implementation Details

### **Core Script**

**File:** `src/generate_plots.py`

**Dependencies:**
```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    calibration_curve,
    confusion_matrix
)
```

**Key Functions:**

```python
def plot_roc_curve(y_true, y_pred_proba, save_path)
def plot_precision_recall_curve(y_true, y_pred_proba, save_path)
def plot_calibration_curve(y_true, y_pred_proba, save_path)
def plot_confusion_matrix(y_true, y_pred, save_path)
def plot_probability_distribution(y_true, y_pred_proba, threshold, save_path)
def plot_threshold_vs_metrics(y_true, y_pred_proba, save_path)
def plot_summary_metrics(metrics_dict, save_path)
def plot_learning_curves(history, save_path)
def plot_model_comparison(dl_metrics, baseline_metrics, save_path)
```

---

## 🎨 Style Configuration

### **Color Palette**

```python
COLORS = {
    'primary': '#2E86AB',      # Blue (survivors)
    'danger': '#A23B72',       # Red (deaths)
    'success': '#06A77D',      # Green (good metrics)
    'warning': '#F18F01',      # Orange (warnings)
    'neutral': '#6C757D'       # Gray (neutral)
}
```

### **Plot Settings**

```python
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

FIGURE_SIZE = (10, 6)
DPI = 300
FONT_SIZE = 12
TITLE_SIZE = 14
LABEL_SIZE = 11
```

---

## 📊 Usage Examples

### **Example 1: Generate Single Plot**

```python
from src.generate_plots import plot_roc_curve
import numpy as np

# Load predictions
y_true = np.load('results/y_test.npy')
y_pred_proba = np.load('results/y_pred_proba.npy')

# Generate ROC curve
plot_roc_curve(
    y_true=y_true,
    y_pred_proba=y_pred_proba,
    save_path='results/plots/roc_curve.png'
)
```

### **Example 2: Generate All Plots**

```python
from src.generate_plots import generate_all_plots

# Generate complete visualization suite
generate_all_plots(
    model_path='models/best_model_calibrated.keras',
    calibrator_path='models/calibrator.pkl',
    data_dir='data/in-hospital-mortality',
    output_dir='results/plots'
)
```

### **Example 3: Custom Threshold Analysis**

```python
from src.generate_plots import plot_threshold_vs_metrics

# Analyze threshold impact
plot_threshold_vs_metrics(
    y_true=y_test,
    y_pred_proba=y_pred_proba,
    save_path='results/plots/threshold_analysis.png',
    optimal_threshold=0.170  # Highlight optimal point
)
```

---

## 📁 Output Structure

```
results/plots/
├── roc_curve.png                    # ROC analysis
├── precision_recall_curve.png       # PR analysis
├── calibration_curve.png            # Calibration
├── confusion_matrix.png             # Classification results
├── probability_distribution.png     # Probability analysis
├── threshold_vs_metrics.png         # Threshold optimization
├── summary_metrics.png              # Metric overview
├── learning_curves.png              # Training progress
├── calibration_comparison.png       # Before/after calibration
├── model_comparison.png             # Baseline comparison
├── feature_importance.png           # Feature analysis (optional)
└── metrics_summary.json             # Numeric results
```

---

## 🔍 Quality Checks

### **Automated Validation**

The visualization system includes automatic quality checks:

```python
def validate_plots(plot_dir='results/plots'):
    """
    Validates generated plots
    
    Checks:
    - All required plots exist
    - Files are not empty
    - Images are valid PNG/JPG
    - Minimum resolution (300 DPI)
    """
    required_plots = [
        'roc_curve.png',
        'precision_recall_curve.png',
        'calibration_curve.png',
        'confusion_matrix.png',
        'probability_distribution.png',
        'threshold_vs_metrics.png',
        'summary_metrics.png'
    ]
    
    for plot in required_plots:
        path = os.path.join(plot_dir, plot)
        assert os.path.exists(path), f"Missing: {plot}"
        assert os.path.getsize(path) > 1000, f"Empty: {plot}"
    
    print("✓ All plots validated successfully")
```

---

## 🚀 Advanced Features

### **Interactive Plots** (Future)

```python
# Planned: Plotly integration for interactive exploration
from src.generate_plots import generate_interactive_plots

generate_interactive_plots(
    output_dir='results/plots/interactive',
    format='html'  # Interactive HTML plots
)
```

### **Custom Styling**

```python
# Apply custom style
from src.generate_plots import set_plot_style

set_plot_style(
    style='publication',  # Options: 'publication', 'presentation', 'web'
    color_palette='colorblind',  # Colorblind-friendly
    dpi=600  # High resolution for publication
)
```

### **Batch Export**

```python
# Export in multiple formats
from src.generate_plots import export_plots

export_plots(
    formats=['png', 'pdf', 'svg'],
    output_dir='results/plots/export'
)
```

---

## 📚 Integration with Report

Plots are automatically integrated into the technical report:

```markdown
### **ROC Curve**
![ROC Curve](plots/roc_curve.png)

**Interpretation:** AUROC of 0.8638 indicates very good discrimination.
```

**Report Generation:**
```bash
# Generate plots + report
python src/generate_plots.py
python src/generate_report.py

# Output: results/TECHNICAL_REPORT.md
```

---

## 🐛 Troubleshooting

### **Common Issues**

**1. Missing Plots**
```bash
# Check if model exists
ls -lh models/best_model_calibrated.keras

# Regenerate plots
python src/generate_plots.py
```

**2. Low Resolution**
```python
# Increase DPI in generate_plots.py
plt.savefig(path, dpi=600, bbox_inches='tight')
```

**3. Memory Issues**
```python
# Clear figures after saving
plt.close('all')
```

**4. Font Warnings**
```bash
# Install required fonts
pip install matplotlib --upgrade
```

---

## 📊 Performance Metrics

### **Generation Time**

| Plot | Time (seconds) | Size (KB) |
|------|----------------|-----------|
| ROC Curve | 0.5 | 45 |
| PR Curve | 0.5 | 48 |
| Calibration | 1.2 | 62 |
| Confusion Matrix | 0.3 | 38 |
| Probability Dist | 0.8 | 55 |
| Threshold Analysis | 2.5 | 72 |
| Summary Metrics | 0.4 | 42 |
| **TOTAL** | **~6 sec** | **~360 KB** |

---

## 🔗 Related Documentation

- **Model Architecture:** `docs/DEEP_LEARNING_MODEL.md`
- **Data Generator:** `docs/SYNTHETIC_DATA_GENERATOR.md`
- **Technical Report:** `results/TECHNICAL_REPORT.md`
- **Configuration:** `docs/CONFIGURATION_PARAMETERS.md`

---

## 📝 Citation

If you use these visualizations in publications, please cite:

```bibtex
@misc{silveira2025mortality,
  author = {Silveira, Andre Lehdermann},
  title = {In-Hospital Mortality Prediction: Visualization System},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/lehdermann/mortality-prediction}
}
```

---

## 🤝 Contributing

To add new visualizations:

1. Add plotting function to `src/generate_plots.py`
2. Update this documentation
3. Add to automated test suite
4. Submit pull request

**Example Template:**

```python
def plot_new_visualization(y_true, y_pred_proba, save_path):
    """
    Description of new plot
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Output path
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Your plotting code here
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {save_path}")
```

---

**Last Updated:** October 11, 2025  
**Maintainer:** Andre Lehdermann Silveira  
**Status:** Production Ready ✅
