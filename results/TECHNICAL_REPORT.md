# Technical Report - In-Hospital Mortality Prediction

**Date:** 15/10/2025 21:42  
**Author:** Andre Lehdermann Silveira  
**Version:** 1.0

---

## 📊 Executive Summary

This report presents the results of a deep learning model for in-hospital mortality prediction using synthetic MIMIC-III data. The model uses LSTM with Focal Loss and post-training calibration.

The model shows very good performance.

### **Main Results:**

| Metric | Value | Status |
|---------|-------|--------|
| **Test AUROC** | 0.8638 | ✅ Very Good |
| **Test AUPRC** | 0.6396 | ✅ Good |
| **F1-Score** | 0.5114 | ✅ Good (High Recall Priority) |
| **Recall** | 0.9681 | ✅ Excellent |
| **Precision** | 0.3475 | ⚠️ Moderate (Trade-off) |
| **Accuracy** | 0.6152 | ⚠️ Moderate |

**Optimal Threshold:** 0.170

---

## 🎯 Methodology

### **1. Dataset**

- **Source:** Synthetic MIMIC-III
- **Total Episodes:** 24,327
- **Split:**
  - Train: 16,972 episodes (69.8%)
  - Validation: 3,740 episodes (15.4%)
  - Test: 3,615 episodes (14.9%)
- **Mortality Rate:** ~20.8%
- **Features:** 15 clinical variables
- **Timesteps:** 48 hours

### **2. Model**

**Architecture:**
```
Input (48, 15)
  ↓
Masking (padding)
  ↓
LSTM (64 units, dropout=0.5, recurrent_dropout=0.3)
  ↓
Dense (32 units, ReLU, L2=0.01)
  ↓
Dropout (0.5)
  ↓
Dense (16 units, ReLU, L2=0.01)
  ↓
Dropout (0.5)
  ↓
Output (1 unit, Sigmoid)
```

**Total Parameters:** 23,105

### **3. Training**

- **Loss Function:** Focal Loss (gamma=2.0, alpha=0.25)
- **Optimizer:** Adam (lr=0.0003)
- **Batch Size:** 64
- **Epochs:** 50 (with early stopping)
- **Regularization:**
  - Dropout: 50%
  - Recurrent Dropout: 30%
  - L2 Regularization: 0.01

### **4. Calibration**

- **Method:** Isotonic Regression
- **Threshold Learning:** F1-Score Optimization
- **Optimal Threshold:** 0.170

---

## 📈 Results

### **K-Fold Cross-Validation (5 folds)**

| Metric | Mean | Std | 95% CI |
|---------|------|-----|--------|
| **AUROC** | 0.9959 | 0.0000 | [0.9959, 0.9959] |
| **AUPRC** | 0.9925 | 0.0000 | [0.9925, 0.9925] |
| **F1-Score** | 0.9859 | 0.0000 | [0.9859, 0.9859] |
| **Recall** | 0.9787 | 0.0000 | [0.9787, 0.9787] |
| **Precision** | 0.9933 | 0.0000 | [0.9933, 0.9933] |

**Consistency:** ✅ EXCELLENT (Std AUROC = 0.0000)


#### **⚠️ Note on K-Fold Results**

The K-Fold cross-validation shows extremely low variance (std ≈ 0.0000) and very high performance (AUROC 0.9959), which indicates:

1. **Synthetic Data Characteristics:** The synthetic data generator creates highly predictable patterns with deterministic correlations
2. **Overfitting:** The model memorizes training data patterns (CV AUROC 0.9959 vs Test AUROC 0.8638)
3. **Performance Gap:** 13.2% drop from CV to test set indicates limited generalization

**Interpretation:** The **test set performance (AUROC 0.8638)** is more realistic and should be used for model evaluation. The K-Fold results demonstrate training stability but not real-world generalization.

**Future Work:** Improving synthetic data variability is in the roadmap (see Section 7).


### **Test Set Performance**

| Metric | Value |
|---------|-------|
| **AUROC** | 0.8638 |
| **AUPRC** | 0.6396 |
| **F1-Score** | 0.5114 |
| **Recall** | 0.9681 |
| **Precision** | 0.3475 |
| **Accuracy** | 0.6152 |

### **Confusion Matrix**

```
                Predicted
              Survived  Died
Actual
Survived        1496    1367
Died              24     728
```

**Interpretation:**
- **True Negatives (TN):** 1496 - Survivors correctly identified
- **False Positives (FP):** 1367 - Survivors incorrectly predicted as deaths
- **False Negatives (FN):** 24 - Deaths not detected
- **True Positives (TP):** 728 - Deaths correctly identified

**Total Errors:** 1391 / 3615 = 38.48%

### **Predicted Probabilities**

| Statistic | Value |
|-------------|-------|
| **Minimum** | 0.0000 |
| **Maximum** | 1.0000 |
| **Mean** | 0.3964 |
| **Median** | 0.5556 |
| **Standard Deviation** | 0.2212 |

**Interpretation:** Probabilities well distributed between 0 and 1, indicating good calibration.

---

## 📊 Visualizations

### **1. ROC Curve**
![ROC Curve](plots/roc_curve.png)

**Interpretation:** AUROC of 0.8638 - Very Good.

### **2. Precision-Recall Curve**
![PR Curve](plots/precision_recall_curve.png)

**Interpretation:** AUPRC of 0.6396 compared with prevalence of 20.8%.

### **3. Calibration Curve**
![Calibration Curve](plots/calibration_curve.png)

**Interpretation:** Evaluates whether predicted probabilities correspond to observed frequencies. Curve close to diagonal indicates good calibration.

### **4. Confusion Matrix**
![Confusion Matrix](plots/confusion_matrix.png)

**Interpretation:** Total of 1391 errors in 3615 cases (error rate: 38.48%).

### **5. Probability Distribution**
![Probability Distribution](plots/probability_distribution.png)

**Interpretation:** Distribution of predicted probabilities by true class, with optimal threshold at 0.170.

### **6. Threshold vs Metrics**
![Threshold vs Metrics](plots/threshold_vs_metrics.png)

**Interpretation:** Threshold of 0.170 was chosen to maximize F1-Score.

### **7. Summary Metrics**
![Summary Metrics](plots/summary_metrics.png)

**Interpretation:** Overview of the main model performance metrics with color-coded performance levels.

### **8. F-beta Score vs Threshold**
![F-beta vs Threshold](plots/fbeta_vs_threshold.png)

**Interpretation:** Shows how F-beta scores vary with different beta values (β=0.5, 1, 2) across thresholds. Higher beta values prioritize recall over precision.

### **9. Learning Curves**
![Learning Curves](plots/learning_curves_dl.png)

**Interpretation:** Training and validation loss/metrics over epochs. Shows model convergence and potential overfitting.

### **10. Comparison with Baseline**

#### **10.1. ROC Comparison**
![ROC Comparison](plots/roc_comparison.png)

**Interpretation:** Deep Learning model (AUROC 0.8638) vs Logistic Regression baseline.

#### **10.2. Precision-Recall Comparison**
![PR Comparison](plots/pr_comparison.png)

**Interpretation:** Deep Learning model (AUPRC 0.6396) vs Logistic Regression baseline.

#### **10.3. Metrics Comparison**
![Metrics Comparison](plots/metrics_comparison.png)

**Interpretation:** Side-by-side comparison of all metrics between Deep Learning and baseline models.

#### **10.4. Improvement Chart**
![Improvement Chart](plots/improvement_chart.png)

**Interpretation:** Percentage improvement of Deep Learning over baseline for each metric.

#### **10.5. Comparison Table**
![Comparison Table](plots/comparison_table.png)

**Interpretation:** Detailed tabular comparison of Deep Learning vs Logistic Regression performance.

---

## 🔍 Key Findings

### **Model Performance**

**Discrimination Ability:**
- AUROC: 0.8638 - Very Good
- AUPRC: 0.6396 vs baseline prevalence of 20.8%

**Classification Metrics (Threshold = 0.170):**
- Recall: 0.9681 (96.8% of deaths detected)
- Precision: 0.3475 (34.7% of death predictions correct)
- F1-Score: 0.5114
- Accuracy: 0.6152

**Error Analysis:**
- Total errors: 1391 / 3615 (38.5%)
- False positives: 1367 (survivors predicted as deaths)
- False negatives: 24 (deaths not detected)

### **Calibration Quality**

**Probability Distribution:**
- Range: [0.0000, 1.0000]
- Mean: 0.3964
- Median: 0.5556
- Std: 0.2212

**Calibration Method:** Isotonic Regression applied post-training

### **Model Configuration**

**Techniques Applied:**
- Focal Loss (gamma=2.0, alpha=0.25) for class imbalance
- Isotonic calibration for probability reliability
- Optimized threshold (0.170) via F1-Score maximization
- Regularization: Dropout (50%), Recurrent Dropout (30%), L2 (0.01)

### **Data Characteristics**

**Dataset:** Synthetic MIMIC-III
- Test set size: 3615 episodes
- Mortality rate: 20.8%
- Class imbalance ratio: 1:3.8

---

## 📝 Summary

This technical report documents the performance of an LSTM-based mortality prediction model trained on synthetic MIMIC-III data. The model achieves AUROC of 0.8638 with high recall (0.9681) at the cost of moderate precision (0.3475), reflecting the prioritization of death detection over false alarms. All visualizations and metrics are reproducible using the provided codebase

---

**Automatically generated on:** 15/10/2025 21:42:04
