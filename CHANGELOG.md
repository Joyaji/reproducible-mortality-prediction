# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [1.0.0] - 2025-10-11

### Added
- LSTM model with Focal Loss for mortality prediction
- Synthetic MIMIC-III data generator
- Post-training calibration with Isotonic Regression
- Threshold optimization by F1-Score
- K-Fold cross-validation (5 folds)
- Baseline comparison (Logistic Regression)
- 14 visualization plots
- Automatic technical report generator
- Complete documentation suite

### Model Performance
- AUROC: 0.8638
- AUPRC: 0.6396
- Recall: 96.81%
- F1-Score: 0.5114
- Optimal threshold: 0.170

### Features
- **Data Generator:**
  - 13 realistic clinical features
  - Temporal variability and noise
  - Realistic missingness patterns (MCAR, MAR, MNAR)
  - Multivariate correlations
  - Individual patient variability

- **Model Architecture:**
  - LSTM (128 units)
  - Strong regularization (Dropout 50%, Recurrent Dropout 30%, L2 0.01)
  - Focal Loss (gamma=2.0, alpha=0.25)
  - Isotonic calibration
  - Optimized threshold (0.170)

- **Evaluation:**
  - K-Fold cross-validation
  - Calibration analysis
  - Threshold optimization
  - Baseline comparison
  - 14 visualization plots

### Documentation
- Technical paper (PDF): docs/reproducible-mortality-prediction.pdf
- README with quick start
- REPRODUCIBILITY guide
- Synthetic data generator docs
- Model architecture docs
- Configuration parameters reference

## [0.2.0] - 2025-10-08

### Added
- Calibration with Isotonic Regression
- Threshold optimization
- Improved visualizations
- Baseline comparison

### Changed
- Updated data generator
- Enhanced regularization
- Improved documentation

## [0.1.0] - 2025-10-04

### Added
- Initial LSTM implementation
- Basic synthetic data generator
- Training pipeline
- Basic evaluation metrics

---

## Version History Summary

- **v1.0.0** (2025-10-11): Complete open-source release with full documentation
- **v0.2.0** (2025-10-08): Added calibration and threshold optimization
- **v0.1.0** (2025-10-04): Initial implementation

## Future Roadmap

### Short Term
- Error analysis and failure pattern identification
- Feature importance analysis
- Improved synthetic data variability

### Medium Term
- Validation on real MIMIC-III data
- Comparison with other architectures (XGBoost, Random Forest)
- Interpretability analysis (SHAP, LIME)
- Hyperparameter optimization

### Long Term
- External validation (eICU, MIMIC-IV)
- Clinical deployment preparation
- Prospective evaluation
- Scientific publication

---

For detailed changes, see commit history: https://github.com/lehdermann/reproducible-mortality-prediction/commits/main
