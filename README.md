# Reproducible Mortality Prediction

Open-source, reproducible implementation of deep learning for in-hospital mortality prediction.
Simplified replication of Rajkomar et al. (2018) using LSTM + Focal Loss + Calibration.

## 📄 Papers

**This Work:**
- 📘 [Technical Paper (PDF)](docs/reproducible-mortality-prediction.pdf) - Complete methodology and results

**Original Paper:**
- **Scalable and accurate deep learning with electronic health records**  
  Rajkomar, A., Oren, E., Chen, K. et al.  
  *npj Digital Medicine* 1, 18 (2018)  
  https://doi.org/10.1038/s41746-018-0029-1

## 🎯 Objective

Implement and validate a Deep Learning model for ICU mortality prediction, incorporating:

- **Architecture**: LSTM with strong regularization
- **Loss Function**: Focal Loss (handles class imbalance)
- **Calibration**: Post-training Isotonic Regression
- **Threshold Learning**: Optimized by F1-Score
- **Dataset**: Synthetic MIMIC-III (24,327 episodes)
- **Baseline**: Logistic Regression

## 🏆 Main Results

| Metric | Deep Learning | Baseline (LR) | Improvement |
|---------|---------------|---------------|----------|
| **AUROC** | **0.8638** | 0.7042 | **+22.7%** |
| **AUPRC** | **0.6396** | 0.4564 | **+40.1%** |
| **Recall** | **96.81%** | 33.78% | **+186.6%** |
| **F1-Score** | **0.5114** | 0.4025 | **+27.1%** |

✅ **High Sensitivity:** Detects 96.8% of deaths (critical in medicine)

## 📁 Project Structure

```
Systematic_Review/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
│
├── data/
│   └── in-hospital-mortality/     # Synthetic MIMIC-III Dataset (24.3K episodes)
│       ├── train/                 # 16,972 episodes (69.9%)
│       ├── val/                   # 3,740 episodes (14.8%)
│       └── test/                  # 3,615 episodes (15.2%)
│
├── src/                           # Main source code
│   ├── train_dl.py                # DL Training (LSTM + Focal Loss + Calibration)
│   ├── train_baseline.py          # Baseline Training (Logistic Regression)
│   ├── generate_plots.py          # Generation of 14 visualizations
│   ├── generate_report.py         # Technical report generation
│   ├── calibration_utils.py       # Focal Loss + Calibration + Threshold Learning
│   ├── data_loader.py             # Data loading
│   └── validate_kfold.py          # K-Fold validation
│
├── scripts/                             # Generation and execution scripts
│   ├── generate_synthetic_mimic3.py     # Synthetic data generator
│   ├── process_synthetic_data.py        # Data processor
│   ├── regenerate_data.sh               # Complete regeneration
│   ├── run_plots_and_report.sh          # Generate plots + report
│   └── run_validation_and_report.sh     # K-Fold validation
│
├── config/
│   └── constants.py               # Centralized configuration parameters
│
├── models/                        # Trained models
│   ├── best_model_calibrated.keras     # Deep Learning model
│   ├── calibrator.pkl                  # Isotonic Regression calibrator
│   ├── baseline_model.pkl              # Baseline Logistic Regression
│   └── optimal_threshold.txt           # Optimal threshold (0.170)
│
├── results/                       # Results and visualizations
│   ├── plots/                     # 14 high-quality plots
│   ├── baseline/                  # Baseline metrics
│   └── TECHNICAL_REPORT.md        # Automatic technical report
│
├── docs/                          # Technical documentation
│   ├── GERADOR_DADOS_SINTETICOS.md     # Generator documentation
│   ├── CONFIGURACAO_PARAMETROS.md      # Parameters reference
│   └── MODELO_DEEP_LEARNING.md         # DL model documentation
```

## 🚀 Initial Setup

### 1. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # No Mac/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate synthetic data

```bash
# Generate complete dataset (24,327 episodes)
python scripts/generate_synthetic_mimic3.py

# Process data
python scripts/process_synthetic_data.py
```

### 4. Verify data

```bash
ls -lh data/in-hospital-mortality/train/
head data/in-hospital-mortality/train/listfile.csv
```

## 📊 Dataset

### **Synthetic MIMIC-III (Current Version)**

- **Total:** 24,327 ICU episodes
- **Features:** 15 clinical variables (vital signs + labs)
- **Time Window:** 48 hours of observation
- **Mortality Rate:** 20.8% (imbalanced, realistic)
- **Splits:**
  - Train: 16,972 episodes (69.9%)
  - Validation: 3,740 episodes (14.8%)
  - Test: 3,615 episodes (15.2%)

### **Generator Features:**

✅ 13 implemented features:

- Circadian patterns (24h)
- Sleep and meal simulation
- Temporal variability (noise, jitter, dropout, artifacts)
- Realistic missingness (MCAR, MAR, MNAR)
- Multivariate correlations
- Documentation quality by shift
- Individual variability (anti-overfitting)
- Multivariate mortality model

📚 **Complete documentation:** `docs/GERADOR_DADOS_SINTETICOS.md`


## 🧠 Model Architecture

```
Input (48 timesteps, 15 features)
    ↓
Masking Layer (ignores padding)
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

**Techniques Employed:**

- ✅ **Focal Loss** (gamma=2.0, alpha=0.25) - Handles class imbalance
- ✅ **Calibration** - Post-training Isotonic Regression
- ✅ **Threshold Learning** - Optimized by F1-Score (0.170)
- ✅ **Strong Regularization** - Dropout 50%, Recurrent Dropout 30%, L2 0.01

📚 **Complete documentation:** `docs/DEEP_LEARNING_MODEL.md`

## 📈 Training

### **Option 1: Complete Pipeline (RECOMMENDED)**

```bash
# Regenerate data + train baseline + train DL + generate visualizations
./scripts/regenerate_data.sh
```

### **Option 2: Individual Training**

```bash
# 1. Train Baseline (Logistic Regression)
python src/train_baseline.py

# 2. Train Deep Learning (LSTM + Focal Loss + Calibration)
python src/train_calibrated.py --epochs 50 --batch-size 64

# 3. Generate visualizations and report
./scripts/run_plots_and_report.sh
```

### **Option 3: K-Fold Validation**

```bash
# 5-fold cross-validation
./scripts/run_validation_and_report.sh
```

## 📊 Visualizations

The project automatically generates 14 high-quality visualizations:

1. ROC Curve
2. Precision-Recall Curve
3. Calibration Curve
4. Confusion Matrix
5. Probability Distribution
6. Threshold vs Metrics
7. Summary Metrics
8. Learning Curves (Loss, AUROC, AUPRC, Overfitting)
9. ROC Comparison (DL vs Baseline)
10. PR Comparison (DL vs Baseline)
11. Metrics Comparison (bars)
12. Improvement Chart (%)
13. Comparison Table (visual table)
14. metrics_summary.json

**Location:** `results/plots/`

## 🔄 Reproducibility

This project follows open-source and reproducibility best practices:

- ✅ **Complete source code** - All scripts included
- ✅ **Synthetic data generator** - No proprietary data needed
- ✅ **Pre-trained models** - Available in releases (optional)
- ✅ **Reference results** - Metrics and visualizations included
- ✅ **Detailed documentation** - Step-by-step reproduction guide

📚 **Full reproduction guide:** [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)

## 📚 Documentation

### **Main Documents:**

- **README.md** - This file (overview and quick start)
- **[Technical Paper (PDF)](docs/reproducible-mortality-prediction.pdf)** - Complete methodology and results
- **REPRODUCIBILITY.md** - Complete reproduction guide
- **results/TECHNICAL_REPORT.md** - Automatic technical report

### **Technical Documentation (docs/):**

- **SYNTHETIC_DATA_GENERATOR.md** - Generator's 13 features and implementation
- **DEEP_LEARNING_MODEL.md** - Architecture, techniques, and usage
- **CONFIGURATION_PARAMETERS.md** - Complete parameters reference
- **VISUALIZATION_SYSTEM.md** - Visualization and plotting system

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) before submitting pull requests.

### Ways to Contribute:
- 🐛 Report bugs or issues
- 💡 Suggest new features or improvements
- 📝 Improve documentation
- 🧪 Add tests
- 🔧 Submit bug fixes or enhancements

## 📝 Citation

If you use this code in your research, please cite:

**This repository:**
```bibtex
@software{lehdermann2025mortality,
  title={Reproducible Mortality Prediction: LSTM with Focal Loss and Calibration},
  author={Lehdermann Silveira, André},
  year={2025},
  url={https://github.com/lehdermann/reproducible-mortality-prediction},
  note={Open-source replication study}
}
```

**Original paper:**
```bibtex
@article{rajkomar2018scalable,
  title={Scalable and accurate deep learning with electronic health records},
  author={Rajkomar, Alvin and Oren, Eyal and Chen, Kai and others},
  journal={npj Digital Medicine},
  volume={1},
  number={1},
  pages={18},
  year={2018},
  publisher={Nature Publishing Group},
  doi={10.1038/s41746-018-0029-1}
}
```

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Important:** If using real MIMIC-III data, you must comply with PhysioNet's Data Use Agreement.

## 📜 Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed history of changes and version information.

## 👤 Author

**André Lehdermann Silveira**  
Master's Student in Applied Computing  
Universidade do Vale do Rio dos Sinos (Unisinos)  
📧 Contact: [GitHub](https://github.com/lehdermann)

## ⚠️ Disclaimer

This is an **academic replication project** for educational purposes. Synthetic data should not be used for real clinical decisions. For clinical use, only use properly approved real MIMIC-III data.
