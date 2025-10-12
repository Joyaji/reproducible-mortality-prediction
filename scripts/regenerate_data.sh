#!/bin/bash
# Script to regenerate synthetic data and retrain complete model

echo "======================================================================"
echo "COMPLETE REGENERATION: DATA + TRAINING + VISUALIZATIONS"
echo "======================================================================"
echo ""
echo "This script will:"
echo "  1. Generate new synthetic MIMIC-III data (24K+ episodes)"
echo "  2. Process and structure the data"
echo "  3. Backup old models and results"
echo "  4. Train baseline (Logistic Regression)"
echo "  5. Train Deep Learning model (LSTM + Focal Loss + Calibration)"
echo "  6. Generate visualizations and reports"
echo ""
echo "Estimated time: 30-40 minutes"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

# 1. Backup old data
echo ""
echo "======================================================================"
echo "STEP 1/6: Backup old data, models and results"
echo "======================================================================"

if [ -d "data/in-hospital-mortality" ]; then
    echo "Backing up old data..."
    mv data/in-hospital-mortality data/in-hospital-mortality_backup_$(date +%Y%m%d_%H%M%S)
    echo "  ✓ Data backup created"
fi

if [ -d "models" ]; then
    echo "Backing up old models..."
    mv models archived/backups/models_backup_$(date +%Y%m%d_%H%M%S)
    mkdir models
    echo "  ✓ Models backup created"
fi

if [ -d "results" ]; then
    echo "Backing up old results..."
    mv results archived/backups/results_backup_$(date +%Y%m%d_%H%M%S)
    mkdir results
    mkdir results/plots
    mkdir results/baseline
    echo "  ✓ Results backup created"
fi

# 2. Generate new synthetic data
echo ""
echo "======================================================================"
echo "STEP 2/6: Generating synthetic MIMIC-III data"
echo "======================================================================"
echo "Dataset: 24,327 episodes with realistic correlations"
echo "Features: 15 clinical variables, 48h timesteps"
echo "Improvements: Circadian patterns, sleep, meals, anti-overfitting"
echo ""
python scripts/generate_synthetic_mimic3.py
if [ $? -ne 0 ]; then
    echo "❌ Error generating data"
    exit 1
fi

# 3. Process data
echo ""
echo "======================================================================"
echo "STEP 3/6: Processing data (patient structure)"
echo "======================================================================"
python scripts/process_synthetic_data.py
if [ $? -ne 0 ]; then
    echo "❌ Error processing data"
    exit 1
fi

# 4. Train baseline
echo ""
echo "======================================================================"
echo "STEP 4/6: Training Baseline (Logistic Regression)"
echo "======================================================================"
python src/train_baseline.py
if [ $? -ne 0 ]; then
    echo "❌ Error training baseline"
    exit 1
fi

# 5. Train Deep Learning model
echo ""
echo "======================================================================"
echo "STEP 5/6: Training Deep Learning (LSTM + Focal Loss + Calibration)"
echo "======================================================================"
echo "Configuration:"
echo "  - LSTM: 64 units, Dropout 50%, Recurrent Dropout 30%"
echo "  - Loss: Focal Loss (gamma=2.0, alpha=0.25)"
echo "  - Calibration: Isotonic Regression"
echo "  - Threshold: Optimized by F1-Score"
echo "  - Epochs: 50 (with early stopping)"
echo ""
python src/train_calibrated.py --epochs 50 --batch-size 64
if [ $? -ne 0 ]; then
    echo "❌ Error training model"
    exit 1
fi

# 6. Generate visualizations and reports
echo ""
echo "======================================================================"
echo "STEP 6/6: Generating Visualizations and Reports"
echo "======================================================================"
./run_plots_and_report.sh
if [ $? -ne 0 ]; then
    echo "❌ Error generating visualizations"
    exit 1
fi

echo ""
echo "======================================================================"
echo "✅ COMPLETE REGENERATION FINISHED SUCCESSFULLY!"
echo "======================================================================"
echo ""
echo "Generated files:"
echo "  📊 Data:"
echo "     - data/in-hospital-mortality/ (24,327 episodes)"
echo ""
echo "  🤖 Models:"
echo "     - models/best_model_calibrated.keras"
echo "     - models/calibrator.pkl"
echo "     - models/baseline_model.pkl"
echo ""
echo "  📈 Visualizations:"
echo "     - results/plots/*.png (11+ plots)"
echo ""
echo "  📄 Reports:"
echo "     - results/TECHNICAL_REPORT.md"
echo ""
echo "Next steps:"
echo "  1. Review: results/TECHNICAL_REPORT.md"
echo "  2. Visualize: open results/plots/"
echo "  3. K-Fold Validation: ./run_validation_and_report.sh"
echo ""
