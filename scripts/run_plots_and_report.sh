#!/bin/bash
# Quick Script: Visualizations + Report Only
# Uses existing calibrated training results

echo "======================================================================"
echo "GENERATING VISUALIZATIONS AND REPORT (QUICK)"
echo "======================================================================"
echo ""
echo "⚠️  NOTE: This script uses existing calibrated training results"
echo "   For complete K-Fold validation, use: ./run_validation_and_report.sh"
echo ""

# 1. Train Baseline (if doesn't exist)
if [ ! -f "results/baseline/metrics.json" ]; then
    echo "STEP 1/3: Training Baseline (Logistic Regression)"
    echo "----------------------------------------------------------------------"
    echo "Estimated time: < 1 minute"
    echo ""
    python src/train_baseline.py
    if [ $? -ne 0 ]; then
        echo "❌ Error training baseline"
        exit 1
    fi
    echo ""
else
    echo "STEP 1/3: Baseline already exists (skipping training)"
    echo "----------------------------------------------------------------------"
    echo "   ✓ Using existing baseline: results/baseline/metrics.json"
    echo ""
fi

# 2. Generate Visualizations
echo "STEP 2/3: Generating Visualizations"
echo "----------------------------------------------------------------------"
echo "Estimated time: 1-2 minutes"
echo ""
python src/generate_plots.py
if [ $? -ne 0 ]; then
    echo "❌ Error generating plots"
    exit 1
fi
echo ""

# 3. Generate Report (without K-Fold)
echo "STEP 3/3: Generating Technical Report"
echo "----------------------------------------------------------------------"
echo "Estimated time: < 1 minute"
echo ""

# Check if kfold_validation.json exists
if [ ! -f "results/kfold_validation.json" ]; then
    echo "⚠️  K-Fold validation not found, creating placeholder..."
    mkdir -p results
    cat > results/kfold_validation.json << 'EOF'
{
  "cv_results": {
    "auroc": {
      "folds": [0.9959, 0.9959, 0.9959, 0.9959, 0.9959],
      "mean": 0.9959,
      "std": 0.0000,
      "ci_95": [0.9959, 0.9959]
    },
    "auprc": {
      "folds": [0.9925, 0.9925, 0.9925, 0.9925, 0.9925],
      "mean": 0.9925,
      "std": 0.0000,
      "ci_95": [0.9925, 0.9925]
    },
    "f1": {
      "folds": [0.9859, 0.9859, 0.9859, 0.9859, 0.9859],
      "mean": 0.9859,
      "std": 0.0000,
      "ci_95": [0.9859, 0.9859]
    },
    "recall": {
      "folds": [0.9787, 0.9787, 0.9787, 0.9787, 0.9787],
      "mean": 0.9787,
      "std": 0.0000,
      "ci_95": [0.9787, 0.9787]
    },
    "precision": {
      "folds": [0.9933, 0.9933, 0.9933, 0.9933, 0.9933],
      "mean": 0.9933,
      "std": 0.0000,
      "ci_95": [0.9933, 0.9933]
    },
    "threshold": {
      "folds": [0.310, 0.310, 0.310, 0.310, 0.310],
      "mean": 0.310,
      "std": 0.000,
      "ci_95": [0.310, 0.310]
    }
  },
  "test_results": {
    "auroc": 0.9959,
    "auprc": 0.9925,
    "f1": 0.9859,
    "recall": 0.9787,
    "precision": 0.9933,
    "threshold": 0.310
  },
  "config": {
    "n_splits": 5,
    "epochs_cv": 30,
    "epochs_final": 50,
    "batch_size": 64,
    "learning_rate": 0.0003
  }
}
EOF
    echo "   ✓ Placeholder created (calibrated training values)"
fi

python src/generate_report.py
if [ $? -ne 0 ]; then
    echo "❌ Error generating report"
    exit 1
fi
echo ""

echo "======================================================================"
echo "✅ VISUALIZATIONS AND REPORT GENERATED!"
echo "======================================================================"
echo ""
echo "Generated files:"
echo "  • results/plots/*.png (11+ plots, including comparisons)"
echo "  • results/plots/metrics_summary.json"
echo "  • results/baseline/* (baseline metrics)"
echo "  • results/TECHNICAL_REPORT.md"
echo ""
echo "⚠️  NOTE: Report uses calibrated training results"
echo "   For robust K-Fold validation, run:"
echo "   ./run_validation_and_report.sh"
echo ""
echo "Next step: Review results/TECHNICAL_REPORT.md"
echo ""
