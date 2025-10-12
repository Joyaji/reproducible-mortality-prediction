#!/bin/bash
# Master Script: Validation + Visualizations + Report
# Executes complete validation and report generation pipeline

echo "======================================================================"
echo "VALIDATION AND REPORT PIPELINE"
echo "======================================================================"
echo ""

# 1. K-Fold Cross-Validation
echo "STEP 1/3: K-Fold Cross-Validation"
echo "----------------------------------------------------------------------"
echo "Estimated time: 2-3 hours"
echo ""
python src/validate_kfold.py
if [ $? -ne 0 ]; then
    echo "❌ Error in K-Fold CV"
    exit 1
fi
echo ""

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

# 3. Generate Report
echo "STEP 3/3: Generating Technical Report"
echo "----------------------------------------------------------------------"
echo "Estimated time: < 1 minute"
echo ""
python src/generate_report.py
if [ $? -ne 0 ]; then
    echo "❌ Error generating report"
    exit 1
fi
echo ""

echo "======================================================================"
echo "✅ PIPELINE COMPLETE!"
echo "======================================================================"
echo ""
echo "Generated files:"
echo "  • results/kfold_validation.json"
echo "  • results/plots/*.png (7 plots)"
echo "  • results/plots/metrics_summary.json"
echo "  • results/TECHNICAL_REPORT.md"
echo ""
echo "Next step: Review results/TECHNICAL_REPORT.md"
echo ""
