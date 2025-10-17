#!/bin/bash
# Complete Make MLOps Easy Workflow Example
# This script demonstrates all features of Make MLOps Easy

set -e

echo "============================================"
echo "Make MLOps Easy - Complete Workflow Example"
echo "============================================"
echo ""

# Step 1: Initialize project
CONFIG_PATH="examples/pipeline/configs/quickstart.yaml"
echo "Step 1: Use curated quickstart configuration"
echo "Configuration: ${CONFIG_PATH}"
echo ""

# Step 2: Train model
echo "Step 2: Train model on sample data"
make-mlops-easy train examples/sample_data.csv --target approved -c "${CONFIG_PATH}"
echo ""

# Get the latest deployment directory
DEPLOYMENT_DIR=$(ls -dt models/deployment_* | head -1)
echo "Using deployment: $DEPLOYMENT_DIR"
echo ""

# Step 3: Check status
echo "Step 3: Check model status"
make-mlops-easy status "$DEPLOYMENT_DIR"
echo ""

# Step 4: Make predictions
echo "Step 4: Make predictions on sample data"
make-mlops-easy predict examples/sample_data.csv "$DEPLOYMENT_DIR" -c "${CONFIG_PATH}" -o examples/pipeline/predictions_workflow.json
echo ""

# Step 5: View observability report
echo "Step 5: Generate observability report"
make-mlops-easy observe "$DEPLOYMENT_DIR" -c examples/pipeline/configs/observability_strict.yaml -o examples/pipeline/monitoring_report.txt
echo ""

# Step 6: Use the prediction endpoint script
if [ -f "$DEPLOYMENT_DIR/predict.py" ]; then
    echo "Step 6: Test prediction endpoint script"
    python "$DEPLOYMENT_DIR/predict.py" examples/sample_data.csv
    echo ""
fi

echo "============================================"
echo "Workflow completed successfully!"
echo "============================================"
echo ""
echo "Generated files:"
echo "  - Configuration: ${CONFIG_PATH}"
echo "  - Predictions: examples/pipeline/predictions_workflow.json"
echo "  - Report: examples/pipeline/monitoring_report.txt"
echo "  - Model: $DEPLOYMENT_DIR"
echo ""
