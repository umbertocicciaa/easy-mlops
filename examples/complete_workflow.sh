#!/bin/bash
# Complete Easy MLOps Workflow Example
# This script demonstrates all features of Easy MLOps

set -e

echo "============================================"
echo "Easy MLOps - Complete Workflow Example"
echo "============================================"
echo ""

# Step 1: Initialize project
echo "Step 1: Initialize project configuration"
easy-mlops init -o examples/workflow-config.yaml
echo ""

# Step 2: Train model
echo "Step 2: Train model on sample data"
easy-mlops train examples/sample_data.csv --target approved -c examples/workflow-config.yaml
echo ""

# Get the latest deployment directory
DEPLOYMENT_DIR=$(ls -dt models/deployment_* | head -1)
echo "Using deployment: $DEPLOYMENT_DIR"
echo ""

# Step 3: Check status
echo "Step 3: Check model status"
easy-mlops status "$DEPLOYMENT_DIR"
echo ""

# Step 4: Make predictions
echo "Step 4: Make predictions on sample data"
easy-mlops predict examples/sample_data.csv "$DEPLOYMENT_DIR" -o examples/predictions.json
echo ""

# Step 5: View observability report
echo "Step 5: Generate observability report"
easy-mlops observe "$DEPLOYMENT_DIR" -o examples/monitoring_report.txt
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
echo "  - Configuration: examples/workflow-config.yaml"
echo "  - Predictions: examples/predictions.json"
echo "  - Report: examples/monitoring_report.txt"
echo "  - Model: $DEPLOYMENT_DIR"
echo ""
