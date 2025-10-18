#!/usr/bin/env bash
# Complete Make MLOps Easy Workflow Example
# This script demonstrates all features of Make MLOps Easy

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Shared helpers spin up a master/worker if one is not already running.
source "${SCRIPT_DIR}/distributed_runtime.sh"
ensure_master
ensure_worker

echo "Master service: ${MASTER_URL}"
echo

echo "============================================"
echo "Make MLOps Easy - Complete Workflow Example"
echo "============================================"
echo ""

# Step 1: Initialize project
CONFIG_PATH="${REPO_ROOT}/examples/pipeline/configs/quickstart.yaml"
echo "Step 1: Use curated quickstart configuration"
echo "Configuration: ${CONFIG_PATH}"
echo ""

# Step 2: Train model
echo "Step 2: Train model on sample data"
make-mlops-easy train \
  "${REPO_ROOT}/examples/sample_data.csv" \
  --target approved \
  -c "${CONFIG_PATH}" \
  --master-url "${MASTER_URL}"
echo ""

# Get the latest deployment directory
DEPLOYMENT_DIR=$(ls -dt "${REPO_ROOT}"/models/deployment_* 2>/dev/null | head -1 || true)
if [[ -z "${DEPLOYMENT_DIR}" ]]; then
  echo "Failed to locate deployment artifacts under ${REPO_ROOT}/models/."
  echo "Check ${EASY_MLOPS_RUNTIME_DIR}/worker.log for details."
  exit 1
fi
echo "Using deployment: $DEPLOYMENT_DIR"
echo ""

# Step 3: Check status
echo "Step 3: Check model status"
make-mlops-easy status "$DEPLOYMENT_DIR" --master-url "${MASTER_URL}"
echo ""

# Step 4: Make predictions
echo "Step 4: Make predictions on sample data"
make-mlops-easy predict \
  "${REPO_ROOT}/examples/sample_data.csv" \
  "$DEPLOYMENT_DIR" \
  -c "${CONFIG_PATH}" \
  -o "${REPO_ROOT}/examples/pipeline/predictions_workflow.json" \
  --master-url "${MASTER_URL}"
echo ""

# Step 5: View observability report
echo "Step 5: Generate observability report"
make-mlops-easy observe \
  "$DEPLOYMENT_DIR" \
  -c "${REPO_ROOT}/examples/pipeline/configs/observability_strict.yaml" \
  -o "${REPO_ROOT}/examples/pipeline/monitoring_report.txt" \
  --master-url "${MASTER_URL}"
echo ""

# Step 6: Use the prediction endpoint script
if [ -f "$DEPLOYMENT_DIR/predict.py" ]; then
    echo "Step 6: Test prediction endpoint script"
    python3 "$DEPLOYMENT_DIR/predict.py" "${REPO_ROOT}/examples/sample_data.csv"
    echo ""
fi

echo "============================================"
echo "Workflow completed successfully!"
echo "============================================"
echo ""
echo "Generated files:"
echo "  - Configuration: ${CONFIG_PATH}"
echo "  - Predictions: ${REPO_ROOT}/examples/pipeline/predictions_workflow.json"
echo "  - Report: ${REPO_ROOT}/examples/pipeline/monitoring_report.txt"
echo "  - Model: $DEPLOYMENT_DIR"
echo ""

echo "Runtime logs are stored under: ${EASY_MLOPS_RUNTIME_DIR}"
