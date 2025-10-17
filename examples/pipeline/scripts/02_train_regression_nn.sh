#!/usr/bin/env bash
# Train a regression model with the neural-network backend.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_PATH="${REPO_ROOT}/examples/data/house_prices.csv"
CONFIG_PATH="${REPO_ROOT}/examples/pipeline/configs/regression_neural_network.yaml"

echo "[train-regression] Using data: ${DATA_PATH}"
echo "[train-regression] Using config: ${CONFIG_PATH}"
echo

make-mlops-easy train "${DATA_PATH}" --target price -c "${CONFIG_PATH}"

echo
echo "[train-regression] Latest deployments:"
ls -dt "${REPO_ROOT}"/models/deployment_* 2>/dev/null | head -3 || echo "  (none yet)"
