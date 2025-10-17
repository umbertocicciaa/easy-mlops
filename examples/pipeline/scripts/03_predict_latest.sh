#!/usr/bin/env bash
# Run batch predictions against the most recent deployment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_PATH="${REPO_ROOT}/examples/sample_data.csv"
CONFIG_PATH="${REPO_ROOT}/examples/pipeline/configs/quickstart.yaml"
OUTPUT_PATH="${REPO_ROOT}/examples/pipeline/predictions_latest.json"

LATEST_DEPLOYMENT="$(ls -dt "${REPO_ROOT}"/models/deployment_* 2>/dev/null | head -1 || true)"
if [[ -z "${LATEST_DEPLOYMENT}" ]]; then
  echo "[predict] No deployment directories found under ${REPO_ROOT}/models/"
  echo "[predict] Run a training script first (e.g. 01_train_quickstart.sh)."
  exit 1
fi

echo "[predict] Using deployment: ${LATEST_DEPLOYMENT}"
echo "[predict] Writing predictions to: ${OUTPUT_PATH}"
echo

make-mlops-easy predict "${DATA_PATH}" "${LATEST_DEPLOYMENT}" -c "${CONFIG_PATH}" -o "${OUTPUT_PATH}"

echo
echo "[predict] Sample output:"
cat "${OUTPUT_PATH}" | head -n 10
