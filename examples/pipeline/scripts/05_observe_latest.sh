#!/usr/bin/env bash
# Generate a monitoring report with stricter alert thresholds applied.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${REPO_ROOT}/examples/pipeline/configs/observability_strict.yaml"
OUTPUT_PATH="${REPO_ROOT}/examples/pipeline/monitoring_report.txt"

source "${REPO_ROOT}/examples/distributed_runtime.sh"
ensure_master
ensure_worker

LATEST_DEPLOYMENT="$(ls -dt "${REPO_ROOT}"/models/deployment_* 2>/dev/null | head -1 || true)"
if [[ -z "${LATEST_DEPLOYMENT}" ]]; then
  echo "[observe] No deployment directories found under ${REPO_ROOT}/models/"
  echo "[observe] Run a training script first (e.g. 01_train_quickstart.sh)."
  exit 1
fi

echo "[observe] Inspecting deployment: ${LATEST_DEPLOYMENT}"
echo "[observe] Writing report to: ${OUTPUT_PATH}"
echo "[observe] Master URL: ${MASTER_URL}"
echo

make-mlops-easy observe \
  "${LATEST_DEPLOYMENT}" \
  -c "${CONFIG_PATH}" \
  -o "${OUTPUT_PATH}" \
  --master-url "${MASTER_URL}"

echo
echo "[observe] Report preview:"
head -n 20 "${OUTPUT_PATH}"
