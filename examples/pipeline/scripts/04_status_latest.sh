#!/usr/bin/env bash
# Inspect deployment metadata and monitoring stats for the latest run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${REPO_ROOT}/examples/pipeline/configs/quickstart.yaml"

source "${REPO_ROOT}/examples/distributed_runtime.sh"
ensure_master
ensure_worker

LATEST_DEPLOYMENT="$(ls -dt "${REPO_ROOT}"/models/deployment_* 2>/dev/null | head -1 || true)"
if [[ -z "${LATEST_DEPLOYMENT}" ]]; then
  echo "[status] No deployment directories found under ${REPO_ROOT}/models/"
  echo "[status] Run a training script first (e.g. 01_train_quickstart.sh)."
  exit 1
fi

echo "[status] Inspecting deployment: ${LATEST_DEPLOYMENT}"
echo "[status] Master URL: ${MASTER_URL}"
echo

make-mlops-easy status \
  "${LATEST_DEPLOYMENT}" \
  -c "${CONFIG_PATH}" \
  --master-url "${MASTER_URL}"
