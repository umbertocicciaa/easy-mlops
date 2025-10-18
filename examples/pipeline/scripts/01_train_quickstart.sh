#!/usr/bin/env bash
# Train and deploy a binary classifier with endpoint generation enabled.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_PATH="${REPO_ROOT}/examples/sample_data.csv"
CONFIG_PATH="${REPO_ROOT}/examples/pipeline/configs/quickstart.yaml"

# Automatically manage the master/worker runtime.
source "${REPO_ROOT}/examples/distributed_runtime.sh"
ensure_master
ensure_worker

echo "[train] Using data: ${DATA_PATH}"
echo "[train] Using config: ${CONFIG_PATH}"
echo "[train] Master URL: ${MASTER_URL}"
echo

make-mlops-easy train \
  "${DATA_PATH}" \
  --target approved \
  -c "${CONFIG_PATH}" \
  --master-url "${MASTER_URL}"

echo
echo "[train] Latest deployments:"
ls -dt "${REPO_ROOT}"/models/deployment_* 2>/dev/null | head -3 || echo "  (none yet)"
