#!/usr/bin/env bash
# Generate a starter configuration file using the built-in defaults.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_PATH="${REPO_ROOT}/examples/pipeline/configs/mlops-config.yaml"

if [[ -f "${OUTPUT_PATH}" ]]; then
  echo "[init] Using existing configuration at ${OUTPUT_PATH}"
else
  echo "[init] Creating configuration → ${OUTPUT_PATH}"
  make-mlops-easy init -o "${OUTPUT_PATH}"
fi

echo
echo "You can now customise ${OUTPUT_PATH} and pass it to 'make-mlops-easy train'."
