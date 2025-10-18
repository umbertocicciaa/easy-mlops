#!/usr/bin/env bash
# Helper utilities for running the distributed runtime in example scripts.
#
# The helper will automatically start a local master and worker (using the
# default CLI commands) when none are detected. Processes are torn down when
# the parent script exits. Set EXAMPLES_USE_EMBEDDED_MASTER=0 or
# EXAMPLES_USE_EMBEDDED_WORKER=0 to skip automatic startup.

if [[ -n "${EASY_MLOPS_DISTRIBUTED_RUNTIME_IMPORTED:-}" ]]; then
  return 0
fi
export EASY_MLOPS_DISTRIBUTED_RUNTIME_IMPORTED=1

MASTER_URL="${MASTER_URL:-http://127.0.0.1:8000}"
export MASTER_URL

# Derive a log directory relative to the repository root if available.
HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${EASY_MLOPS_RUNTIME_DIR:-}" ]]; then
  if [[ -n "${REPO_ROOT:-}" ]]; then
    EASY_MLOPS_RUNTIME_DIR="${REPO_ROOT}/examples/.runtime"
  else
    EASY_MLOPS_RUNTIME_DIR="${HELPER_DIR}/.runtime"
  fi
fi
export EASY_MLOPS_RUNTIME_DIR
mkdir -p "${EASY_MLOPS_RUNTIME_DIR}"

MASTER_PID=""
WORKER_PID=""
MASTER_STARTED=0
WORKER_STARTED=0
MASTER_LOG_FILE="${EASY_MLOPS_RUNTIME_DIR}/master.log"
WORKER_LOG_FILE="${EASY_MLOPS_RUNTIME_DIR}/worker.log"

runtime_log() {
  local level="$1"
  shift
  printf '[runtime][%s] %s\n' "${level}" "$*"
}

parse_master_endpoint() {
  local url="${MASTER_URL#*://}"
  url="${url%%/*}"

  MASTER_HOST="${url%%:*}"
  if [[ -z "${MASTER_HOST}" || "${MASTER_HOST}" == "${url}" ]]; then
    MASTER_HOST="${MASTER_HOST:-127.0.0.1}"
  fi

  if [[ "${url}" == *:* ]]; then
    MASTER_PORT="${url##*:}"
  else
    MASTER_PORT="8000"
  fi
}

command -v curl >/dev/null 2>&1 || {
  runtime_log error "The 'curl' command is required to run the examples."
  exit 1
}

command -v make-mlops-easy >/dev/null 2>&1 || {
  runtime_log error "'make-mlops-easy' must be available on PATH."
  exit 1
}

parse_master_endpoint

__easy_mlops_runtime_cleanup() {
  local code=$?

  if [[ "${WORKER_STARTED}" == "1" && -n "${WORKER_PID}" ]]; then
    runtime_log info "Stopping embedded worker (PID ${WORKER_PID}). Logs: ${WORKER_LOG_FILE}"
    kill "${WORKER_PID}" >/dev/null 2>&1 || true
    wait "${WORKER_PID}" 2>/dev/null || true
  fi

  if [[ "${MASTER_STARTED}" == "1" && -n "${MASTER_PID}" ]]; then
    runtime_log info "Stopping embedded master (PID ${MASTER_PID}). Logs: ${MASTER_LOG_FILE}"
    kill "${MASTER_PID}" >/dev/null 2>&1 || true
    wait "${MASTER_PID}" 2>/dev/null || true
  fi

  return $code
}

if [[ -z "${EASY_MLOPS_RUNTIME_CLEANUP_SET:-}" ]]; then
  trap __easy_mlops_runtime_cleanup EXIT
  export EASY_MLOPS_RUNTIME_CLEANUP_SET=1
fi

wait_for_master() {
  for attempt in {1..30}; do
    if curl -fsS "${MASTER_URL}/health" >/dev/null 2>&1; then
      return 0
    fi

    if [[ "${MASTER_STARTED}" == "1" && -n "${MASTER_PID}" ]] && ! kill -0 "${MASTER_PID}" 2>/dev/null; then
      runtime_log error "Embedded master exited unexpectedly. Check ${MASTER_LOG_FILE}."
      return 1
    fi

    sleep 1
  done

  runtime_log error "Master service did not become ready after 30s. Inspect ${MASTER_LOG_FILE}."
  return 1
}

ensure_master() {
  if [[ "${EXAMPLES_USE_EMBEDDED_MASTER:-1}" == "0" ]]; then
    runtime_log info "Skipping embedded master startup (EXAMPLES_USE_EMBEDDED_MASTER=0)."
    return 0
  fi

  if curl -fsS "${MASTER_URL}/health" >/dev/null 2>&1; then
    runtime_log info "Detected master service at ${MASTER_URL}."
    return 0
  fi

  runtime_log info "Starting embedded master at ${MASTER_URL}. Logs: ${MASTER_LOG_FILE}"
  make-mlops-easy master start --host "${MASTER_HOST}" --port "${MASTER_PORT}" \
    >"${MASTER_LOG_FILE}" 2>&1 &
  MASTER_PID=$!
  MASTER_STARTED=1

  if ! wait_for_master; then
    exit 1
  fi

  runtime_log info "Embedded master is ready (PID ${MASTER_PID})."
}

ensure_worker() {
  if [[ "${EXAMPLES_USE_EMBEDDED_WORKER:-1}" == "0" ]]; then
    runtime_log info "Skipping embedded worker startup (EXAMPLES_USE_EMBEDDED_WORKER=0)."
    return 0
  fi

  if [[ "${WORKER_STARTED}" == "1" && -n "${WORKER_PID}" ]]; then
    return 0
  fi

  runtime_log info "Starting embedded worker targeting ${MASTER_URL}. Logs: ${WORKER_LOG_FILE}"
  make-mlops-easy worker start \
    --master-url "${MASTER_URL}" \
    --poll-interval "${EASY_MLOPS_WORKER_POLL_INTERVAL:-2.0}" \
    >"${WORKER_LOG_FILE}" 2>&1 &
  WORKER_PID=$!
  WORKER_STARTED=1

  sleep 2
  if ! kill -0 "${WORKER_PID}" 2>/dev/null; then
    runtime_log error "Embedded worker failed to start. Check ${WORKER_LOG_FILE}."
    exit 1
  fi
}
