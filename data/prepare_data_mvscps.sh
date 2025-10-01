#!/usr/bin/env bash

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Activate uv venv if present; otherwise fall back to `uv run`
USING_VENV=0
if [[ -f "${ROOT_DIR}/.venv-mvscps/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${ROOT_DIR}/.venv-mvscps/bin/activate"
  USING_VENV=1
else
  echo "[WARN] .venv-mvscps not found. Will use 'uv run' for Python steps."
fi

run_py() {
  if [[ ${USING_VENV} -eq 1 ]]; then
    python "$@"
  else
    uv run python "$@"
  fi
}

echo "[INFO] Starting data preparation for MVSCPS..."

hf download cyberagent/mvscps --repo-type dataset --local-dir ./data/mvscps
run_py "${SCRIPT_DIR}/preprocess_data_mvscps.py"

echo "[DONE] Data preparation finished."

