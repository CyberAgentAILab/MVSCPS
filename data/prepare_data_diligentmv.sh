#!/usr/bin/env bash

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Activate uv venv ('.venv-mvscps') if available; otherwise fall back to uv run later
if [[ -f "${ROOT_DIR}/.venv-mvscps/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${ROOT_DIR}/.venv-mvscps/bin/activate"
  USING_VENV=1
else
  echo "[WARN] .venv-mvscps not found. Will use 'uv run' for Python steps."
  USING_VENV=0
fi

# Ensure gdown is available (install into the active venv if missing)
if ! command -v gdown >/dev/null 2>&1; then
  echo "[INFO] Installing gdown..."
  if [[ "${USING_VENV}" -eq 1 ]]; then
    python -m pip install -q gdown
  else
    uv run python -m pip install -q gdown
  fi
fi

# Download DiLiGenT-MV dataset (Google Drive)
FILE_ID="18dheWmAxCNaBpYoH3usuFeH9vGlhODvx"
ZIP_PATH="${SCRIPT_DIR}/diligentmv.zip"
OUT_DIR="${SCRIPT_DIR}/DiLiGenT-MV_origin"

echo "[INFO] Downloading DiLiGenT-MV to ${ZIP_PATH} ..."
gdown "${FILE_ID}" -O "${ZIP_PATH}"

echo "[INFO] Unzipping into ${OUT_DIR} ..."
mkdir -p "${OUT_DIR}"
unzip -o "${ZIP_PATH}" -d "${OUT_DIR}"
rm -f "${ZIP_PATH}"

run_py() {
  if [[ ${USING_VENV} -eq 1 ]]; then
    python "$@"
  else
    uv run python "$@"
  fi
}

# Run preprocessing
echo "[INFO] Running preprocessing ..."

run_py ${SCRIPT_DIR}/preprocess_data_diligentmv.py

echo "[DONE] Data prepared at: ${OUT_DIR}"