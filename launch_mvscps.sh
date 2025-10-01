#!/usr/bin/env bash
# --- Activate uv venv ---
source .venv-mvscps/bin/activate

# --- CUDA toolchain & build settings (adjust if needed) ---
export PATH="/usr/local/cuda/bin:$PATH"
export LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.9;8.0"

# --- Runtime toggles ---
export PL_WEIGHTS_ONLY=0
export WANDB_MODE=disabled

obj_name_list=("20250205_ceramic_buddha" "20250303_bronze_loong" "20250304_diffuse_dog" "20250304_diffuse_flower_girl" "20250304_metallic_fox" "20250515_ceramic_buddha")

for obj_name in "${obj_name_list[@]}"; do
  echo "Running for object: $obj_name"
  python launch.py +conf=mvscps conf.dataset.obj_name=$obj_name
done
