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

# --- Experiment knobs ---
NUM_LIGHT_LIST=(32)
OBJ_NAME_LIST=("buddha" "reading" "cow" "pot2" "bear")

# to reproduce geometry evaluation results, uncomment the following line.
# NUM_LIGHT_LIST=(1 2 3 4 5 6 7 8 10 12 14 16 20 24 28 32 48 56 64 80 96)
# and use the following settings to save time
# conf.exp.predict_after_train=false
# conf.dataset.test.view_light_index_fname=view_20_light_1

for obj_name in "${OBJ_NAME_LIST[@]}"; do
  for num_lights in "${NUM_LIGHT_LIST[@]}"; do
    echo "Running: object=${obj_name}, lights=${num_lights}"
    python launch.py +conf=diligentmv \
      conf.dataset.obj_name="${obj_name}" \
      conf.dataset.train.view_light_index_fname=view_20_light_${num_lights} \
      conf.dataset.val.view_light_index_fname=view_20_light_${num_lights}
  done
done
