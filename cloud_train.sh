#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# cloud_train.sh
# RunPod launcher: download sim_mixed data, install deps,
# pick a model + ratio, train GR00T.
#
# Usage:
#   bash cloud_train.sh                  # interactive menu
#   bash cloud_train.sh 80               # skip menu, train 80:20
#   bash cloud_train.sh 60 --max-steps 15000  # ratio + extra args
# ============================================================

WORKSPACE="/workspace/capstone-vla"
GROOT_DIR="${WORKSPACE}/third_party/gr00t"
GROOT_VENV="${GROOT_DIR}/.venv/bin/activate"
DOWNLOAD_SCRIPT="${WORKSPACE}/download_dataset_folder.py"
DATASETS_DIR="${WORKSPACE}/datasets"
CHKPT_DIR="${GROOT_DIR}/chkpt_sim"

HF_REPO="Lusmse/syn_realDataset"
BASE_MODEL="nvidia/GR00T-N1.6-3B"
MODALITY_CFG="${GROOT_DIR}/examples/kuavoV4Pro/kuavoV4Pro_config.py"

ALL_RATIOS="80_20 60_40 50_50 40_60"

# ============================================================
# 0. Source uv + activate gr00t venv
# ============================================================
echo ""
echo "================================================================"
echo "  STEP 0: Environment setup"
echo "================================================================"

cd "$WORKSPACE"

if [[ -f ./source_uv.sh ]]; then
  source ./source_uv.sh
  echo "  ✓ uv sourced"
else
  echo "  ✗ source_uv.sh not found"
  exit 1
fi

if [[ -f "$GROOT_VENV" ]]; then
  source "$GROOT_VENV"
  echo "  ✓ gr00t venv activated ($(python --version))"
else
  echo "  ✗ gr00t .venv not found at ${GROOT_VENV}"
  exit 1
fi

# ============================================================
# 1. Install apt dependencies
# ============================================================
echo ""
echo "================================================================"
echo "  STEP 1: Installing apt dependencies"
echo "================================================================"

apt-get update -qq
apt-get install -y -qq python3-dev ffmpeg > /dev/null 2>&1
echo "  ✓ python3-dev installed"
echo "  ✓ ffmpeg installed"

if python -c "import torchcodec" 2>/dev/null; then
  echo "  ✓ torchcodec OK"
else
  echo "  ⚠ torchcodec import failed — training may error on video loading"
fi

# SSH key setup
if [[ -f /workspace/.ssh/id_ed25519 ]]; then
  mkdir -p /root/.ssh
  cp /workspace/.ssh/id_* /root/.ssh/ 2>/dev/null || true
  cp /workspace/.ssh/config /root/.ssh/ 2>/dev/null || true
  chmod 600 /root/.ssh/id_* 2>/dev/null || true
  echo "  ✓ SSH keys copied"
fi

# ============================================================
# 2. Check / download datasets
# ============================================================
echo ""
echo "================================================================"
echo "  STEP 2: Checking sim_mixed datasets"
echo "================================================================"

mkdir -p "$DATASETS_DIR"

AVAILABLE=()
MISSING=()

for RATIO in $ALL_RATIOS; do
  FOLDER="sim_mixed_${RATIO}"
  DS_PATH="${DATASETS_DIR}/${FOLDER}"

  if [[ -f "${DS_PATH}/meta/info.json" ]]; then
    EPS=$(python -c "import json; print(json.load(open('${DS_PATH}/meta/info.json'))['total_episodes'])" 2>/dev/null || echo "?")
    echo "  ✓ ${FOLDER}  (${EPS} episodes)"
    AVAILABLE+=("$RATIO")
  else
    echo "  ✗ ${FOLDER}  — not found"
    MISSING+=("$RATIO")
  fi
done

# Download missing datasets
if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo ""
  echo "  Downloading missing datasets from ${HF_REPO}..."

  if [[ ! -f "$DOWNLOAD_SCRIPT" ]]; then
    echo "  ✗ Download script not found at: ${DOWNLOAD_SCRIPT}"
    echo "    Place download_dataset_folder.py there, or download datasets manually."
    exit 1
  fi

  FOLDERS_CSV=$(printf "sim_mixed_%s," "${MISSING[@]}")
  FOLDERS_CSV="${FOLDERS_CSV%,}"

  python "$DOWNLOAD_SCRIPT" \
    --dataset "$HF_REPO" \
    --folder "$FOLDERS_CSV" \
    --output "$DATASETS_DIR" \
    --flat

  # Re-check
  for RATIO in "${MISSING[@]}"; do
    FOLDER="sim_mixed_${RATIO}"
    DS_PATH="${DATASETS_DIR}/${FOLDER}"
    if [[ -f "${DS_PATH}/meta/info.json" ]]; then
      echo "  ✓ ${FOLDER} downloaded"
      AVAILABLE+=("$RATIO")
    else
      echo "  ✗ ${FOLDER} download FAILED"
    fi
  done
fi

if [[ ${#AVAILABLE[@]} -eq 0 ]]; then
  echo "  No datasets available. Exiting."
  exit 1
fi

# ============================================================
# 3. Pick ratio to train
# ============================================================
echo ""
echo "================================================================"
echo "  STEP 3: Select training configuration"
echo "================================================================"

SELECTED_RATIO=""
EXTRA_ARGS=""

if [[ $# -ge 1 && "$1" =~ ^[0-9]+$ ]]; then
  PCT="$1"
  shift
  EXTRA_ARGS="${*:-}"

  case "$PCT" in
    80) SELECTED_RATIO="80_20" ;;
    60) SELECTED_RATIO="60_40" ;;
    50) SELECTED_RATIO="50_50" ;;
    40) SELECTED_RATIO="40_60" ;;
    *)  echo "  Unknown ratio: ${PCT}. Use 80, 60, 50, or 40."; exit 1 ;;
  esac

  echo "  Selected via CLI: sim_mixed_${SELECTED_RATIO}"
else
  echo ""
  echo "  Available datasets:"
  IDX=1
  declare -A MENU
  for RATIO in ${ALL_RATIOS}; do
    for A in "${AVAILABLE[@]}"; do
      if [[ "$A" == "$RATIO" ]]; then
        FOLDER="sim_mixed_${RATIO}"
        DS_PATH="${DATASETS_DIR}/${FOLDER}"
        EPS=$(python -c "import json; print(json.load(open('${DS_PATH}/meta/info.json'))['total_episodes'])" 2>/dev/null || echo "?")
        H="${RATIO%%_*}"
        M="${RATIO##*_}"
        echo "    [${IDX}] ${FOLDER}  (${H}% human, ${M}% mimic, ${EPS} eps)"
        MENU[$IDX]="$RATIO"
        IDX=$((IDX + 1))
        break
      fi
    done
  done

  echo ""
  read -p "  Pick a number: " PICK
  SELECTED_RATIO="${MENU[$PICK]:-}"

  if [[ -z "$SELECTED_RATIO" ]]; then
    echo "  Invalid selection. Exiting."
    exit 1
  fi
fi

FOLDER="sim_mixed_${SELECTED_RATIO}"
DS_PATH="${DATASETS_DIR}/${FOLDER}"
RUN_NAME="run_sim_${SELECTED_RATIO}"
OUTPUT_DIR="${CHKPT_DIR}/${RUN_NAME}"

echo ""
echo "  Dataset:    ${DS_PATH}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  Run name:   ${RUN_NAME}"

# ============================================================
# 4. Train
# ============================================================
echo ""
echo "================================================================"
echo "  STEP 4: Launching GR00T fine-tuning"
echo "  Model:   ${BASE_MODEL}"
echo "  Dataset: ${FOLDER}"
echo "================================================================"

cd "$GROOT_DIR"

CUDA_VISIBLE_DEVICES=0 python \
  gr00t/experiment/launch_finetune.py \
  --base-model-path "$BASE_MODEL" \
  --dataset-path "$DS_PATH" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$MODALITY_CFG" \
  --num-gpus 1 \
  --output-dir "$OUTPUT_DIR" \
  --save-total-limit 3 \
  --save-steps 3000 \
  --max-steps 10000 \
  --use-wandb \
  --global-batch-size 32 \
  --tune-llm \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --dataloader-num-workers 4 \
  $EXTRA_ARGS

echo ""
echo "================================================================"
echo "  ✓ Training complete: ${RUN_NAME}"
echo "  Checkpoint: ${OUTPUT_DIR}"
echo "================================================================"