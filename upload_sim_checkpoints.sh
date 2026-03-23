#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# upload_sim_checkpoints.sh
# Uploads GR00T sim_mixed training checkpoints to HuggingFace.
#
# Switches from gr00t uv venv to conda kdc environment
# before running upload_checkpoints.py.
#
# Usage:
#   bash upload_sim_checkpoints.sh 80        # upload run_sim_80_20
#   bash upload_sim_checkpoints.sh 60        # upload run_sim_60_40
#   bash upload_sim_checkpoints.sh all       # upload all sim runs
# ============================================================

WORKSPACE="/workspace/capstone-vla"
GROOT_DIR="${WORKSPACE}/third_party/gr00t"
CHKPT_SIM="${GROOT_DIR}/chkpt_sim"
UPLOAD_SCRIPT="${WORKSPACE}/upload_checkpoints.py"
HF_REPO="Lusmse/capstone-vla-checkpoints"

# ============================================================
# 0. Switch environment: deactivate uv venv → conda kdc
# ============================================================
echo ""
echo "================================================================"
echo "  Setting up environment (conda kdc)"
echo "================================================================"

# Deactivate any active venv
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo "  Deactivating venv: ${VIRTUAL_ENV}"
  deactivate 2>/dev/null || true
fi

# Source conda (use conda.sh, not activate — activate leaks $@ as env name)
if [[ -f /workspace/miniconda3/etc/profile.d/conda.sh ]]; then
  source /workspace/miniconda3/etc/profile.d/conda.sh
  echo "  ✓ conda sourced"
elif [[ -f /workspace/miniconda3/bin/conda ]]; then
  eval "$(/workspace/miniconda3/bin/conda shell.bash hook)"
  echo "  ✓ conda sourced (via hook)"
else
  echo "  ✗ miniconda3 not found at /workspace/miniconda3"
  exit 1
fi

conda activate kdc
echo "  ✓ conda kdc activated ($(python --version))"

# ============================================================
# 1. Parse user input
# ============================================================
if [[ $# -lt 1 ]]; then
  echo ""
  echo "Usage:  bash upload_sim_checkpoints.sh <ratio|all>"
  echo ""
  echo "  bash upload_sim_checkpoints.sh 80     # upload run_sim_80_20"
  echo "  bash upload_sim_checkpoints.sh 60     # upload run_sim_60_40"
  echo "  bash upload_sim_checkpoints.sh 50     # upload run_sim_50_50"
  echo "  bash upload_sim_checkpoints.sh 40     # upload run_sim_40_60"
  echo "  bash upload_sim_checkpoints.sh all    # upload all sim runs"
  exit 0
fi

SELECTION="$1"

# ============================================================
# 2. Show what's available
# ============================================================
echo ""
echo "================================================================"
echo "  Available sim checkpoints in ${CHKPT_SIM}"
echo "================================================================"

if [[ ! -d "$CHKPT_SIM" ]]; then
  echo "  ✗ chkpt_sim directory not found"
  exit 1
fi

for D in "$CHKPT_SIM"/run_sim_*; do
  [[ -d "$D" ]] || continue
  NAME=$(basename "$D")
  FILE_COUNT=$(find "$D" -type f | wc -l)
  SIZE_MB=$(du -sm "$D" | cut -f1)
  echo "  ${NAME}  (${FILE_COUNT} files, ${SIZE_MB} MB)"
done

# ============================================================
# 3. Upload
# ============================================================
echo ""
echo "================================================================"
echo "  Uploading to ${HF_REPO}"
echo "================================================================"

if [[ "$SELECTION" == "all" ]]; then
  # upload_checkpoints.py uploads all runs under chkpt_sim
  # We temporarily point its config at chkpt_sim
  echo "  Uploading ALL sim runs..."
  python "$UPLOAD_SCRIPT" \
    --repo "$HF_REPO" \
    --base "$WORKSPACE" \
    --model gr00t

else
  # Single ratio — map to run dir name
  case "$SELECTION" in
    80) RUN_DIR="run_sim_80_20" ;;
    60) RUN_DIR="run_sim_60_40" ;;
    50) RUN_DIR="run_sim_50_50" ;;
    40) RUN_DIR="run_sim_40_60" ;;
    *)  echo "  Unknown ratio: ${SELECTION}. Use 80, 60, 50, 40, or all."; exit 1 ;;
  esac

  RUN_PATH="${CHKPT_SIM}/${RUN_DIR}"
  if [[ ! -d "$RUN_PATH" ]]; then
    echo "  ✗ ${RUN_DIR} not found at ${CHKPT_SIM}"
    exit 1
  fi

  FILE_COUNT=$(find "$RUN_PATH" -type f | wc -l)
  SIZE_MB=$(du -sm "$RUN_PATH" | cut -f1)
  echo "  Run:    ${RUN_DIR}"
  echo "  Files:  ${FILE_COUNT}"
  echo "  Size:   ${SIZE_MB} MB"
  echo ""

  # Upload just this one run folder
  # We use HfApi.upload_folder directly via a small inline python
  python - <<PYEOF
import sys
sys.path.insert(0, "${WORKSPACE}")
from huggingface_hub import HfApi, create_repo
import os

token_file = "${WORKSPACE}/.hf_token"
token = os.environ.get("HF_TOKEN", "").strip()
if not token and os.path.exists(token_file):
    token = open(token_file).read().strip()
if not token:
    print("[ERROR] No HF token found. Set HF_TOKEN or run upload_checkpoints.py --reauth first.")
    sys.exit(1)

api = HfApi(token=token)
user = api.whoami()
print(f"  [AUTH] Logged in as: {user['name']}")

create_repo(repo_id="${HF_REPO}", repo_type="model", private=False, exist_ok=True, token=token)

hf_prefix = "gr00t/${RUN_DIR}"
print(f"  Uploading ${RUN_PATH} -> {hf_prefix}/")

api.upload_folder(
    folder_path="${RUN_PATH}",
    path_in_repo=hf_prefix,
    repo_id="${HF_REPO}",
    repo_type="model",
    commit_message="Add ${RUN_DIR} (sim-mixed checkpoint)",
)
print(f"  [OK] Uploaded to https://huggingface.co/${HF_REPO}/tree/main/{hf_prefix}")
PYEOF

fi

echo ""
echo "================================================================"
echo "  ✓ Upload complete"
echo "  https://huggingface.co/${HF_REPO}"
echo "================================================================"