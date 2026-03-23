#!/usr/bin/env bash
# setup_instance.sh
# Source this at the start of every session on a fresh instance:
#   source /workspace/setup_instance.sh
#
# Or add to ~/.bashrc:
#   echo 'source /workspace/setup_instance.sh' >> ~/.bashrc

PERSISTENT=/workspace

# --- uv ---
export UV_INSTALL_DIR="$PERSISTENT/.local/bin"
export UV_CACHE_DIR="$PERSISTENT/.uv_cache"
export PATH="$PERSISTENT/.local/bin:$PATH"

# --- LeRobot ---
unset LEROBOT_HOME
export HF_LEROBOT_HOME="$PERSISTENT/capstone-vla/dataset_tools/data"
export LEROBOT_VIDEO_BACKEND=pyav

# --- openpi / JAX ---
export OPENPI_DATA_HOME="$PERSISTENT/checkpoints/openpi_assets"
export TMPDIR="$PERSISTENT/tmp"
export HF_HOME="$PERSISTENT/hf_cache"

# --- Create dirs if missing ---
mkdir -p "$OPENPI_DATA_HOME" "$TMPDIR" "$HF_HOME"

echo "✅ Instance environment loaded."
echo "   uv:                $(uv --version 2>/dev/null || echo 'NOT FOUND - run bootstrap_uv.sh')"
echo "   HF_LEROBOT_HOME:   $HF_LEROBOT_HOME"
echo "   OPENPI_DATA_HOME:  $OPENPI_DATA_HOME"
echo "   TMPDIR:            $TMPDIR"
