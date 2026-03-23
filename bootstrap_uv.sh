#!/usr/bin/env bash
# bootstrap_uv.sh
#
# Installs uv to persistent storage and sets up all environment variables
# needed for Pi0 data preparation and fine-tuning.
#
# Run this on a fresh instance or after uv gets wiped:
#   bash /workspace/capstone-vla/bootstrap_uv.sh
#
# It is SAFE to run multiple times (idempotent).

set -e

PERSISTENT=/workspace
UV_BIN="$PERSISTENT/.local/bin/uv"
UV_CACHE="$PERSISTENT/.uv_cache"
OPENPI_DIR="$PERSISTENT/capstone-vla/third_party/openpi"
DATA_DIR="$PERSISTENT/capstone-vla/dataset_tools/data"
CHECKPOINTS_DIR="$PERSISTENT/checkpoints"

echo "============================================================"
echo " Bootstrap: uv + Pi0 environment"
echo " Persistent storage: $PERSISTENT"
echo "============================================================"
echo ""

# ---------------------------------------------------------------
# Step 1: Install uv to persistent storage
# ---------------------------------------------------------------
echo ">>> Step 1: Installing uv to persistent storage..."

export UV_INSTALL_DIR="$PERSISTENT/.local/bin"
export UV_CACHE_DIR="$UV_CACHE"
mkdir -p "$UV_INSTALL_DIR" "$UV_CACHE_DIR"

if [ -f "$UV_BIN" ]; then
    echo "  uv already installed: $($UV_BIN --version)"
else
    echo "  Downloading and installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "  uv installed: $($UV_BIN --version)"
fi

export PATH="$PERSISTENT/.local/bin:$PATH"
echo ""

# ---------------------------------------------------------------
# Step 2: Write persistent env setup script
#         This goes on /workspace so it survives instance restarts.
#         Source it at the start of every session.
# ---------------------------------------------------------------
echo ">>> Step 2: Writing persistent env script to $PERSISTENT/setup_instance.sh..."

cat > "$PERSISTENT/setup_instance.sh" << ENVSCRIPT
#!/usr/bin/env bash
# setup_instance.sh
# Source this at the start of every session on a fresh instance:
#   source /workspace/setup_instance.sh
#
# Or add to ~/.bashrc:
#   echo 'source /workspace/setup_instance.sh' >> ~/.bashrc

PERSISTENT=/workspace

# --- uv ---
export UV_INSTALL_DIR="\$PERSISTENT/.local/bin"
export UV_CACHE_DIR="\$PERSISTENT/.uv_cache"
export PATH="\$PERSISTENT/.local/bin:\$PATH"

# --- LeRobot ---
unset LEROBOT_HOME
export HF_LEROBOT_HOME="\$PERSISTENT/capstone-vla/dataset_tools/data"
export LEROBOT_VIDEO_BACKEND=pyav

# --- openpi / JAX ---
export OPENPI_DATA_HOME="\$PERSISTENT/checkpoints/openpi_assets"
export TMPDIR="\$PERSISTENT/tmp"
export HF_HOME="\$PERSISTENT/hf_cache"

# --- Create dirs if missing ---
mkdir -p "\$OPENPI_DATA_HOME" "\$TMPDIR" "\$HF_HOME"

echo "✅ Instance environment loaded."
echo "   uv:                \$(uv --version 2>/dev/null || echo 'NOT FOUND - run bootstrap_uv.sh')"
echo "   HF_LEROBOT_HOME:   \$HF_LEROBOT_HOME"
echo "   OPENPI_DATA_HOME:  \$OPENPI_DATA_HOME"
echo "   TMPDIR:            \$TMPDIR"
ENVSCRIPT

chmod +x "$PERSISTENT/setup_instance.sh"
echo "  Written to $PERSISTENT/setup_instance.sh"
echo ""

# ---------------------------------------------------------------
# Step 3: Source the env script now
# ---------------------------------------------------------------
echo ">>> Step 3: Loading environment..."
source "$PERSISTENT/setup_instance.sh"
echo ""

# ---------------------------------------------------------------
# Step 4: Add to ~/.bashrc so it auto-loads every session
# ---------------------------------------------------------------
echo ">>> Step 4: Adding to ~/.bashrc..."
BASHRC="$HOME/.bashrc"
MARKER="# === capstone Pi0 bootstrap ==="

if grep -q "$MARKER" "$BASHRC" 2>/dev/null; then
    echo "  Already in ~/.bashrc, skipping."
else
    echo "" >> "$BASHRC"
    echo "$MARKER" >> "$BASHRC"
    echo "source $PERSISTENT/setup_instance.sh" >> "$BASHRC"
    echo "  Added 'source $PERSISTENT/setup_instance.sh' to ~/.bashrc"
fi
echo ""

# ---------------------------------------------------------------
# Step 5: Reinstall openpi venv if missing
# ---------------------------------------------------------------
echo ">>> Step 5: Checking openpi venv..."
if [ -d "$OPENPI_DIR/.venv" ]; then
    echo "  .venv exists at $OPENPI_DIR/.venv"
else
    echo "  .venv missing — reinstalling..."
    cd "$OPENPI_DIR"
    uv sync
    echo "  .venv reinstalled."
fi
echo ""

# ---------------------------------------------------------------
# Done
# ---------------------------------------------------------------
echo "============================================================"
echo " Bootstrap complete!"
echo ""
echo " On every NEW instance, run:"
echo "   source /workspace/setup_instance.sh"
echo ""
echo " Or if uv is wiped, re-run:"
echo "   bash /workspace/capstone-vla/bootstrap_uv.sh"
echo ""
echo " To start training:"
echo "   cd $OPENPI_DIR"
echo "   WANDB_MODE=disabled uv run scripts/train.py \\"
echo "     pi05_kuavo_armhand_26d_full_finetune \\"
echo "     --exp-name=kuavo_pouring_v1 --overwrite"
echo "============================================================"