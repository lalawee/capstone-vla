if [ -f /workspace/.local/bin/env ]; then
    source /workspace/.local/bin/env
    export PATH="/workspace/.local/bin:$PATH"
    export HF_HOME=/workspace/hf_cache
    export IMAGINAIRE_OUTPUT_ROOT=/workspace/cosmos_outputs
    export UV_CACHE_DIR=/workspace/.uv_cache   # <-- add this
    echo "✅ uv environment loaded."
else
    echo "❌ uv env file not found at /workspace/.local/bin/env"
fi