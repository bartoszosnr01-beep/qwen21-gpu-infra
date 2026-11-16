#!/usr/bin/env bash
set -euo pipefail

# Setup and launch ComfyUI + Qwen-Image-2.1 (full-bf16) on AMD Developer Cloud (MI300X ROCm).
# Requires GPU/Wagi/qwen_image_2_1 files available at SHARED_DIR (default: ~/qwen21_shared).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_SHARED="$(cd "$SCRIPT_DIR/.." 2>/dev/null && pwd)/Wagi/qwen_image_2_1"
if [ -n "${SHARED_DIR:-}" ]; then
  WAGI_DIR="$SHARED_DIR"
elif [ -d "$HOME/qwen21_shared" ]; then
  WAGI_DIR="$HOME/qwen21_shared"
elif [ -d "$DEFAULT_REPO_SHARED" ]; then
  WAGI_DIR="$DEFAULT_REPO_SHARED"
else
  echo "[ERROR] Cannot find Wagi/qwen_image_2_1 directory. Set SHARED_DIR or copy it to ~/qwen21_shared." >&2
  exit 1
fi

COMFY_DIR="${COMFY_DIR:-$HOME/ComfyUI}"
VENV_DIR="${VENV_DIR:-$HOME/comfy_rocm_venv}"
PROFILE="${QWEN21_MODEL_PROFILE:-full-bf16}"
ROCM_INDEX_URL="${ROCM_INDEX_URL:-https://download.pytorch.org/whl/rocm6.3}"
PORT="${COMFY_PORT:-8188}"

echo "[1/5] Preparing Python environment ($VENV_DIR)..."
if [ ! -d "$VENV_DIR" ]; then
  # Inherit system site-packages if ROCm PyTorch is pre-installed on the AMD VM.
  python3 -m venv --system-site-packages "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[2/5] Checking ROCm PyTorch..."
if python3 -c "import torch; assert getattr(torch.version, 'hip', None) and torch.cuda.is_available(); print('[OK] Preinstalled ROCm PyTorch:', torch.__version__, 'HIP:', torch.version.hip, 'Device:', torch.cuda.get_device_name(0))" 2>/dev/null; then
  :
else
  echo "[INFO] Installing ROCm PyTorch from $ROCM_INDEX_URL ..."
  pip install --upgrade pip
  pip install --pre torch torchvision torchaudio --index-url "$ROCM_INDEX_URL"
fi

echo "[3/5] Installing/updating ComfyUI in $COMFY_DIR ..."
if [ ! -d "$COMFY_DIR/.git" ]; then
  git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
fi

# Strip torch/torchvision/torchaudio lines so pip does not overwrite the ROCm PyTorch build with CUDA wheels.
FILTERED_REQS="$(mktemp)"
trap 'rm -f "$FILTERED_REQS"' EXIT
grep -viE '^(torch|torchvision|torchaudio)([<>=!~ ]|$)' "$COMFY_DIR/requirements.txt" > "$FILTERED_REQS"
pip install -r "$FILTERED_REQS"
pip install -U "huggingface-hub[hf_xet]"

echo "[4/5] Downloading Qwen-Image-2.1 weights (profile=$PROFILE) and seeding workflows..."
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DISABLE_TELEMETRY=1
python3 "$WAGI_DIR/common.py" \
  --model-dir "$COMFY_DIR/models" \
  --workflow-dir "$COMFY_DIR/user/default/workflows" \
  --profile "$PROFILE"

echo "[5/5] Launching ComfyUI bound to 127.0.0.1:$PORT (use SSH tunnel from Windows)..."
echo "      Run locally in PowerShell: ssh -N -L ${PORT}:127.0.0.1:${PORT} <user>@<vm-ip>"
cd "$COMFY_DIR"
exec python3 main.py \
  --listen 127.0.0.1 \
  --port "$PORT" \
  --preview-method taesd \
  --disable-auto-launch
