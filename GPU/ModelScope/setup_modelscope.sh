#!/usr/bin/env bash
set -euo pipefail

# Setup and launch ComfyUI + Qwen-Image-2.1 on ModelScope PAI-DSW (A10 24GB VRAM, 32GB RAM, 100GB /mnt/workspace).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_SHARED="$(cd "$SCRIPT_DIR/.." 2>/dev/null && pwd)/Wagi/qwen_image_2_1"
if [ -n "${SHARED_DIR:-}" ]; then
  WAGI_DIR="$SHARED_DIR"
elif [ -d "/mnt/workspace/qwen21_shared" ]; then
  WAGI_DIR="/mnt/workspace/qwen21_shared"
elif [ -d "$HOME/qwen21_shared" ]; then
  WAGI_DIR="$HOME/qwen21_shared"
elif [ -d "$DEFAULT_REPO_SHARED" ]; then
  WAGI_DIR="$DEFAULT_REPO_SHARED"
else
  echo "[ERROR] Cannot find Wagi/qwen_image_2_1. Copy it to /mnt/workspace/qwen21_shared or set SHARED_DIR." >&2
  exit 1
fi

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/workspace}"
if [ ! -d "$WORKSPACE_ROOT" ]; then
  WORKSPACE_ROOT="$HOME"
fi

COMFY_DIR="${COMFY_DIR:-$WORKSPACE_ROOT/ComfyUI}"
VENV_DIR="${VENV_DIR:-$WORKSPACE_ROOT/comfy_venv}"
PROFILE="${QWEN21_MODEL_PROFILE:-full-int8}"
PORT="${COMFY_PORT:-8188}"
START_TUNNEL="${START_TUNNEL:-1}"

# Use mainland China HF mirror by default on ModelScope DSW; override HF_ENDPOINT if direct access is fast.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_CACHE="$WORKSPACE_ROOT/.cache/hf_cache"
export HF_HUB_DISABLE_TELEMETRY=1
export QWEN21_ALLOW_OFFLINE_TEMPLATES=1

echo "[1/5] Preparing persistent Python venv in $VENV_DIR..."
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv --system-site-packages "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[2/5] Installing/updating ComfyUI in $COMFY_DIR..."
if [ ! -d "$COMFY_DIR/.git" ]; then
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
fi

FILTERED_REQS="$(mktemp)"
trap 'rm -f "$FILTERED_REQS"' EXIT
if python3 -c "import torch; assert torch.cuda.is_available(); print('[OK] Existing CUDA PyTorch:', torch.__version__)" 2>/dev/null; then
  grep -viE '^(torch|torchvision|torchaudio)([<>=!~ ]|$)' "$COMFY_DIR/requirements.txt" > "$FILTERED_REQS"
  pip install -r "$FILTERED_REQS"
else
  pip install -r "$COMFY_DIR/requirements.txt"
fi
pip install -U "huggingface-hub[hf_xet]"

echo "[3/5] Downloading Qwen-Image-2.1 (profile=$PROFILE, HF_ENDPOINT=$HF_ENDPOINT)..."
python3 "$WAGI_DIR/common.py" \
  --model-dir "$COMFY_DIR/models" \
  --workflow-dir "$COMFY_DIR/user/default/workflows" \
  --profile "$PROFILE"

echo "[4/5] Starting ComfyUI on 0.0.0.0:$PORT..."
cd "$COMFY_DIR"
if [ "$START_TUNNEL" = "1" ]; then
  python3 main.py --listen 0.0.0.0 --port "$PORT" --preview-method taesd --disable-auto-launch &
  COMFY_PID=$!
  trap 'kill "$COMFY_PID" 2>/dev/null || true' EXIT

  CF_BIN="${WORKSPACE_ROOT}/bin/cloudflared"
  if ! command -v cloudflared >/dev/null 2>&1 && [ ! -x "$CF_BIN" ]; then
    mkdir -p "${WORKSPACE_ROOT}/bin"
    echo "[5/5] Downloading cloudflared tunnel..."
    curl -fSL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" -o "$CF_BIN" || true
    chmod +x "$CF_BIN" 2>/dev/null || true
  fi

  if command -v cloudflared >/dev/null 2>&1; then
    exec cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$PORT"
  elif [ -x "$CF_BIN" ]; then
    exec "$CF_BIN" tunnel --no-autoupdate --url "http://127.0.0.1:$PORT"
  else
    echo "[WARNING] cloudflared unavailable; use ModelScope PAI-DSW Port Forwarding for port $PORT." >&2
    wait "$COMFY_PID"
  fi
else
  exec python3 main.py --listen 0.0.0.0 --port "$PORT" --preview-method taesd --disable-auto-launch
fi
