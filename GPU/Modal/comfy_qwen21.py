"""Modal app running ComfyUI with Qwen-Image-2.1 on L4/A10 GPUs using shared Wagi definitions."""

import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import modal

REMOTE_SHARED_DIR = "/root/qwen21_shared"
LOCAL_SHARED_DIR = Path(__file__).resolve().parent.parent / "Wagi" / "qwen_image_2_1"
SHARED_DIR = Path(REMOTE_SHARED_DIR) if Path(REMOTE_SHARED_DIR).exists() else LOCAL_SHARED_DIR
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

import common  # noqa: E402

APP_NAME = "qwen21-comfyui"
VOLUME_NAME = "qwen21-models-vol"
USER_VOLUME_NAME = "qwen21-user-vol"
COMFY_WORKSPACE = "/root/comfy"
MODEL_DIR = f"{COMFY_WORKSPACE}/models"
USER_DIR = f"{COMFY_WORKSPACE}/user"
WORKFLOW_DIR = f"{USER_DIR}/default/workflows"
USER_VOLUME_DIR = "/root/comfy_user_vol"
USER_WORKFLOW_DIR = f"{USER_VOLUME_DIR}/default/workflows"

# Free-tier optimized defaults: avoid L40S fallback (~2.4x cost), trim idle window, and keep 24 GiB RAM for PE.
DEFAULT_GPUS = [
    gpu.strip()
    for gpu in os.environ.get("QWEN21_MODAL_GPUS", "L4,A10").split(",")
    if gpu.strip()
]
DEFAULT_PROFILE = os.environ.get("QWEN21_MODEL_PROFILE", "full-int8")
DEFAULT_MEMORY_MIB = int(os.environ.get("QWEN21_MODAL_MEMORY", "24576"))
DEFAULT_SCALEDOWN_SECONDS = int(os.environ.get("QWEN21_MODAL_SCALEDOWN", "180"))
ENABLE_CACHE_NONE = os.environ.get("QWEN21_CACHE_NONE", "0") == "1"

MANIFEST = common.load_manifest(SHARED_DIR / "models.json")
WORKFLOW_SEED_COMMANDS = [
    f"wget -q -O {WORKFLOW_DIR}/{filename} {url}"
    for filename, url in MANIFEST["workflow_templates"].items()
]

model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
user_volume = modal.Volume.from_name(USER_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "libgl1", "libglib2.0-0")
    .uv_pip_install("comfy-cli==1.26.0", "huggingface-hub[hf_xet]", "uv")
    .run_commands(
        f"comfy --skip-prompt --workspace={COMFY_WORKSPACE} install --nvidia --fast-deps"
    )
    .run_commands(
        f"mkdir -p {WORKFLOW_DIR}",
        *WORKFLOW_SEED_COMMANDS,
        f"rm -rf {MODEL_DIR}/*",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_CACHE": "/tmp/hf_cache",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    .add_local_dir(
        LOCAL_SHARED_DIR,
        remote_path=REMOTE_SHARED_DIR,
        ignore=["weights/**", "weights", "__pycache__/**", "*.pyc"],
    )
)

app = modal.App(name=APP_NAME, image=image)


@app.function(
    cpu=4.0,
    memory=8192,
    timeout=1800,
    volumes={MODEL_DIR: model_volume},
)
def download_models(profile: str = DEFAULT_PROFILE) -> None:
    """Download Qwen-Image-2.1 weights onto the persistent Modal Volume using CPU."""
    model_volume.reload()
    if common.ensure_models(MODEL_DIR, profile=profile, commit_cb=model_volume.commit):
        print("[OK] Committed Qwen-Image-2.1 models to Modal Volume.", flush=True)


WS_FIX_DIR = "/tmp/comfy_ws_fix"
WS_FIX_SCRIPT = """import importlib
import os
import sys
import threading
import urllib.parse

_fix_dir = os.path.normcase(os.path.dirname(os.path.abspath(__file__)))
_self_mod = sys.modules.get("sitecustomize")
_saved_path = list(sys.path)
_added = []
try:
    sys.path = [
        p for p in _saved_path
        if not (p and os.path.normcase(os.path.abspath(p)) == _fix_dir)
    ]
    sys.modules.pop("sitecustomize", None)
    try:
        importlib.import_module("sitecustomize")
    except ImportError:
        pass
    _added = [p for p in sys.path if p not in _saved_path]
finally:
    sys.path = _saved_path + _added
    if _self_mod is not None:
        sys.modules["sitecustomize"] = _self_mod

try:
    import aiohttp.web
    import yarl

    _orig_ws_init = aiohttp.web.WebSocketResponse.__init__

    def _ws_init_uncompressed(self, *args, **kwargs):
        # Modal web_server proxies WebSockets with compress=0, so permessage-deflate frames break /ws.
        kwargs["compress"] = False
        return _orig_ws_init(self, *args, **kwargs)

    aiohttp.web.WebSocketResponse.__init__ = _ws_init_uncompressed

    # Modal's reverse proxy decodes %2F into '/' before forwarding, which breaks ComfyUI's single-segment /userdata/{file} route (405).
    _orig_resolve = aiohttp.web.UrlDispatcher.resolve

    def _normalize_userdata_raw_path(raw_path, method):
        path_part, sep, query = raw_path.partition("?")
        for prefix in ("/api/userdata/", "/userdata/"):
            if not path_part.startswith(prefix):
                continue
            rest = path_part[len(prefix):]
            if not rest:
                return None
            if method.upper() == "POST" and "/move/" in rest:
                src_part, dst_part = rest.split("/move/", 1)
                norm_rest = (
                    urllib.parse.quote(urllib.parse.unquote(src_part), safe="")
                    + "/move/"
                    + urllib.parse.quote(urllib.parse.unquote(dst_part), safe="")
                )
            else:
                norm_rest = urllib.parse.quote(urllib.parse.unquote(rest), safe="")
            new_raw = f"{prefix}{norm_rest}{sep}{query}"
            return new_raw if new_raw != raw_path else None
        return None

    async def _resolve_modal_userdata(self, request):
        try:
            new_raw = _normalize_userdata_raw_path(request.raw_path, request.method)
            if new_raw is not None:
                cloned = request.clone(rel_url=yarl.URL(new_raw, encoded=True))
                return await _orig_resolve(self, cloned)
        except Exception:
            pass
        return await _orig_resolve(self, request)

    aiohttp.web.UrlDispatcher.resolve = _resolve_modal_userdata

    # Persist user workflows and settings on the Modal Volume right after successful HTTP mutations.
    _user_vol_name = os.environ.get("COMFY_USER_VOLUME_NAME", "")
    if _user_vol_name:
        _vol_lock = threading.Lock()
        _vol_timer = None
        _user_vol = None

        def _commit_user_volume_now():
            global _vol_timer, _user_vol
            with _vol_lock:
                _vol_timer = None
            try:
                if _user_vol is None:
                    import modal
                    _user_vol = modal.Volume.from_name(_user_vol_name)
                _user_vol.commit()
            except Exception as exc:
                print(f"[WARNING] Failed to commit {_user_vol_name}: {exc}", flush=True)

        def _schedule_user_vol_commit(delay=0.25):
            global _vol_timer
            with _vol_lock:
                if _vol_timer is not None:
                    _vol_timer.cancel()
                timer = threading.Timer(delay, _commit_user_volume_now)
                timer.daemon = True
                _vol_timer = timer
                timer.start()

        _user_write_prefixes = (
            "/userdata/",
            "/api/userdata/",
            "/settings",
            "/api/settings",
            "/users",
            "/api/users",
        )
        _orig_app_handle = aiohttp.web.Application._handle

        async def _app_handle_user_vol_sync(self, request):
            is_user_write = (
                request.method in ("POST", "PUT", "DELETE")
                and request.path.startswith(_user_write_prefixes)
            )
            try:
                response = await _orig_app_handle(self, request)
                if is_user_write and getattr(response, "status", 500) < 400:
                    _schedule_user_vol_commit()
                return response
            except aiohttp.web.HTTPException as exc:
                if is_user_write and exc.status < 400:
                    _schedule_user_vol_commit()
                raise

        aiohttp.web.Application._handle = _app_handle_user_vol_sync
except Exception as exc:
    print(f"[WARNING] Failed to install ComfyUI runtime fixes: {exc}", flush=True)
"""


def _with_uncompressed_ws_env(base_env: Mapping[str, str], fix_dir: str = WS_FIX_DIR) -> dict[str, str]:
    """Install a Python sitecustomize hook for Modal WebSocket, userdata routing, and volume commits."""
    fix_path = Path(fix_dir)
    fix_path.mkdir(parents=True, exist_ok=True)
    (fix_path / "sitecustomize.py").write_text(WS_FIX_SCRIPT, encoding="utf-8")

    env = dict(base_env)
    env["COMFY_USER_VOLUME_NAME"] = USER_VOLUME_NAME
    existing = env.get("PYTHONPATH", "")
    parts = [part for part in existing.split(os.pathsep) if part] if existing else []
    if fix_dir not in parts:
        env["PYTHONPATH"] = os.pathsep.join([fix_dir, *parts])
    return env


@app.function(
    gpu=DEFAULT_GPUS,
    cpu=2.0,
    memory=DEFAULT_MEMORY_MIB,
    max_containers=1,
    scaledown_window=DEFAULT_SCALEDOWN_SECONDS,
    timeout=3600,
    volumes={MODEL_DIR: model_volume, USER_VOLUME_DIR: user_volume},
)
@modal.concurrent(max_inputs=100)
@modal.web_server(8000, startup_timeout=600)
def ui() -> None:
    """Start ComfyUI web server bound to 0.0.0.0:8000."""
    model_volume.reload()
    user_volume.reload()
    if common.ensure_models(MODEL_DIR, profile=DEFAULT_PROFILE, commit_cb=model_volume.commit):
        print("[OK] Committed Qwen-Image-2.1 models to Modal Volume.", flush=True)
    if common.ensure_user_workflows(
        USER_WORKFLOW_DIR,
        commit_cb=user_volume.commit,
        profile=DEFAULT_PROFILE,
    ):
        print("[OK] Committed user workflows to Modal Volume.", flush=True)

    # Preserve Modal's PYTHONPATH entries while chaining our WebSocket and userdata fixes via sitecustomize.py.
    env = _with_uncompressed_ws_env(os.environ)
    launch_cmd = [
        "comfy",
        "--skip-prompt",
        f"--workspace={COMFY_WORKSPACE}",
        "launch",
        "--",
        "--listen",
        "0.0.0.0",
        "--port",
        "8000",
        "--user-directory",
        USER_VOLUME_DIR,
        "--preview-method",
        "taesd",
        "--disable-auto-launch",
    ]
    if ENABLE_CACHE_NONE:
        launch_cmd.append("--cache-none")

    print(
        "[STARTUP] Launching ComfyUI on 0.0.0.0:8000 "
        f"(gpu={DEFAULT_GPUS}, ram={DEFAULT_MEMORY_MIB}MiB, scaledown={DEFAULT_SCALEDOWN_SECONDS}s, "
        f"profile={DEFAULT_PROFILE}, cache_none={ENABLE_CACHE_NONE}) ...",
        flush=True,
    )
    subprocess.Popen(launch_cmd, env=env)


@app.local_entrypoint()
def main(profile: str = DEFAULT_PROFILE) -> None:
    """Populate the persistent Modal Volume with Qwen-Image-2.1 models."""
    download_models.remote(profile=profile)
