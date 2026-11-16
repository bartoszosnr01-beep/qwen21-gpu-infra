"""Beam.cloud deployment running ComfyUI with Qwen-Image-2.1 using shared Wagi definitions."""

import argparse
import os
import sys
from collections.abc import Mapping
from pathlib import Path

REMOTE_SHARED_DIR = "/root/qwen21_shared"
CURRENT_FILE = Path(__file__).resolve()
LOCAL_SHARED_DIR = CURRENT_FILE.parent.parent / "Wagi" / "qwen_image_2_1"
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

DEFAULT_GPUS = [
    gpu.strip()
    for gpu in os.environ.get("QWEN21_BEAM_GPUS", "RTX4090,L4,A10G").split(",")
    if gpu.strip()
]
DEFAULT_PROFILE = os.environ.get("QWEN21_MODEL_PROFILE", "full-int8")
DEFAULT_MEMORY = os.environ.get("QWEN21_BEAM_MEMORY", "24Gi")
DEFAULT_KEEP_WARM_SECONDS = int(os.environ.get("QWEN21_BEAM_KEEP_WARM", "180"))
ENABLE_CACHE_NONE = os.environ.get("QWEN21_CACHE_NONE", "0") == "1"

WS_FIX_DIR = "/tmp/comfy_ws_fix"
WS_FIX_SCRIPT = """import importlib
import os
import sys
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
        # Disable permessage-deflate so reverse proxies without WS compression never break /ws.
        kwargs["compress"] = False
        return _orig_ws_init(self, *args, **kwargs)

    aiohttp.web.WebSocketResponse.__init__ = _ws_init_uncompressed

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

    async def _resolve_proxy_userdata(self, request):
        try:
            new_raw = _normalize_userdata_raw_path(request.raw_path, request.method)
            if new_raw is not None:
                cloned = request.clone(rel_url=yarl.URL(new_raw, encoded=True))
                return await _orig_resolve(self, cloned)
        except Exception:
            pass
        return await _orig_resolve(self, request)

    aiohttp.web.UrlDispatcher.resolve = _resolve_proxy_userdata
except Exception as exc:
    print(f"[WARNING] Failed to install ComfyUI runtime fixes: {exc}", flush=True)
"""


def _with_proxy_ws_env(base_env: Mapping[str, str], fix_dir: str = WS_FIX_DIR) -> dict[str, str]:
    """Install a Python sitecustomize hook for WebSocket compression and /userdata routing."""
    fix_path = Path(fix_dir)
    fix_path.mkdir(parents=True, exist_ok=True)
    (fix_path / "sitecustomize.py").write_text(WS_FIX_SCRIPT, encoding="utf-8")

    env = dict(base_env)
    existing = env.get("PYTHONPATH", "")
    parts = [part for part in existing.split(os.pathsep) if part] if existing else []
    if fix_dir not in parts:
        env["PYTHONPATH"] = os.pathsep.join([fix_dir, *parts])
    return env


def _serve_comfyui() -> None:
    """Ensure weights/workflows exist on Beam Volumes and exec ComfyUI on 0.0.0.0:8000."""
    common.ensure_models(MODEL_DIR, profile=DEFAULT_PROFILE)
    common.ensure_user_workflows(USER_WORKFLOW_DIR, profile=DEFAULT_PROFILE)

    env = _with_proxy_ws_env(os.environ)
    cmd = [
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
        cmd.append("--cache-none")

    print(
        "[STARTUP] Launching ComfyUI on Beam 0.0.0.0:8000 "
        f"(gpu={DEFAULT_GPUS}, ram={DEFAULT_MEMORY}, keep_warm={DEFAULT_KEEP_WARM_SECONDS}s, "
        f"profile={DEFAULT_PROFILE}, cache_none={ENABLE_CACHE_NONE}) ...",
        flush=True,
    )
    os.execvpe(cmd[0], cmd, env)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Beam.cloud ComfyUI Qwen-Image-2.1 launcher.")
    parser.add_argument("--serve", action="store_true", help="Run ComfyUI inside Beam container.")
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Trigger remote CPU download_models() onto the Beam Volume.",
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help="Model profile to populate when running --download-models.",
    )
    return parser


# Handle container --serve and CLI --help before importing Beam decorators (which require BEAM_TOKEN/login).
if __name__ == "__main__":
    _early_args = _build_parser().parse_args()
    if _early_args.serve:
        _serve_comfyui()

from beam import Image, Service, Volume, function  # noqa: E402

MANIFEST = common.load_manifest(SHARED_DIR / "models.json")
WORKFLOW_SEED_COMMANDS = [
    f"wget -q -O {WORKFLOW_DIR}/{filename} {url}"
    for filename, url in MANIFEST["workflow_templates"].items()
]

model_volume = Volume(name=VOLUME_NAME, mount_path=MODEL_DIR)
user_volume = Volume(name=USER_VOLUME_NAME, mount_path=USER_VOLUME_DIR)


def _attach_shared_files(img: Image) -> Image:
    """Attach shared Wagi files and this script to /root/qwen21_shared when running from repo root."""
    if not LOCAL_SHARED_DIR.is_dir():
        return img

    cwd = os.getcwd()
    files_to_sync = [
        (LOCAL_SHARED_DIR / "models.json", f"{REMOTE_SHARED_DIR}/models.json"),
        (LOCAL_SHARED_DIR / "common.py", f"{REMOTE_SHARED_DIR}/common.py"),
        (
            LOCAL_SHARED_DIR / "workflows" / MANIFEST["live_preview_workflow_filename"],
            f"{REMOTE_SHARED_DIR}/workflows/{MANIFEST['live_preview_workflow_filename']}",
        ),
        (CURRENT_FILE, f"{REMOTE_SHARED_DIR}/comfy_qwen21_beam.py"),
    ]
    for local_path, remote_path in files_to_sync:
        rel = Path(os.path.relpath(local_path, cwd)).as_posix()
        if rel == ".." or rel.startswith("../") or os.path.isabs(rel):
            raise RuntimeError(
                f"Run Beam commands from the repository root (G:\\Projekty\\Infrastruktura) so '{local_path}' is inside cwd."
            )
        img = img.add_local_file(str(local_path), remote_path=remote_path)
    return img


image = (
    Image(python_version="python3.12")
    .apt_install("git", "wget", "libgl1", "libglib2.0-0")
    .pip_install("comfy-cli==1.26.0", "huggingface-hub[hf_xet]", "uv", "aiohttp", "yarl")
    .run_commands(
        f"comfy --skip-prompt --workspace={COMFY_WORKSPACE} install --nvidia --fast-deps",
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
)
image = _attach_shared_files(image)


@function(
    name=f"{APP_NAME}-download-models",
    cpu=4.0,
    memory="8Gi",
    timeout=1800,
    image=image,
    volumes=[model_volume],
)
def download_models(profile: str = DEFAULT_PROFILE) -> dict[str, str]:
    """Download Qwen-Image-2.1 weights onto the persistent Beam Volume using CPU."""
    downloaded = common.ensure_models(MODEL_DIR, profile=profile)
    return {"status": "downloaded" if downloaded else "cached", "profile": profile}


ui_service = Service(
    name=APP_NAME,
    command=["python3", f"{REMOTE_SHARED_DIR}/comfy_qwen21_beam.py", "--serve"],
    port=8000,
    cpu=2.0,
    memory=DEFAULT_MEMORY,
    gpu=DEFAULT_GPUS,
    gpu_count=1,
    image=image,
    volumes=[model_volume, user_volume],
    keep_warm_seconds=DEFAULT_KEEP_WARM_SECONDS,
    min_replicas=0,
    max_replicas=1,
    always_on=False,
    authorized=False,
)


def main() -> None:
    """Trigger remote CPU download_models() onto the Beam Volume."""
    args = _build_parser().parse_args()
    if args.download_models:
        result = download_models.remote(profile=args.profile)
        print("[OK] Remote download_models result:", result, flush=True)
    else:
        _build_parser().print_help()


if __name__ == "__main__":
    main()
