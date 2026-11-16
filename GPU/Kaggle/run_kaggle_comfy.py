"""Kaggle Notebook runner for ComfyUI + Qwen-Image-2.1 using shared Wagi definitions."""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
TRYCLOUDFLARE_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def _resolve_shared_dir(explicit: Path | None = None) -> Path:
    """Locate the shared Wagi/qwen_image_2_1 directory in Kaggle or local workspace."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    if os.environ.get("SHARED_DIR"):
        candidates.append(Path(os.environ["SHARED_DIR"]))
    this_dir = Path(__file__).resolve().parent
    candidates.extend(
        [
            this_dir,
            this_dir.parent / "Wagi" / "qwen_image_2_1",
            Path("/kaggle/input/qwen21-shared/qwen_image_2_1"),
            Path("/kaggle/input/qwen21-shared"),
            Path("/kaggle/working/qwen21_shared"),
        ]
    )
    for candidate in candidates:
        if (candidate / "models.json").is_file() and (candidate / "common.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Cannot locate shared Wagi/qwen_image_2_1 (models.json + common.py). "
        "Pass --shared-dir or attach it under /kaggle/input/qwen21-shared."
    )


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("[CMD]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _ensure_comfyui(comfy_dir: Path) -> None:
    if not (comfy_dir / ".git").exists():
        _run(["git", "clone", "--depth", "1", "https://github.com/comfyanonymous/ComfyUI.git", str(comfy_dir)])
    _run([sys.executable, "-m", "pip", "install", "-q", "-r", str(comfy_dir / "requirements.txt")])
    _run([sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface-hub[hf_xet]"])


def _ensure_cloudflared(bin_path: Path) -> Path:
    existing = shutil.which("cloudflared")
    if existing:
        return Path(existing)
    if bin_path.exists() and bin_path.stat().st_size > 0:
        return bin_path
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    print("[DOWNLOAD] Fetching cloudflared...", flush=True)
    urllib.request.urlretrieve(CLOUDFLARED_URL, bin_path)
    bin_path.chmod(0o755)
    return bin_path


def _wait_for_comfy(port: int, timeout_s: int = 180) -> None:
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/system_stats"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if 200 <= resp.status < 400:
                    print(f"[OK] ComfyUI listening on 127.0.0.1:{port}", flush=True)
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"ComfyUI did not become ready on port {port} within {timeout_s}s.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ComfyUI + Qwen-Image-2.1 on Kaggle (T4).")
    parser.add_argument(
        "--shared-dir",
        type=Path,
        default=None,
        help="Path to Wagi/qwen_image_2_1 directory containing models.json & common.py.",
    )
    parser.add_argument(
        "--profile",
        default=os.environ.get("QWEN21_MODEL_PROFILE", "lite-int8"),
        help="Model profile (default: lite-int8 for Kaggle /tmp & 16GB T4 VRAM).",
    )
    parser.add_argument(
        "--comfy-dir",
        type=Path,
        default=Path("/tmp/ComfyUI"),
        help="Install ComfyUI & models outside /kaggle/working 20GB cap (default: /tmp/ComfyUI).",
    )
    parser.add_argument(
        "--user-dir",
        type=Path,
        default=Path("/kaggle/working/comfy_user"),
        help="Persistent user workflows/settings directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/outputs"),
        help="Persistent directory for generated images.",
    )
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument(
        "--no-cache-none",
        action="store_true",
        help="Disable --cache-none (kept enabled by default on 16GB T4 to prevent OOM).",
    )
    parser.add_argument("--no-tunnel", action="store_true", help="Skip launching cloudflared tunnel.")
    args = parser.parse_args()

    shared_dir = _resolve_shared_dir(args.shared_dir)
    if str(shared_dir) not in sys.path:
        sys.path.insert(0, str(shared_dir))
    import common  # noqa: E402

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_HUB_CACHE", "/tmp/hf_cache")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("QWEN21_ALLOW_OFFLINE_TEMPLATES", "1")

    args.user_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _ensure_comfyui(args.comfy_dir)
    common.ensure_models(args.comfy_dir / "models", profile=args.profile)
    common.ensure_user_workflows(args.user_dir / "default" / "workflows", profile=args.profile)

    comfy_cmd = [
        sys.executable,
        "main.py",
        "--listen",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--user-directory",
        str(args.user_dir),
        "--output-directory",
        str(args.output_dir),
        "--preview-method",
        "taesd",
        "--disable-auto-launch",
    ]
    if not args.no_cache_none:
        comfy_cmd.append("--cache-none")

    print("[STARTUP] Launching ComfyUI:", " ".join(comfy_cmd), flush=True)
    comfy_proc = subprocess.Popen(comfy_cmd, cwd=str(args.comfy_dir))
    tunnel_proc = None
    try:
        _wait_for_comfy(args.port)
        if args.no_tunnel:
            comfy_proc.wait()
            return

        cf_bin = _ensure_cloudflared(Path("/tmp/bin/cloudflared"))
        tunnel_proc = subprocess.Popen(
            [str(cf_bin), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{args.port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert tunnel_proc.stdout is not None
        printed_url = False
        for line in tunnel_proc.stdout:
            match = TRYCLOUDFLARE_RE.search(line)
            if match and not printed_url:
                print(f"\n[READY] ComfyUI public URL (paste into Bartosz OS): {match.group(0)}/\n", flush=True)
                printed_url = True
            elif "ERR" in line or "error" in line.lower():
                print(line.rstrip(), flush=True)
            if comfy_proc.poll() is not None:
                break
    finally:
        for proc in (tunnel_proc, comfy_proc):
            if proc is not None and proc.poll() is None:
                proc.terminate()


if __name__ == "__main__":
    main()
