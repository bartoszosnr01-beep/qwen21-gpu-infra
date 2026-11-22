"""Environment bootstrapper, diagnostics, and fast model staging for MiniMax-H3-Image on Kaggle."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from minimax_kaggle.constants import (
    COMFYUI_REPO_URL,
    DEFAULT_COMFY_DIR,
    DEFAULT_HF_CACHE_DIR,
    DEFAULT_INPUT_DIR,
    DEFAULT_VENV_DIR,
    DEFAULT_WORKING_DIR,
    GGUF_NODE_DIRNAME,
    GGUF_NODE_REPO_URL,
    GGUF_PRESETS,
    GPU_EXTRA_PIP_PACKAGES,
    MODELS,
    PRESETS,
    RUNTIME_COMFY_ARCHIVE,
    RUNTIME_GGUF_ARCHIVE,
    RUNTIME_META_FILENAME,
    RUNTIME_WHEELS_ARCHIVE,
    STANDARD_MODEL_SUBDIRS,
    WORKFLOW_TEMPLATE_URLS,
)

COPY_BUFFER_BYTES = 16 * 1024 * 1024  # 16 MiB sequential read buffer for GCSFuse -> NVMe


def venv_python_path(venv_dir: Path = DEFAULT_VENV_DIR) -> Path:
    """Return the Python executable inside the isolated virtual environment."""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _read_linux_meminfo_gb() -> dict[str, float]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return {}
    values: dict[str, int] = {}
    for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split(":", 1)
        if len(parts) != 2:
            continue
        key = parts[0].strip()
        num_tokens = parts[1].strip().split()
        if num_tokens and num_tokens[0].isdigit():
            values[key] = int(num_tokens[0])  # KiB
    out: dict[str, float] = {}
    if "MemTotal" in values:
        out["ram_total_gb"] = round(values["MemTotal"] / (1024 * 1024), 2)
    if "MemAvailable" in values:
        out["ram_available_gb"] = round(values["MemAvailable"] / (1024 * 1024), 2)
    return out


def _disk_stats_gb(path: Path) -> dict[str, float] | None:
    target = path if path.exists() else path.parent
    if not target.exists():
        return None
    usage = shutil.disk_usage(target)
    return {
        "total_gb": round(usage.total / (1024**3), 2),
        "used_gb": round(usage.used / (1024**3), 2),
        "free_gb": round(usage.free / (1024**3), 2),
    }


def collect_system_diagnostics(
    tmp_dir: Path = Path("/tmp"),
    working_dir: Path = DEFAULT_WORKING_DIR,
    input_dir: Path = DEFAULT_INPUT_DIR,
) -> dict[str, Any]:
    """Collect hardware, OS, disk, RAM, GPU, and TPU facts before running workloads."""
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "memory": _read_linux_meminfo_gb(),
        "disks": {
            "tmp": _disk_stats_gb(tmp_dir),
            "working": _disk_stats_gb(working_dir),
            "input": _disk_stats_gb(input_dir),
        },
        "nvidia_smi": None,
    }
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        proc = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,compute_cap,memory.total,memory.free",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            report["nvidia_smi"] = [line.strip() for line in proc.stdout.splitlines() if line.strip()]

    vfio_dir = Path("/dev/vfio")
    accel_devs = sorted(str(p) for p in Path("/dev").glob("accel*")) if Path("/dev").exists() else []
    vfio_devs = sorted(str(p) for p in vfio_dir.iterdir()) if vfio_dir.exists() else []
    tpu_env = {
        k: v
        for k, v in os.environ.items()
        if k in ("PJRT_DEVICE", "TPU_NAME", "TPU_ACCELERATOR_TYPE", "TPU_LIBRARY_PATH", "ACCELERATOR_TYPE")
        or k.startswith(("TPU_", "CLOUD_TPU_"))
    }
    if accel_devs or vfio_devs or tpu_env:
        report["tpu_devices"] = accel_devs or vfio_devs or ["PJRT_TPU"]
        report["tpu_env"] = tpu_env
    return report


def _run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    print(f"[CMD] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=str(cwd) if cwd else None)


def ensure_uv() -> str:
    """Ensure `uv` is available and return its executable path."""
    uv_bin = shutil.which("uv")
    if uv_bin:
        return uv_bin
    _run([sys.executable, "-m", "pip", "install", "-q", "uv"])
    uv_bin = shutil.which("uv")
    if not uv_bin:
        raise RuntimeError("Failed to install or locate `uv` binary.")
    return uv_bin


TPU_VENV_PYTHON_VERSION = "3.12"

TPU_PYTORCH_XLA_DEPS: tuple[str, ...] = (
    # libtpu==0.0.21 pinned by torch-xla[tpu]==2.9.0 publishes cp310-cp312 wheels
    "torch>=2.9.0,<2.10",
    "torch_xla[tpu]==2.9.0",
    "torchvision>=0.24,<0.25",
    "torchaudio>=2.9,<2.10",
)

TPU_COMFY_SAFE_DEPS: tuple[str, ...] = (
    "torchsde",
    "einops",
    "transformers>=4.50.3",
    "tokenizers>=0.13.3",
    "sentencepiece",
    "protobuf",
    "safetensors>=0.4.2",
    "aiohttp>=3.11.8",
    "yarl>=1.18.0",
    "pyyaml",
    "Pillow",
    "scipy",
    "tqdm",
    "psutil",
    "alembic",
    "SQLAlchemy",
    "av>=14.2.0",
    "comfyui-frontend-package",
    "comfyui-workflow-templates",
    "comfyui-embedded-docs",
    "pydantic~=2.0",
    "pydantic-settings~=2.0",
    "kornia>=0.7.1",
    "spandrel",
    "soundfile",
    "comfy-aimdo",
    "comfy-angle",
    "comfy-kitchen",
    "huggingface-hub[hf_xet]>=0.32.0",
    "packaging",
    "requests",
    "wrapt",
)


def _tpu_venv_ready(py_exe: Path) -> bool:
    """Return True if `py_exe` is Python 3.12 with `torch_xla` already installed."""
    if not py_exe.exists():
        return False
    probe = subprocess.run(
        [
            str(py_exe),
            "-c",
            "import sys, torch, torch_xla.runtime as xr; assert sys.version_info[:2] == (3, 12)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0


def probe_outer_tpu_jax(timeout_s: int = 45) -> str:
    """Inspect whether the outer system Python already detects TPU vs CPU fallback via JAX."""
    code = (
        "try:\n"
        "    import jax\n"
        "    d = jax.devices()\n"
        "    print('OUTER_JAX_RESULT:', len(d), d[0].platform, getattr(d[0], 'device_kind', ''))\n"
        "except ModuleNotFoundError:\n"
        "    print('OUTER_JAX_MISSING')\n"
        "except Exception as exc:\n"
        "    print('OUTER_JAX_ERROR:', str(exc)[:200])\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        out_lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
        for ln in reversed(out_lines):
            if ln.startswith(("OUTER_JAX_RESULT:", "OUTER_JAX_MISSING", "OUTER_JAX_ERROR:")):
                return ln
        return ((proc.stderr or proc.stdout).strip())[-300:]
    except Exception as exc:
        return f"OUTER_JAX_ERROR: {exc}"


def ensure_physical_tpu_or_raise(diag: dict[str, Any] | None = None) -> str:
    """Fail fast in <5s when Kaggle started a TPU-labeled session on CPU-only hardware."""
    jax_line = probe_outer_tpu_jax()
    print("[TPU] Outer jax probe:", jax_line, flush=True)
    if jax_line.startswith("OUTER_JAX_RESULT:") and " tpu" not in f" {jax_line.lower()}":
        info = diag or collect_system_diagnostics()
        raise RuntimeError(
            f"Kaggle started this session WITHOUT physical TPU hardware ({jax_line}; "
            f"ram_total_gb={info.get('memory', {}).get('ram_total_gb')}, "
            f"cpu_count={info.get('cpu_count')}). "
            "Verify TPU queue/access in Kaggle Settings and ensure "
            "Accelerator is set to 'TPU VM v5e-8' before running."
        )
    return jax_line


def _extract_tar_safely(tar_path: Path, dest_dir: Path) -> None:
    """Safely extract a tar archive rejecting path-traversal or symlink escapes."""
    dest_resolved = dest_dir.resolve()
    dest_resolved.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, mode="r:*") as tf:
        members = tf.getmembers()
        for member in members:
            target = (dest_resolved / member.name).resolve()
            if target != dest_resolved and dest_resolved not in target.parents:
                raise RuntimeError(f"Unsafe tar member path {member.name!r} in {tar_path}")
            if member.issym() or member.islnk():
                link_target = (target.parent / member.linkname).resolve()
                if link_target != dest_resolved and dest_resolved not in link_target.parents:
                    raise RuntimeError(f"Unsafe tar link {member.name!r} -> {member.linkname!r}")
        extract_kwargs: dict[str, Any] = {}
        if sys.version_info >= (3, 12):
            extract_kwargs["filter"] = "data"
        tf.extractall(dest_resolved, members=members, **extract_kwargs)


def _gpu_venv_ready(py_exe: Path) -> bool:
    """Return True when the isolated GPU venv already has CUDA PyTorch, GGUF, and ComfyUI deps."""
    if not py_exe.exists():
        return False
    probe = subprocess.run(
        [
            str(py_exe),
            "-c",
            (
                "import torch, torchvision, aiohttp, PIL, safetensors, comfy_kitchen, gguf, huggingface_hub; "
                "assert torch.cuda.is_available()"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0


def _ensure_comfy_repo(comfy_dir: Path, input_root: Path) -> None:
    if (comfy_dir / "main.py").exists():
        print(f"[OK] ComfyUI already present at {comfy_dir}", flush=True)
        return
    cached_tar = find_input_asset(RUNTIME_COMFY_ARCHIVE, input_root)
    if cached_tar is not None:
        t0 = time.monotonic()
        stage_root = comfy_dir.parent / f"._extract_comfy_{os.getpid()}"
        tmp_tar = comfy_dir.parent / f"._stage_{RUNTIME_COMFY_ARCHIVE}"
        try:
            shutil.rmtree(stage_root, ignore_errors=True)
            sequential_copy_file(cached_tar, tmp_tar)
            _extract_tar_safely(tmp_tar, stage_root)
            extracted_repo = stage_root / "ComfyUI" if (stage_root / "ComfyUI" / "main.py").exists() else stage_root
            if (extracted_repo / "main.py").exists():
                if comfy_dir.exists():
                    shutil.rmtree(comfy_dir, ignore_errors=True)
                comfy_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(extracted_repo), str(comfy_dir))
                dt = time.monotonic() - t0
                print(f"[CACHE] Restored ComfyUI from {cached_tar.name} in {dt:.1f}s", flush=True)
                return
        except Exception as exc:
            print(f"[CACHE] Failed extracting cached ComfyUI ({exc}); falling back to git clone.", flush=True)
        finally:
            tmp_tar.unlink(missing_ok=True)
            shutil.rmtree(stage_root, ignore_errors=True)

    comfy_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", COMFYUI_REPO_URL, str(comfy_dir)])


def ensure_gguf_custom_node(comfy_dir: Path, input_root: Path = DEFAULT_INPUT_DIR) -> Path:
    """Install or restore `Nif00/ComfyUI-GGUF` (with truncated Qwen3-VL-32B + mmproj support)."""
    node_dir = comfy_dir / "custom_nodes" / GGUF_NODE_DIRNAME
    if (node_dir / "nodes.py").exists():
        print(f"[OK] {GGUF_NODE_DIRNAME} already present at {node_dir}", flush=True)
        return node_dir

    cached_tar = find_input_asset(RUNTIME_GGUF_ARCHIVE, input_root)
    if cached_tar is not None:
        t0 = time.monotonic()
        stage_root = comfy_dir.parent / f"._extract_gguf_{os.getpid()}"
        tmp_tar = comfy_dir.parent / f"._stage_{RUNTIME_GGUF_ARCHIVE}"
        try:
            shutil.rmtree(stage_root, ignore_errors=True)
            sequential_copy_file(cached_tar, tmp_tar)
            _extract_tar_safely(tmp_tar, stage_root)
            candidate = (
                stage_root / GGUF_NODE_DIRNAME
                if (stage_root / GGUF_NODE_DIRNAME / "nodes.py").exists()
                else stage_root
            )
            if (candidate / "nodes.py").exists():
                node_dir.parent.mkdir(parents=True, exist_ok=True)
                if node_dir.exists():
                    shutil.rmtree(node_dir, ignore_errors=True)
                shutil.move(str(candidate), str(node_dir))
                dt = time.monotonic() - t0
                print(f"[CACHE] Restored {GGUF_NODE_DIRNAME} from {cached_tar.name} in {dt:.1f}s", flush=True)
                return node_dir
        except Exception as exc:
            print(f"[CACHE] Failed extracting cached {GGUF_NODE_DIRNAME} ({exc}); falling back to git clone.", flush=True)
        finally:
            tmp_tar.unlink(missing_ok=True)
            shutil.rmtree(stage_root, ignore_errors=True)

    node_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", GGUF_NODE_REPO_URL, str(node_dir)])
    return node_dir


def setup_comfy_and_venv(
    comfy_dir: Path = DEFAULT_COMFY_DIR,
    venv_dir: Path = DEFAULT_VENV_DIR,
    *,
    install_xformers: bool = False,
    install_gguf_node: bool | None = None,
    accelerator: str = "gpu",
    input_root: Path = DEFAULT_INPUT_DIR,
) -> Path:
    """Restore or clone ComfyUI into /tmp, install optional GGUF nodes, and build an isolated uv venv."""
    if accelerator == "tpu":
        ensure_physical_tpu_or_raise()

    uv_bin = ensure_uv()
    _ensure_comfy_repo(comfy_dir, input_root)

    want_gguf = (accelerator != "tpu") if install_gguf_node is None else bool(install_gguf_node)
    if want_gguf:
        ensure_gguf_custom_node(comfy_dir, input_root)

    py_exe = venv_python_path(venv_dir)
    if accelerator == "tpu":
        if not _tpu_venv_ready(py_exe):
            if venv_dir.exists():
                shutil.rmtree(venv_dir, ignore_errors=True)
            _run([uv_bin, "venv", "--python", TPU_VENV_PYTHON_VERSION, str(venv_dir)])
            _run(
                [
                    uv_bin,
                    "pip",
                    "install",
                    "--python",
                    str(py_exe),
                    "--torch-backend=cpu",
                    *TPU_PYTORCH_XLA_DEPS,
                    *TPU_COMFY_SAFE_DEPS,
                ]
            )
        tpu_env = dict(os.environ)
        tpu_env.pop("TPU_LIBRARY_PATH", None)
        tpu_env.setdefault("PJRT_DEVICE", "TPU")
        cmd_check = [
            str(py_exe),
            "-c",
            (
                "import pathlib, shutil, subprocess, sys, torch, torch_xla.runtime as xr; "
                "xr.use_spmd(); "
                "dtype = xr.device_type(); "
                "chips = xr.global_runtime_device_count(); "
                "devs = sorted(str(p) for p in pathlib.Path('/dev').glob('accel*')); "
                "vfio = sorted(str(p) for p in pathlib.Path('/dev/vfio').iterdir()) if pathlib.Path('/dev/vfio').exists() else []; "
                "print(f'[TORCH_XLA] py={sys.version.split()[0]} torch={torch.__version__} device={dtype} chips={chips} accel={devs} vfio={vfio}', flush=True); "
                "tpu_info_bin = shutil.which('tpu-info'); "
                "\nif tpu_info_bin:\n"
                "    subprocess.run([tpu_info_bin], check=False)\n"
                "assert str(dtype).upper() == 'TPU' and chips >= 1, f'No physical TPU detected (device_type={dtype!r}, chips={chips})'"
            ),
        ]
        print(f"[CMD] {' '.join(cmd_check)}", flush=True)
        subprocess.run(cmd_check, check=True, env=tpu_env, timeout=45)
        return py_exe

    req_file = comfy_dir / "requirements.txt"
    extra_pkgs = list(GPU_EXTRA_PIP_PACKAGES)
    if install_xformers:
        extra_pkgs.append("xformers")

    if not py_exe.exists():
        _run([uv_bin, "venv", "--python", sys.executable, str(venv_dir)])

    if not install_xformers and _gpu_venv_ready(py_exe):
        print(f"[OK] Isolated GPU venv already ready at {venv_dir}", flush=True)
    else:
        wheels_tar = find_input_asset(RUNTIME_WHEELS_ARCHIVE, input_root)
        installed_from_cache = False
        if wheels_tar is not None:
            wheels_dir = venv_dir.parent / f"._wheels_{os.getpid()}"
            tmp_wheels_tar = venv_dir.parent / f"._stage_{RUNTIME_WHEELS_ARCHIVE}"
            try:
                shutil.rmtree(wheels_dir, ignore_errors=True)
                t0 = time.monotonic()
                sequential_copy_file(wheels_tar, tmp_wheels_tar)
                _extract_tar_safely(tmp_wheels_tar, wheels_dir)
                meta_hit = find_input_asset(RUNTIME_META_FILENAME, input_root)
                meta_mm = ""
                if meta_hit is not None:
                    try:
                        meta_mm = str(json.loads(meta_hit.read_text(encoding="utf-8")).get("python_mm") or "")
                    except Exception:
                        meta_mm = ""
                host_mm = f"{sys.version_info[0]}.{sys.version_info[1]}"
                base_install = [uv_bin, "pip", "install", "--python", str(py_exe), "--find-links", str(wheels_dir)]
                pkg_args = ["-r", str(req_file), *extra_pkgs]
                if not meta_mm or meta_mm == host_mm:
                    try:
                        _run([*base_install, "--no-index", *pkg_args])
                        installed_from_cache = True
                    except subprocess.CalledProcessError:
                        print(
                            "[CACHE] Offline wheel install missed a dependency; retrying with PyPI fallback + local cache.",
                            flush=True,
                        )
                if not installed_from_cache:
                    _run([*base_install, *pkg_args])
                    installed_from_cache = True
                print(f"[CACHE] Installed GPU venv from cached wheels in {time.monotonic() - t0:.1f}s", flush=True)
            except Exception as exc:
                print(f"[CACHE] Wheel cache error ({exc}); falling back to standard PyPI install.", flush=True)
            finally:
                tmp_wheels_tar.unlink(missing_ok=True)
                shutil.rmtree(wheels_dir, ignore_errors=True)

        if not installed_from_cache:
            _run([uv_bin, "pip", "install", "--python", str(py_exe), "-r", str(req_file), *extra_pkgs])

    _run(
        [
            str(py_exe),
            "-c",
            (
                "import importlib.metadata, torch, gguf; "
                "gguf_ver = getattr(gguf, '__version__', None) or importlib.metadata.version('gguf'); "
                "print('[TORCH]', torch.__version__, 'cuda:', torch.version.cuda, "
                "'count:', torch.cuda.device_count(), 'gguf:', gguf_ver); "
                "assert torch.cuda.is_available(), 'CUDA unavailable'; "
                "print('[GPUS]', [torch.cuda.get_device_name(i) + ' ' + str(torch.cuda.get_device_capability(i)) "
                "for i in range(torch.cuda.device_count())])"
            ),
        ]
    )
    return py_exe


def find_input_asset(filename: str, input_root: Path = DEFAULT_INPUT_DIR) -> Path | None:
    """Locate a pre-cached asset under /kaggle/input (flat or nested directory layout)."""
    if not input_root.exists():
        return None
    for candidate in input_root.rglob(filename):
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def sequential_copy_file(src: Path, dst: Path) -> float:
    """Copy a large file sequentially using a 16 MiB buffer to avoid FUSE mmap stalls."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst.with_name(dst.name + ".tmp")
    t0 = time.monotonic()
    try:
        with src.open("rb") as f_in, tmp_dst.open("wb") as f_out:
            while True:
                chunk = f_in.read(COPY_BUFFER_BYTES)
                if not chunk:
                    break
                f_out.write(chunk)
        tmp_dst.replace(dst)
    except Exception:
        tmp_dst.unlink(missing_ok=True)
        raise
    elapsed = max(time.monotonic() - t0, 1e-6)
    gb = dst.stat().st_size / (1024**3)
    print(f"[COPY] {src} -> {dst} ({gb:.2f} GiB in {elapsed:.1f}s, {gb*1024/elapsed:.0f} MiB/s)", flush=True)
    return elapsed


def _download_url_atomic(url: str, dst: Path, timeout: int = 120) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst.with_name(dst.name + ".tmp")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, tmp_dst.open("wb") as out:
            shutil.copyfileobj(resp, out, length=COPY_BUFFER_BYTES)
        if tmp_dst.stat().st_size <= 0:
            raise RuntimeError(f"Downloaded empty file from {url}")
        tmp_dst.replace(dst)
    except Exception:
        tmp_dst.unlink(missing_ok=True)
        raise


def _hf_download_in_venv(
    py_exe: Path,
    repo_id: str,
    rel_path: str,
    target_file: Path,
) -> None:
    """Download `repo_id/rel_path` via `hf_xet` into `target_file` regardless of nested repo layout."""
    env = dict(os.environ)
    env["HF_XET_HIGH_PERFORMANCE"] = "1"
    env["HF_HUB_CACHE"] = str(DEFAULT_HF_CACHE_DIR)
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    dl_root = target_file.parent
    dl_root.mkdir(parents=True, exist_ok=True)
    script = (
        "from pathlib import Path\n"
        "from huggingface_hub import hf_hub_download\n"
        f"p = Path(hf_hub_download(repo_id={repo_id!r}, filename={rel_path!r}, local_dir={str(dl_root)!r}))\n"
        f"dst = Path({str(target_file)!r})\n"
        "if p.resolve() != dst.resolve():\n"
        "    dst.parent.mkdir(parents=True, exist_ok=True)\n"
        "    p.replace(dst)\n"
        "print(dst)\n"
    )
    _run([str(py_exe), "-c", script], env=env)
    shutil.rmtree(dl_root / ".cache", ignore_errors=True)
    # Clean empty nested subdirs created by repo paths like `diffusion_models/...`
    nested_parts = Path(rel_path).parts
    if len(nested_parts) > 1:
        nested_top = dl_root / nested_parts[0]
        if nested_top.exists() and nested_top.is_dir() and nested_top != dl_root:
            shutil.rmtree(nested_top, ignore_errors=True)


def ensure_models_and_templates(
    preset: str = "tpu_int8",
    *,
    comfy_dir: Path = DEFAULT_COMFY_DIR,
    venv_dir: Path = DEFAULT_VENV_DIR,
    input_root: Path = DEFAULT_INPUT_DIR,
    force_hf: bool = False,
) -> dict[str, str]:
    """Ensure MiniMax-H3-Image weights and official templates exist in ComfyUI workspace."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset={preset!r}; expected one of {tuple(PRESETS)}")

    py_exe = venv_python_path(venv_dir)
    if not py_exe.exists():
        raise RuntimeError(f"Venv Python not found at {py_exe}; run setup_comfy_and_venv() first.")

    if preset in GGUF_PRESETS:
        ensure_gguf_custom_node(comfy_dir, input_root)

    model_root = comfy_dir / "models"
    for subdir in STANDARD_MODEL_SUBDIRS:
        (model_root / subdir).mkdir(parents=True, exist_ok=True)

    sources: dict[str, str] = {}
    for key in PRESETS[preset]:
        spec = MODELS[key]
        dst = model_root / spec.subdir / spec.filename
        if dst.exists() and dst.stat().st_size > 0:
            print(f"[OK] Model ready: {dst}", flush=True)
            sources[spec.filename] = "cached_local"
            continue

        input_hit = None if force_hf else find_input_asset(spec.filename, input_root)
        if input_hit is not None:
            sequential_copy_file(input_hit, dst)
            sources[spec.filename] = f"dataset_copy:{input_hit}"
        else:
            print(
                f"[HF] Downloading {spec.hf_repo_id}/{spec.hf_path} (~{spec.approx_gib:.2f} GiB)...",
                flush=True,
            )
            t0 = time.monotonic()
            _hf_download_in_venv(py_exe, spec.hf_repo_id, spec.hf_path, dst)
            if not dst.exists() or dst.stat().st_size <= 0:
                raise RuntimeError(f"Expected HF download at {dst}, but file is missing/empty.")
            dt = time.monotonic() - t0
            print(f"[HF] Done {spec.filename} in {dt:.1f}s", flush=True)
            sources[spec.filename] = f"hf_xet:{dt:.1f}s"

    # Official Comfy-Org MiniMax-H3-Image workflow templates
    wf_dir = comfy_dir / "user" / "default" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    for wf_name, wf_url in WORKFLOW_TEMPLATE_URLS.items():
        wf_dst = wf_dir / wf_name
        if wf_dst.exists() and wf_dst.stat().st_size > 0:
            continue
        wf_hit = None if force_hf else find_input_asset(wf_name, input_root)
        if wf_hit is not None:
            sequential_copy_file(wf_hit, wf_dst)
        else:
            print(f"[DL] Fetching template {wf_name}", flush=True)
            _download_url_atomic(wf_url, wf_dst)

    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap ComfyUI and MiniMax-H3-Image assets on Kaggle.")
    parser.add_argument("--preset", default="t4_gguf", choices=tuple(PRESETS))
    parser.add_argument("--accelerator", default="gpu", choices=("gpu", "tpu"))
    parser.add_argument("--comfy-dir", type=Path, default=DEFAULT_COMFY_DIR)
    parser.add_argument("--venv-dir", type=Path, default=DEFAULT_VENV_DIR)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--xformers", action="store_true")
    parser.add_argument("--force-hf", action="store_true")
    args = parser.parse_args()

    print("[DIAG]", collect_system_diagnostics(working_dir=DEFAULT_WORKING_DIR, input_dir=args.input_dir), flush=True)
    setup_comfy_and_venv(
        args.comfy_dir,
        args.venv_dir,
        install_xformers=args.xformers,
        accelerator=args.accelerator,
        input_root=args.input_dir,
    )
    sources = ensure_models_and_templates(
        preset=args.preset,
        comfy_dir=args.comfy_dir,
        venv_dir=args.venv_dir,
        input_root=args.input_dir,
        force_hf=args.force_hf,
    )
    print("[SOURCES]", sources, flush=True)


if __name__ == "__main__":
    main()
