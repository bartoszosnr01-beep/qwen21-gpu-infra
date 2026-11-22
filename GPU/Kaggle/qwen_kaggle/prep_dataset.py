"""CPU-only script to stage Qwen-Image 2.1 models into a private Kaggle Dataset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

from qwen_kaggle.bootstrap import ensure_uv, venv_python_path
from qwen_kaggle.constants import (
    CLOUDFLARED_LINUX_URL,
    COMFYUI_REPO_URL,
    DEFAULT_VENV_DIR,
    GPU_EXTRA_PIP_PACKAGES,
    HF_REPO_ID,
    MODELS,
    PRESETS,
    RUNTIME_CACHE_SLUG,
    RUNTIME_CLOUDFLARED_FILENAME,
    RUNTIME_COMFY_ARCHIVE,
    RUNTIME_META_FILENAME,
    RUNTIME_WHEELS_ARCHIVE,
    TAESD_DECODER_FILENAME,
    TAESD_DECODER_URL,
    WORKFLOW_TEMPLATE_URLS,
)


def _read_kaggle_secret(name: str) -> str:
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        val = UserSecretsClient().get_secret(name)
        return str(val or "").strip()
    except Exception:
        return ""


def _resolve_kaggle_auth(owner: str = "") -> tuple[str, dict[str, str]]:
    """Resolve Kaggle owner handle and auth env for `kaggle>=1.8.0` (`KAGGLE_API_TOKEN` or legacy key)."""
    api_token = os.environ.get("KAGGLE_API_TOKEN", "").strip() or _read_kaggle_secret("KAGGLE_API_TOKEN")
    username = owner.strip() or os.environ.get("KAGGLE_USERNAME", "").strip() or _read_kaggle_secret("KAGGLE_USERNAME")
    legacy_key = os.environ.get("KAGGLE_KEY", "").strip() or _read_kaggle_secret("KAGGLE_KEY")

    kaggle_dir = Path.home() / ".kaggle"
    if not api_token:
        access_token_file = kaggle_dir / "access_token"
        if access_token_file.exists():
            api_token = access_token_file.read_text(encoding="utf-8").strip()

    kaggle_json = kaggle_dir / "kaggle.json"
    if kaggle_json.exists() and (not username or (not api_token and not legacy_key)):
        data = json.loads(kaggle_json.read_text(encoding="utf-8"))
        username = username or str(data.get("username", "")).strip()
        legacy_key = legacy_key or str(data.get("key", "")).strip()

    if api_token:
        if not username:
            raise RuntimeError(
                "Found KAGGLE_API_TOKEN, but dataset metadata also needs your Kaggle username. "
                "Set KAGGLE_OWNER in the notebook cell or add KAGGLE_USERNAME in Kaggle Secrets."
            )
        return username, {"KAGGLE_API_TOKEN": api_token}

    if username and legacy_key:
        return username, {"KAGGLE_USERNAME": username, "KAGGLE_KEY": legacy_key}

    raise RuntimeError(
        "Kaggle API credentials not found. Add KAGGLE_API_TOKEN (+ KAGGLE_OWNER/KAGGLE_USERNAME) "
        "or legacy KAGGLE_USERNAME + KAGGLE_KEY in Kaggle Notebook Secrets."
    )


def _ensure_prep_venv(venv_dir: Path, *, include_pip: bool = False) -> Path:
    uv_bin = ensure_uv()
    py_exe = venv_python_path(venv_dir)
    if not py_exe.exists():
        subprocess.run([uv_bin, "venv", "--python", sys.executable, str(venv_dir)], check=True)
    pkgs = [
        "huggingface-hub[hf_xet]>=0.32.0",
        "kaggle>=1.8.0",
        "six",
        "wrapt",
    ]
    if include_pip:
        pkgs.append("pip")
    subprocess.run(
        [
            uv_bin,
            "pip",
            "install",
            "--python",
            str(py_exe),
            *pkgs,
        ],
        check=True,
    )
    return py_exe


def _stage_flat_assets(py_exe: Path, staging_dir: Path, preset: str) -> None:
    staging_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["HF_XET_HIGH_PERFORMANCE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"

    for key in PRESETS[preset]:
        spec = MODELS[key]
        flat_target = staging_dir / spec.filename
        if flat_target.exists() and flat_target.stat().st_size > 0:
            print(f"[OK] Already staged: {flat_target.name}", flush=True)
            continue

        print(f"[HF] Downloading {spec.hf_path} (~{spec.approx_gib:.2f} GiB)...", flush=True)
        dl_code = (
            "from huggingface_hub import hf_hub_download; "
            f"p = hf_hub_download(repo_id={HF_REPO_ID!r}, filename={spec.hf_path!r}, local_dir={str(staging_dir)!r}); "
            "print(p)"
        )
        subprocess.run([str(py_exe), "-c", dl_code], check=True, env=env)

        nested_file = staging_dir / spec.hf_path
        if nested_file.exists():
            nested_file.replace(flat_target)
        shutil.rmtree(staging_dir / ".cache", ignore_errors=True)

    # Clean nested folders so staging_dir contains only flat assets
    for subdir_name in ("diffusion_models", "text_encoders", "vae"):
        sub = staging_dir / subdir_name
        if sub.exists():
            shutil.rmtree(sub, ignore_errors=True)

    # TAESD decoder
    taesd_dst = staging_dir / TAESD_DECODER_FILENAME
    if not (taesd_dst.exists() and taesd_dst.stat().st_size > 0):
        with urllib.request.urlopen(TAESD_DECODER_URL, timeout=60) as resp, taesd_dst.open("wb") as out:
            shutil.copyfileobj(resp, out)

    # Official templates
    for wf_name, wf_url in WORKFLOW_TEMPLATE_URLS.items():
        wf_dst = staging_dir / wf_name
        if not (wf_dst.exists() and wf_dst.stat().st_size > 0):
            with urllib.request.urlopen(wf_url, timeout=60) as resp, wf_dst.open("wb") as out:
                shutil.copyfileobj(resp, out)


def _pack_tar_flat_root(src_dir: Path, tar_dst: Path, *, arc_prefix: str = "") -> None:
    tmp_tar = tar_dst.with_name(tar_dst.name + ".tmp")
    try:
        with tarfile.open(tmp_tar, mode="w") as tf:
            for item in sorted(src_dir.iterdir()):
                arcname = f"{arc_prefix}/{item.name}" if arc_prefix else item.name
                tf.add(item, arcname=arcname)
        tmp_tar.replace(tar_dst)
    except Exception:
        tmp_tar.unlink(missing_ok=True)
        raise


def stage_runtime_cache(
    staging_dir: Path,
    *,
    work_dir: Path = Path("/tmp/qwen21_runtime_work"),
    venv_dir: Path = DEFAULT_VENV_DIR,
    include_xformers: bool = False,
) -> dict[str, Any]:
    """Stage offline GPU wheels (`qwen_t4_wheels.tar`), `comfyui_repo.tar`, and `cloudflared` into flat `staging_dir`."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    py_exe = _ensure_prep_venv(venv_dir, include_pip=True)

    comfy_src = work_dir / "ComfyUI"
    wheels_dir = work_dir / "wheels"
    try:
        shutil.rmtree(comfy_src, ignore_errors=True)
        shutil.rmtree(wheels_dir, ignore_errors=True)
        wheels_dir.mkdir(parents=True, exist_ok=True)

        print(f"[CACHE-BUILD] Cloning shallow ComfyUI into {comfy_src}...", flush=True)
        subprocess.run(["git", "clone", "--depth", "1", COMFYUI_REPO_URL, str(comfy_src)], check=True)

        comfy_tar = staging_dir / RUNTIME_COMFY_ARCHIVE
        _pack_tar_flat_root(comfy_src, comfy_tar, arc_prefix="ComfyUI")
        print(f"[CACHE-BUILD] Packed {comfy_tar.name} ({comfy_tar.stat().st_size / (1024**2):.1f} MiB)", flush=True)

        req_file = comfy_src / "requirements.txt"
        extra = list(GPU_EXTRA_PIP_PACKAGES)
        if include_xformers:
            extra.append("xformers")
        print(f"[CACHE-BUILD] Downloading offline wheels into {wheels_dir}...", flush=True)
        subprocess.run(
            [
                str(py_exe),
                "-m",
                "pip",
                "download",
                "--no-cache-dir",
                "--dest",
                str(wheels_dir),
                "-r",
                str(req_file),
                *extra,
            ],
            check=True,
        )
        shutil.rmtree(comfy_src, ignore_errors=True)
        wheel_files = [p for p in wheels_dir.iterdir() if p.is_file()]
        wheels_tar = staging_dir / RUNTIME_WHEELS_ARCHIVE
        _pack_tar_flat_root(wheels_dir, wheels_tar)
        shutil.rmtree(wheels_dir, ignore_errors=True)
        print(
            f"[CACHE-BUILD] Packed {wheels_tar.name}: {len(wheel_files)} files "
            f"({wheels_tar.stat().st_size / (1024**3):.2f} GiB)",
            flush=True,
        )

        cf_dst = staging_dir / RUNTIME_CLOUDFLARED_FILENAME
        cf_tmp = cf_dst.with_name(cf_dst.name + ".tmp")
        with urllib.request.urlopen(CLOUDFLARED_LINUX_URL, timeout=90) as resp, cf_tmp.open("wb") as out:
            shutil.copyfileobj(resp, out, length=4 * 1024 * 1024)
        os.chmod(cf_tmp, 0o755)
        cf_tmp.replace(cf_dst)

        meta = {
            "python_mm": f"{sys.version_info[0]}.{sys.version_info[1]}",
            "python_full": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "wheel_count": len(wheel_files),
            "include_xformers": include_xformers,
            "created_at_iso": datetime.now(timezone.utc).isoformat(),
        }
        (staging_dir / RUNTIME_META_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _upload_private_kaggle_dataset(
    py_exe: Path,
    staging_dir: Path,
    *,
    username: str,
    auth_env: dict[str, str],
    dataset_slug: str,
    title: str,
    update_existing: bool,
    version_notes: str,
) -> str:
    dataset_id = f"{username}/{dataset_slug}"
    meta = {
        "title": title,
        "id": dataset_id,
        "licenses": [{"name": "other"}],
    }
    (staging_dir / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    kaggle_cli = py_exe.parent / ("kaggle.exe" if os.name == "nt" else "kaggle")
    env = dict(os.environ)
    for k in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        env.pop(k, None)
    env.update(auth_env)

    if update_existing:
        cmd = [
            str(kaggle_cli),
            "datasets",
            "version",
            "-p",
            str(staging_dir),
            "-m",
            version_notes,
            "--dir-mode",
            "skip",
        ]
    else:
        cmd = [
            str(kaggle_cli),
            "datasets",
            "create",
            "-p",
            str(staging_dir),
            "--dir-mode",
            "skip",
        ]

    print(f"[KAGGLE] Uploading PRIVATE dataset {dataset_id} from {staging_dir}...", flush=True)
    subprocess.run(cmd, check=True, env=env)
    print(f"[DONE] Dataset ready: https://www.kaggle.com/datasets/{dataset_id}", flush=True)
    return dataset_id


def build_and_upload_dataset(
    *,
    dataset_slug: str = "qwen-image-21-int8-convrot",
    title: str = "Qwen-Image 2.1 INT8 ConvRot Weights",
    preset: str = "full_i2i",
    owner: str = "",
    staging_dir: Path = Path("/tmp/qwen21_dataset_staging"),
    venv_dir: Path = DEFAULT_VENV_DIR,
    update_existing: bool = False,
) -> str:
    """Download weights to /tmp on CPU and push a PRIVATE Kaggle Dataset (flat layout)."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset={preset!r}; expected one of {tuple(PRESETS)}")

    username, auth_env = _resolve_kaggle_auth(owner=owner)
    py_exe = _ensure_prep_venv(venv_dir)
    _stage_flat_assets(py_exe, staging_dir, preset)
    return _upload_private_kaggle_dataset(
        py_exe,
        staging_dir,
        username=username,
        auth_env=auth_env,
        dataset_slug=dataset_slug,
        title=title,
        update_existing=update_existing,
        version_notes=f"Update Qwen-Image 2.1 preset={preset}",
    )


def build_and_upload_runtime_cache(
    *,
    dataset_slug: str = RUNTIME_CACHE_SLUG,
    title: str = "Qwen-Image 2.1 T4 Runtime Cache (Wheels + ComfyUI + cloudflared)",
    owner: str = "",
    staging_dir: Path = Path("/tmp/qwen21_runtime_staging"),
    work_dir: Path = Path("/tmp/qwen21_runtime_work"),
    venv_dir: Path = DEFAULT_VENV_DIR,
    update_existing: bool = False,
    include_xformers: bool = False,
) -> str:
    """Stage offline Python wheels + ComfyUI repo + cloudflared on CPU and upload as a PRIVATE Kaggle Dataset."""
    username, auth_env = _resolve_kaggle_auth(owner=owner)
    py_exe = _ensure_prep_venv(venv_dir, include_pip=True)
    stage_runtime_cache(
        staging_dir,
        work_dir=work_dir,
        venv_dir=venv_dir,
        include_xformers=include_xformers,
    )
    return _upload_private_kaggle_dataset(
        py_exe,
        staging_dir,
        username=username,
        auth_env=auth_env,
        dataset_slug=dataset_slug,
        title=title,
        update_existing=update_existing,
        version_notes=f"Update T4 runtime cache (py {sys.version_info[0]}.{sys.version_info[1]})",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage Qwen-Image 2.1 weights or runtime cache into private Kaggle Datasets.")
    parser.add_argument("--target", default="weights", choices=("weights", "runtime_cache", "both"))
    parser.add_argument("--slug", default="qwen-image-21-int8-convrot")
    parser.add_argument("--title", default="Qwen-Image 2.1 INT8 ConvRot Weights")
    parser.add_argument("--runtime-slug", default=RUNTIME_CACHE_SLUG)
    parser.add_argument("--preset", default="full_i2i", choices=tuple(PRESETS))
    parser.add_argument("--owner", default="")
    parser.add_argument("--staging-dir", type=Path, default=Path("/tmp/qwen21_dataset_staging"))
    parser.add_argument("--runtime-staging-dir", type=Path, default=Path("/tmp/qwen21_runtime_staging"))
    parser.add_argument("--venv-dir", type=Path, default=DEFAULT_VENV_DIR)
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()

    if args.target in ("weights", "both"):
        build_and_upload_dataset(
            dataset_slug=args.slug,
            title=args.title,
            preset=args.preset,
            owner=args.owner,
            staging_dir=args.staging_dir,
            venv_dir=args.venv_dir,
            update_existing=args.update,
        )
    if args.target in ("runtime_cache", "both"):
        build_and_upload_runtime_cache(
            dataset_slug=args.runtime_slug,
            owner=args.owner,
            staging_dir=args.runtime_staging_dir,
            venv_dir=args.venv_dir,
            update_existing=args.update,
        )


if __name__ == "__main__":
    main()
