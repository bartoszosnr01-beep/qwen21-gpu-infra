"""Dual-T4 ComfyUI gateway server with Staggered Load Lock, Stage-1/2 orchestration, and SSE."""

from __future__ import annotations

import argparse
import asyncio
import base64
from collections import deque
from dataclasses import dataclass, field
import hashlib
import hmac
import io
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import shutil
import struct
import subprocess
import time
from typing import Any
import urllib.parse
import urllib.request
import uuid

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web

from qwen_kaggle.bootstrap import (
    collect_system_diagnostics,
    ensure_models_and_templates,
    find_input_asset,
    sequential_copy_file,
    setup_comfy_and_venv,
    venv_python_path,
)
from qwen_kaggle.constants import (
    CLOUDFLARED_LINUX_URL,
    DEFAULT_COMFY_DIR,
    DEFAULT_INPUT_DIR,
    DEFAULT_VENV_DIR,
    MODELS,
    PRESETS,
    RUNTIME_CLOUDFLARED_FILENAME,
    SAFE_COMFY_FLAGS,
    TAESD_DECODER_FILENAME,
    TAESD_DECODER_SUBDIR,
    TPU_SAFE_COMFY_FLAGS,
)
from qwen_kaggle.workflows import (
    GenerationRequest,
    IDENTITY_BLEND_MODES,
    MAX_REFERENCE_IMAGES,
    build_qwen21_api_prompt,
    build_refine_prompt_graph,
)

TRYCLOUDFLARE_RE = re.compile(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com")
MAX_IMAGE_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_RETAINED_IMAGES = 24
MAX_RETAINED_JOBS = 120
PREVIEW_MIN_INTERVAL_S = 0.65

BINARY_EVENT_PREVIEW_IMAGE = 1
BINARY_EVENT_PREVIEW_IMAGE_WITH_METADATA = 4


def parse_comfy_preview_frame(frame: bytes) -> tuple[str, bytes] | None:
    """Decode ComfyUI binary WebSocket preview frames (`PREVIEW_IMAGE` or `PREVIEW_IMAGE_WITH_METADATA`)."""
    if len(frame) < 8:
        return None
    (event_type,) = struct.unpack(">I", frame[:4])
    if event_type == BINARY_EVENT_PREVIEW_IMAGE:
        (image_type_num,) = struct.unpack(">I", frame[4:8])
        mime = "image/png" if image_type_num == 2 else "image/jpeg"
        raw = frame[8:]
        return (mime, raw) if raw else None

    if event_type == BINARY_EVENT_PREVIEW_IMAGE_WITH_METADATA:
        (meta_len,) = struct.unpack(">I", frame[4:8])
        if meta_len < 0 or 8 + meta_len >= len(frame):
            return None
        mime = "image/jpeg"
        try:
            meta = json.loads(frame[8 : 8 + meta_len].decode("utf-8"))
            if isinstance(meta, dict) and isinstance(meta.get("image_type"), str):
                mime = meta["image_type"]
        except Exception:
            pass
        raw = frame[8 + meta_len :]
        return (mime, raw) if raw else None

    return None


def parse_png_dimensions(raw: bytes) -> tuple[int, int] | None:
    """Extract `(width, height)` from a PNG IHDR chunk without requiring Pillow."""
    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
        return None
    w, h = struct.unpack(">II", raw[16:24])
    return (int(w), int(h)) if w > 0 and h > 0 else None


def inspect_installed_models(comfy_dir: Path) -> dict[str, dict[str, Any]]:
    """Inspect ComfyUI model directories for Qwen-Image 2.1 weights and TAESD decoder."""
    models_root = comfy_dir / "models"
    status: dict[str, dict[str, Any]] = {}
    for key, spec in MODELS.items():
        path = models_root / spec.subdir / spec.filename
        exists = path.exists() and path.stat().st_size > 0
        size_gb = round(path.stat().st_size / (1024**3), 2) if exists else 0.0
        status[key] = {
            "key": key,
            "filename": spec.filename,
            "subdir": spec.subdir,
            "approx_gib": spec.approx_gib,
            "installed": exists,
            "size_gib": size_gb,
            "description": spec.description,
        }
    taesd_path = models_root / TAESD_DECODER_SUBDIR / TAESD_DECODER_FILENAME
    taesd_ok = taesd_path.exists() and taesd_path.stat().st_size > 0
    status["taesd"] = {
        "key": "taesd",
        "filename": TAESD_DECODER_FILENAME,
        "subdir": TAESD_DECODER_SUBDIR,
        "approx_gib": 0.02,
        "installed": taesd_ok,
        "size_gib": round(taesd_path.stat().st_size / (1024**3), 3) if taesd_ok else 0.0,
        "description": "TAESD live preview decoder",
    }
    return status


def required_model_keys(
    *,
    action: str,
    mode: str,
    profile: str,
    enhance_prompt: bool,
) -> tuple[str, ...]:
    """Return exact model catalog keys required for a gateway job."""
    keys: list[str] = []
    if action == "refine" or enhance_prompt:
        keys.append("pe_i2i" if mode == "edit" else "pe_t2i")
    if action == "generate":
        keys.append("unet_turbo" if profile == "fast" else "unet_base")
        keys.append("clip_vl")
        keys.append("vae")
    return tuple(dict.fromkeys(keys))


def _decode_uploaded_image(item: Any, index: int) -> tuple[str, bytes]:
    if not isinstance(item, dict):
        raise ValueError(f"images[{index}] must be an object.")
    raw_name = str(item.get("filename") or f"ref_{index + 1}.png").strip()
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", Path(raw_name).name) or f"ref_{index + 1}.png"
    if len(safe_name) > 96:
        safe_name = safe_name[-96:]

    data_b64 = ""
    if isinstance(item.get("data_url"), str) and item["data_url"].strip():
        data_url = item["data_url"].strip()
        if "," not in data_url:
            raise ValueError(f"images[{index}].data_url is malformed.")
        data_b64 = data_url.split(",", 1)[1]
    elif isinstance(item.get("content_base64"), str) and item["content_base64"].strip():
        data_b64 = item["content_base64"].strip()
    else:
        raise ValueError(f"images[{index}] requires data_url or content_base64.")

    try:
        raw_bytes = base64.b64decode(data_b64, validate=True)
    except Exception as exc:
        raise ValueError(f"images[{index}] has invalid base64 payload.") from exc

    if not raw_bytes:
        raise ValueError(f"images[{index}] is empty.")
    if len(raw_bytes) > MAX_IMAGE_UPLOAD_BYTES:
        raise ValueError(f"images[{index}] exceeds {MAX_IMAGE_UPLOAD_BYTES // (1024 * 1024)} MiB limit.")

    digest = hashlib.sha256(raw_bytes).hexdigest()[:8]
    stem = Path(safe_name).stem or f"ref_{index + 1}"
    ext = Path(safe_name).suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp"):
        ext = ".png"
    return f"{stem}_{digest}{ext}", raw_bytes


def ensure_cloudflared_binary(
    bin_path: Path = Path("/tmp/cloudflared"),
    input_root: Path = DEFAULT_INPUT_DIR,
) -> Path:
    """Copy cached `cloudflared` from Kaggle input when present, else download it."""
    if bin_path.exists() and bin_path.stat().st_size > 1_000_000:
        os.chmod(bin_path, 0o755)
        return bin_path
    for candidate_name in (RUNTIME_CLOUDFLARED_FILENAME, "cloudflared-linux-amd64"):
        cached = find_input_asset(candidate_name, input_root)
        if cached is not None and cached.stat().st_size > 1_000_000:
            bin_path.parent.mkdir(parents=True, exist_ok=True)
            sequential_copy_file(cached, bin_path)
            os.chmod(bin_path, 0o755)
            print(f"[TUNNEL] Restored cached cloudflared -> {bin_path}", flush=True)
            return bin_path
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = bin_path.with_suffix(".tmp")
    print(f"[TUNNEL] Downloading cloudflared -> {bin_path}...", flush=True)
    with urllib.request.urlopen(CLOUDFLARED_LINUX_URL, timeout=90) as resp, tmp_path.open("wb") as out:
        shutil.copyfileobj(resp, out, length=4 * 1024 * 1024)
    os.chmod(tmp_path, 0o755)
    tmp_path.replace(bin_path)
    return bin_path


class CloudflaredTunnel:
    """Async quick-tunnel subprocess wrapper capturing the `trycloudflare.com` HTTPS origin."""

    def __init__(
        self,
        target_port: int,
        bin_path: Path = Path("/tmp/cloudflared"),
        input_root: Path = DEFAULT_INPUT_DIR,
    ) -> None:
        self.target_port = target_port
        self.bin_path = bin_path
        self.input_root = input_root
        self.proc: asyncio.subprocess.Process | None = None
        self.public_url: str | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._url_ready = asyncio.Event()
        self._logs: deque[str] = deque(maxlen=60)

    async def start(self, timeout_s: float = 45.0) -> str:
        bin_exe = await asyncio.to_thread(ensure_cloudflared_binary, self.bin_path, self.input_root)
        self.proc = await asyncio.create_subprocess_exec(
            str(bin_exe),
            "tunnel",
            "--url",
            f"http://127.0.0.1:{self.target_port}",
            "--no-autoupdate",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self._reader_task = asyncio.create_task(self._read_output())
        try:
            await asyncio.wait_for(self._url_ready.wait(), timeout=timeout_s)
        except asyncio.TimeoutError as exc:
            tail = "\n".join(self._logs)
            await self.stop()
            raise RuntimeError(f"Timed out waiting for cloudflared URL:\n{tail}") from exc
        assert self.public_url is not None
        return self.public_url

    async def _read_output(self) -> None:
        if not self.proc or not self.proc.stdout:
            return
        while True:
            line_bytes = await self.proc.stdout.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace").rstrip()
            self._logs.append(line)
            match = TRYCLOUDFLARE_RE.search(line)
            if match and not self.public_url:
                self.public_url = match.group(0)
                self._url_ready.set()

    async def stop(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=8.0)
            except asyncio.TimeoutError:
                self.proc.kill()
                await self.proc.wait()
        self.proc = None


@dataclass
class WorkerSlot:
    gpu_index: int
    port: int
    base_url: str
    managed_process: bool = True
    proc: subprocess.Popen[str] | None = None
    log_path: Path | None = None
    state: str = "starting"  # starting | idle | busy | error
    current_job_id: str | None = None
    completed_count: int = 0
    failed_count: int = 0
    vram_used_mb: float | None = None
    vram_total_mb: float | None = None
    last_error: str | None = None
    last_loaded_profile: str | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "gpu_index": self.gpu_index,
            "port": self.port,
            "state": self.state,
            "current_job_id": self.current_job_id,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "vram_used_mb": self.vram_used_mb,
            "vram_total_mb": self.vram_total_mb,
            "last_error": self.last_error,
            "last_loaded_profile": self.last_loaded_profile,
        }


def compute_face_centered_square_crop(
    img_w: int,
    img_h: int,
    face_box: tuple[int, int, int, int] | None,
    *,
    target_face_ratio: float = 0.44,
    max_expand_factor: float = 1.4,
) -> tuple[int, int, int, int]:
    """Compute a square crop box `(x0, y0, x1, y1)` normalized to equal face scale.

    When `face_box` is provided, the crop side equals `max(fw, fh) / target_face_ratio`
    (capped at `min(img_w, img_h) * max_expand_factor`) so wide portraits and tight
    close-ups present identically sized faces inside the output square. Coordinates may
    extend slightly outside `[0, img_w] x [0, img_h]` when a tight close-up needs soft
    border padding to match the target face scale.
    """
    safe_w = max(1, int(img_w))
    safe_h = max(1, int(img_h))

    if face_box is None or face_box[2] <= 0 or face_box[3] <= 0:
        side = min(safe_w, safe_h)
        x0 = (safe_w - side) // 2
        y0 = int(round((safe_h - side) * 0.25)) if safe_h > safe_w else 0
        return (x0, y0, x0 + side, y0 + side)

    fx, fy, fw, fh = face_box
    ratio = max(0.2, min(0.7, float(target_face_ratio)))
    face_dim = max(float(fw), float(fh))
    desired_side = face_dim / ratio
    max_allowed_side = max(float(min(safe_w, safe_h)) * max(1.0, float(max_expand_factor)), face_dim * 1.15)
    side = max(16, int(round(min(desired_side, max_allowed_side))))

    cx = fx + fw / 2.0
    cy = fy + fh * 0.48
    half = side / 2.0
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))

    if side <= safe_w:
        x0 = max(0, min(safe_w - side, x0))
    if side <= safe_h:
        y0 = max(0, min(safe_h - side, y0))

    return (x0, y0, x0 + side, y0 + side)


def _select_primary_face(
    faces: Any,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int] | None:
    """Pick the primary portrait subject (largest + closest to frame center)."""
    best_box: tuple[int, int, int, int] | None = None
    best_score = -1.0
    half_diag = max(1.0, math.hypot(img_w / 2.0, img_h / 2.0))
    for item in faces:
        fx, fy, fw, fh = int(item[0]), int(item[1]), int(item[2]), int(item[3])
        if fw <= 0 or fh <= 0:
            continue
        cx = fx + fw / 2.0
        cy = fy + fh / 2.0
        dist_norm = min(1.0, math.hypot(cx - img_w / 2.0, cy - img_h / 2.0) / half_diag)
        score = float(fw * fh) * (1.0 - 0.35 * dist_norm)
        if score > best_score:
            best_score = score
            best_box = (fx, fy, fw, fh)
    return best_box


def normalize_reference_face_crops(
    images: list[tuple[str, bytes]],
    *,
    target_size: int = 768,
    target_face_ratio: float = 0.44,
) -> tuple[list[tuple[str, bytes]], int]:
    """Detect the primary face in each reference image and crop to equal face scale 1:1.

    Returns `(normalized_images, faces_detected_count)`. Gracefully falls back to
    original images when OpenCV/Pillow are unavailable in lightweight environments.
    """
    if len(images) < 2:
        return images, 0
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
        from PIL import Image, ImageOps  # type: ignore[import-not-found]
    except Exception:
        return images, 0

    cascade_dir = getattr(getattr(cv2, "data", None), "haarcascades", "")
    cascade_paths = [
        os.path.join(cascade_dir, name)
        for name in (
            "haarcascade_frontalface_alt2.xml",
            "haarcascade_frontalface_default.xml",
            "haarcascade_profileface.xml",
        )
        if cascade_dir
    ]
    cascades = []
    for p in cascade_paths:
        if os.path.isfile(p):
            clf = cv2.CascadeClassifier(p)
            if not clf.empty():
                cascades.append((os.path.basename(p), clf))

    size = max(256, min(1024, int(round(target_size / 32) * 32)))
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", getattr(Image, "LANCZOS", 1))
    out: list[tuple[str, bytes]] = []
    faces_detected = 0

    for fname, raw_bytes in images:
        try:
            with Image.open(io.BytesIO(raw_bytes)) as opened:
                pil_img = ImageOps.exif_transpose(opened).convert("RGB")
        except Exception:
            out.append((fname, raw_bytes))
            continue

        img_w, img_h = pil_img.size
        rgb = np.asarray(pil_img)
        gray = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        min_face_px = max(28, int(min(img_w, img_h) * 0.08))

        best_face: tuple[int, int, int, int] | None = None
        for cname, clf in cascades:
            for min_neighbors in (5, 3):
                detected = clf.detectMultiScale(
                    gray,
                    scaleFactor=1.08,
                    minNeighbors=min_neighbors,
                    minSize=(min_face_px, min_face_px),
                )
                if len(detected) > 0:
                    best_face = _select_primary_face(detected, img_w, img_h)
                    if best_face is not None:
                        break
            if best_face is not None:
                break
            if "profile" in cname:
                flipped = cv2.flip(gray, 1)
                detected_flip = clf.detectMultiScale(
                    flipped,
                    scaleFactor=1.08,
                    minNeighbors=4,
                    minSize=(min_face_px, min_face_px),
                )
                if len(detected_flip) > 0:
                    mirrored = [
                        (img_w - int(x) - int(w), int(y), int(w), int(h))
                        for (x, y, w, h) in detected_flip
                    ]
                    best_face = _select_primary_face(mirrored, img_w, img_h)
                    if best_face is not None:
                        break

        if best_face is not None:
            faces_detected += 1

        x0, y0, x1, y1 = compute_face_centered_square_crop(
            img_w,
            img_h,
            best_face,
            target_face_ratio=target_face_ratio,
        )

        pad_left = max(0, -x0)
        pad_top = max(0, -y0)
        pad_right = max(0, x1 - img_w)
        pad_bottom = max(0, y1 - img_h)

        if pad_left or pad_top or pad_right or pad_bottom:
            padded = cv2.copyMakeBorder(
                rgb,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_REPLICATE,
            )
            blurred = cv2.GaussianBlur(padded, (0, 0), sigmaX=12.0, sigmaY=12.0)
            blurred[pad_top : pad_top + img_h, pad_left : pad_left + img_w] = rgb
            crop_arr = blurred[
                y0 + pad_top : y1 + pad_top,
                x0 + pad_left : x1 + pad_left,
            ]
        else:
            crop_arr = rgb[y0:y1, x0:x1]

        cropped_img = Image.fromarray(crop_arr).resize((size, size), resample)
        buf = io.BytesIO()
        cropped_img.save(buf, format="PNG")
        stem = Path(fname).stem or "ref"
        out.append((f"{stem}_facecrop.png", buf.getvalue()))

    return out, faces_detected


@dataclass
class GatewayJob:
    id: str
    action: str  # "generate" | "refine"
    mode: str  # "t2i" | "edit"
    profile: str  # "fast" | "quality"
    prompt: str
    negative_prompt: str
    width: int
    height: int
    steps: int | None
    cfg: float | None
    seed: int
    sampler_name: str
    scheduler: str
    resolution: int
    enhance_prompt: bool
    thinking: bool
    system_prompt: str
    use_image_latent: bool
    cache_device: str
    cache_dtype: str
    preferred_gpu: int | None
    identity_blend_mode: str = "none"
    identity_blend_ratio: float = 0.5
    normalize_face_crops: bool = False
    face_crops_applied: bool = False
    images: list[tuple[str, bytes]] = field(default_factory=list)
    effective_steps: int = 0
    effective_cfg: float = 1.0
    accelerator: str = "gpu_t4_x2"
    output_width: int | None = None
    output_height: int | None = None
    warm_worker: bool | None = None
    queue_wait_s: float | None = None
    upload_s: float | None = None
    stage1_s: float | None = None
    load_s: float | None = None
    sampling_s: float | None = None
    vae_s: float | None = None
    sec_per_step: float | None = None
    ms_per_step_mp: float | None = None
    status: str = "queued"  # queued | running | completed | failed | cancelled
    stage: str = "queued"
    progress: int = 0
    max_steps: int = 0
    percent: int = 0
    gpu_index: int | None = None
    enhanced_prompt: str | None = None
    thinking_text: str | None = None
    preview_data_url: str | None = None
    error: str | None = None
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    elapsed_s: float | None = None
    image_bytes: bytes | None = None
    output_filename: str | None = None
    cancel_requested: bool = False

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "action": self.action,
            "mode": self.mode,
            "profile": self.profile,
            "accelerator": self.accelerator,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "width": self.width,
            "height": self.height,
            "output_width": self.output_width,
            "output_height": self.output_height,
            "steps": self.steps,
            "effective_steps": self.effective_steps,
            "cfg": self.cfg,
            "effective_cfg": self.effective_cfg,
            "seed": self.seed,
            "enhance_prompt": self.enhance_prompt,
            "thinking": self.thinking,
            "use_image_latent": self.use_image_latent,
            "identity_blend_mode": self.identity_blend_mode,
            "identity_blend_ratio": self.identity_blend_ratio,
            "normalize_face_crops": self.normalize_face_crops,
            "face_crops_applied": self.face_crops_applied,
            "cache_device": self.cache_device,
            "cache_dtype": self.cache_dtype,
            "preferred_gpu": self.preferred_gpu,
            "input_images": [name for name, _ in self.images],
            "input_count": len(self.images),
            "warm_worker": self.warm_worker,
            "queue_wait_s": self.queue_wait_s,
            "upload_s": self.upload_s,
            "stage1_s": self.stage1_s,
            "load_s": self.load_s,
            "sampling_s": self.sampling_s,
            "vae_s": self.vae_s,
            "sec_per_step": self.sec_per_step,
            "ms_per_step_mp": self.ms_per_step_mp,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "max_steps": self.max_steps,
            "percent": self.percent,
            "gpu_index": self.gpu_index,
            "enhanced_prompt": self.enhanced_prompt,
            "thinking_text": self.thinking_text,
            "preview_data_url": self.preview_data_url,
            "error": self.error,
            "created_at_ms": self.created_at_ms,
            "started_at_ms": self.started_at_ms,
            "finished_at_ms": self.finished_at_ms,
            "elapsed_s": self.elapsed_s,
            "image_available": self.image_bytes is not None,
            "output_filename": self.output_filename,
        }


@dataclass
class GatewayConfig:
    comfy_dir: Path = DEFAULT_COMFY_DIR
    venv_dir: Path = DEFAULT_VENV_DIR
    input_dir: Path = DEFAULT_INPUT_DIR
    log_dir: Path = Path("/tmp/qwen_gateway_logs")
    host: str = "127.0.0.1"
    port: int = 8000
    bearer_token: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    worker_urls: tuple[str, ...] | None = None
    base_worker_port: int = 8188
    num_workers: int = 2
    model_preset: str = "full_i2i"
    accelerator: str = "gpu"  # "gpu" | "tpu"
    extra_comfy_flags: tuple[str, ...] = SAFE_COMFY_FLAGS
    ws_poll_interval_s: float = 1.5
    max_load_lock_hold_s: float = 90.0
    prompt_timeout_s: float = 1800.0  # max seconds per ComfyUI prompt; <= 0 disables
    idle_timeout_s: float = 600.0  # auto-shutdown after this many idle seconds; <= 0 disables


class QwenGateway:
    """Async coordinator managing isolated ComfyUI GPU workers, Stage-1/Stage-2 flow, and SSE."""

    def __init__(self, config: GatewayConfig) -> None:
        self.config = config
        self.started_at_ms = int(time.time() * 1000)
        self.workers: list[WorkerSlot] = []
        if config.worker_urls:
            for idx, raw_url in enumerate(config.worker_urls):
                parsed = urllib.parse.urlparse(raw_url)
                self.workers.append(
                    WorkerSlot(
                        gpu_index=idx,
                        port=parsed.port or (config.base_worker_port + idx),
                        base_url=raw_url.rstrip("/"),
                        managed_process=False,
                        state="idle",
                    )
                )
        else:
            for idx in range(max(1, config.num_workers)):
                port = config.base_worker_port + idx
                self.workers.append(
                    WorkerSlot(
                        gpu_index=idx,
                        port=port,
                        base_url=f"http://127.0.0.1:{port}",
                        managed_process=True,
                        log_path=config.log_dir / f"comfy_gpu{idx}.log",
                        state="starting",
                    )
                )

        self._http_client: Any = None
        self._dispatch_cond = asyncio.Condition()
        self._pending_jobs: deque[str] = deque()
        self._jobs: dict[str, GatewayJob] = {}
        self._job_order: deque[str] = deque()
        self._image_order: deque[str] = deque()
        self._subscribers: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._vram_task: asyncio.Task[None] | None = None
        self._model_sync_task: asyncio.Task[None] | None = None
        self._model_sync_state: dict[str, Any] = {
            "running": False,
            "preset": config.model_preset,
            "last_error": None,
            "completed_at_ms": None,
        }
        # Serializes heavy Stage-1 9B loads and short initial model reads to guard 29 GB system RAM.
        self._load_lock = asyncio.Lock()
        self._active_ws: dict[int, Any] = {}
        self._shutting_down = False
        self._cached_gpu_vram: dict[int, tuple[float, float]] = {}
        self.shutdown_event = asyncio.Event()
        self.shutdown_reason: str | None = None
        self._last_busy_mono = time.monotonic()
        self._idle_bonus_s: float = 0.0

    def is_busy(self) -> bool:
        """True while jobs are queued/running, workers boot, or a model sync is in progress."""
        if self._pending_jobs or self._model_sync_state.get("running"):
            return True
        return any(w.state in ("busy", "starting") for w in self.workers)

    def touch_activity(self) -> None:
        self._last_busy_mono = time.monotonic()
        self._idle_bonus_s = 0.0

    def idle_seconds(self) -> float:
        if self.is_busy():
            self.touch_activity()
        return max(0.0, time.monotonic() - self._last_busy_mono)

    def effective_idle_timeout_s(self) -> float:
        """Return base idle timeout plus any active user-requested extension."""
        if self.config.idle_timeout_s <= 0:
            return 0.0
        return self.config.idle_timeout_s + self._idle_bonus_s

    def shutdown_in_seconds(self) -> float | None:
        """Seconds remaining before idle auto-shutdown triggers, or None when disabled."""
        effective = self.effective_idle_timeout_s()
        if effective <= 0:
            return None
        return max(0.0, effective - self.idle_seconds())

    def extend_idle(self, extra_seconds: float = 600.0) -> bool:
        """Add extra idle seconds before auto-shutdown; returns False if already shutting down."""
        if self.shutdown_event.is_set():
            return False
        if self.config.idle_timeout_s <= 0:
            self.touch_activity()
            return True
        add_s = max(0.1, min(7200.0, float(extra_seconds)))
        current_remaining = self.shutdown_in_seconds() or 0.0
        new_remaining = min(28800.0, current_remaining + add_s)
        self._last_busy_mono = time.monotonic()
        self._idle_bonus_s = max(0.0, new_remaining - self.config.idle_timeout_s)
        print(
            f"[GATEWAY] Idle timer extended by {add_s:.0f}s (shutdown in {new_remaining:.0f}s).",
            flush=True,
        )
        return True

    def build_idle_payload(self) -> dict[str, Any]:
        """Build public idle countdown metadata for status and extend responses."""
        idle_s = self.idle_seconds()
        rem = self.shutdown_in_seconds()
        eff = self.effective_idle_timeout_s()
        return {
            "timeout_s": self.config.idle_timeout_s,
            "effective_timeout_s": round(eff, 1),
            "idle_s": round(idle_s, 1),
            "shutdown_in_s": round(rem, 1) if rem is not None else None,
            "busy": self.is_busy(),
            "shutting_down": self.shutdown_event.is_set(),
            "shutdown_reason": self.shutdown_reason,
        }

    def request_shutdown(self, reason: str) -> bool:
        """Ask the server loop to exit; returns False if shutdown was already requested."""
        if self.shutdown_event.is_set():
            return False
        self.shutdown_reason = reason
        self.shutdown_event.set()
        return True

    async def idle_watchdog(self) -> None:
        """Request shutdown after `idle_timeout_s` without queued/running work (status polling does not count)."""
        timeout = self.config.idle_timeout_s
        if timeout <= 0:
            return
        while not self.shutdown_event.is_set():
            await asyncio.sleep(min(5.0, max(0.05, timeout / 4)))
            rem = self.shutdown_in_seconds()
            if rem is not None and rem <= 0.0:
                self.request_shutdown(f"idle_timeout_{int(timeout)}s")
                return

    async def start(self) -> None:
        timeout = ClientTimeout(total=None, sock_connect=15, sock_read=120)
        self._http_client = ClientSession(timeout=timeout)
        self.config.log_dir.mkdir(parents=True, exist_ok=True)

        # Boot ComfyUI worker processes in parallel (with tiny spawn stagger; Staggered Load Lock serializes model reads later)
        for idx, worker in enumerate(self.workers):
            if worker.managed_process:
                await asyncio.to_thread(self._start_managed_worker_sync, worker)
                if idx + 1 < len(self.workers):
                    await asyncio.sleep(1.0)
        await asyncio.gather(*(self._wait_worker_ready(worker) for worker in self.workers))

        await self._refresh_vram_once()
        self._vram_task = asyncio.create_task(self._vram_poll_loop())
        self._worker_tasks = [
            asyncio.create_task(self._worker_loop(worker)) for worker in self.workers
        ]

    async def stop(self) -> None:
        self._shutting_down = True
        async with self._dispatch_cond:
            self._dispatch_cond.notify_all()

        for task in [self._vram_task, self._model_sync_task, *self._worker_tasks]:
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        for worker in self.workers:
            if worker.managed_process and worker.proc is not None:
                await asyncio.to_thread(self._stop_managed_worker_sync, worker)

        if self._http_client is not None:
            await self._http_client.close()
            self._http_client = None

    def _start_managed_worker_sync(self, worker: WorkerSlot) -> None:
        py_exe = venv_python_path(self.config.venv_dir)
        if not py_exe.exists():
            raise RuntimeError(f"ComfyUI venv Python not found: {py_exe}")
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        if self.config.accelerator == "tpu":
            pkg_root = str(Path(__file__).resolve().parent.parent)
            existing_pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{pkg_root}{os.pathsep}{existing_pp}" if existing_pp else pkg_root
            env.pop("TPU_LIBRARY_PATH", None)
            env.setdefault("PJRT_DEVICE", "TPU")
            worker.vram_total_mb = 131072.0
            cmd = [
                str(py_exe),
                "-m",
                "qwen_kaggle.tpu_runtime",
                "--comfy-dir",
                str(self.config.comfy_dir),
                "--listen",
                "127.0.0.1",
                "--port",
                str(worker.port),
                *TPU_SAFE_COMFY_FLAGS,
            ]
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(worker.gpu_index)
            cmd = [
                str(py_exe),
                "main.py",
                "--listen",
                "127.0.0.1",
                "--port",
                str(worker.port),
                *self.config.extra_comfy_flags,
            ]
        assert worker.log_path is not None
        worker.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = worker.log_path.open("w", encoding="utf-8")
        print(f"[COMFY:{worker.gpu_index}] Starting on port {worker.port} ({self.config.accelerator})...", flush=True)
        worker.state = "starting"
        worker.proc = subprocess.Popen(
            cmd,
            cwd=str(self.config.comfy_dir),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
        worker._log_fh = log_fh  # type: ignore[attr-defined]

    def _stop_managed_worker_sync(self, worker: WorkerSlot) -> None:
        proc = worker.proc
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        worker.proc = None
        fh = getattr(worker, "_log_fh", None)
        if fh is not None:
            fh.close()

    async def _wait_worker_ready(self, worker: WorkerSlot, timeout_s: float = 150.0) -> None:
        deadline = time.monotonic() + timeout_s
        last_err = ""
        while time.monotonic() < deadline:
            if worker.managed_process and worker.proc is not None and worker.proc.poll() is not None:
                tail = ""
                if worker.log_path and worker.log_path.exists():
                    tail = worker.log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
                worker.state = "error"
                worker.last_error = f"ComfyUI GPU{worker.gpu_index} exited unexpectedly:\n{tail}"
                raise RuntimeError(worker.last_error)
            try:
                async with self._http_client.get(f"{worker.base_url}/system_stats") as resp:
                    if resp.status == 200:
                        worker.state = "idle"
                        worker.last_error = None
                        print(f"[COMFY:{worker.gpu_index}] Ready at {worker.base_url}", flush=True)
                        return
            except Exception as exc:
                last_err = str(exc)
            await asyncio.sleep(1.0)
        worker.state = "error"
        worker.last_error = f"Timed out waiting for worker GPU{worker.gpu_index}: {last_err}"
        raise RuntimeError(worker.last_error)

    async def _ensure_worker_alive(self, worker: WorkerSlot) -> None:
        if not worker.managed_process:
            return
        if worker.proc is not None and worker.proc.poll() is None:
            return
        print(f"[COMFY:{worker.gpu_index}] Worker exited; restarting cleanly...", flush=True)
        async with self._load_lock:
            await asyncio.to_thread(self._stop_managed_worker_sync, worker)
            await asyncio.to_thread(self._start_managed_worker_sync, worker)
            await self._wait_worker_ready(worker)

    async def _refresh_vram_once(self) -> None:
        if self.config.accelerator == "tpu":
            for worker in self.workers:
                worker.vram_total_mb = 131072.0
                if worker.vram_used_mb is None:
                    worker.vram_used_mb = 0.0
            return
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            return
        try:
            proc = await asyncio.create_subprocess_exec(
                nvidia_smi,
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=4.0)
            if proc.returncode != 0:
                return
            for line in stdout.decode("utf-8", errors="replace").splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 3 and parts[0].isdigit():
                    idx = int(parts[0])
                    used = float(parts[1])
                    total = float(parts[2])
                    self._cached_gpu_vram[idx] = (used, total)
                    for worker in self.workers:
                        if worker.gpu_index == idx:
                            worker.vram_used_mb = used
                            worker.vram_total_mb = total
        except Exception:
            pass

    async def _vram_poll_loop(self) -> None:
        while not self._shutting_down:
            await self._refresh_vram_once()
            await asyncio.sleep(2.5)

    async def _post_free(self, worker: WorkerSlot) -> None:
        payload = {"unload_models": True, "free_memory": True}
        async with self._http_client.post(f"{worker.base_url}/free", json=payload) as resp:
            raw = (await resp.read()).decode("utf-8", errors="replace").strip()
            if resp.status >= 400:
                raise RuntimeError(f"POST /free failed ({resp.status}): {raw}")
            if raw:
                json.loads(raw)
        worker.last_loaded_profile = None

    async def _upload_image_to_worker(
        self,
        worker: WorkerSlot,
        filename: str,
        content: bytes,
    ) -> str:
        import aiohttp

        form = aiohttp.FormData()
        form.add_field(
            "image",
            content,
            filename=filename,
            content_type="application/octet-stream",
        )
        form.add_field("overwrite", "true")
        async with self._http_client.post(f"{worker.base_url}/upload/image", data=form) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"Image upload failed ({resp.status}): {text}")
            data = json.loads(text)
            return str(data["name"])

    async def _acquire_load_lock(self, job: GatewayJob) -> None:
        while True:
            if job.cancel_requested or self._shutting_down:
                raise asyncio.CancelledError("Job cancelled while waiting for load lock")
            try:
                await asyncio.wait_for(self._load_lock.acquire(), timeout=0.25)
                if job.cancel_requested or self._shutting_down:
                    if self._load_lock.locked():
                        self._load_lock.release()
                    raise asyncio.CancelledError("Job cancelled while waiting for load lock")
                return
            except asyncio.TimeoutError:
                continue

    async def _probe_history_entry(self, worker: WorkerSlot, prompt_id: str) -> dict[str, Any] | None:
        try:
            async with self._http_client.get(
                f"{worker.base_url}/history/{prompt_id}",
                timeout=ClientTimeout(total=15),
            ) as hist_resp:
                if hist_resp.status >= 400:
                    return None
                hist_json = json.loads(await hist_resp.text())
                entry = hist_json.get(prompt_id)
                return dict(entry) if isinstance(entry, dict) and entry else None
        except Exception:
            return None

    async def _check_prompt_timeout(
        self,
        worker: WorkerSlot,
        prompt_id: str,
        prompt_sent_mono: float,
        phase_timings: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        prompt_timeout_s = self.config.prompt_timeout_s
        if prompt_timeout_s <= 0 or (time.monotonic() - prompt_sent_mono) < prompt_timeout_s:
            return None
        probed = await self._probe_history_entry(worker, prompt_id)
        if probed is not None:
            if phase_timings is not None:
                phase_timings.setdefault("prompt_done_at", time.monotonic())
            return probed
        await self._interrupt_worker(worker, clear_queue=True, wait_for_idle_s=12.0)
        raise RuntimeError(
            f"ComfyUI execution timed out after {prompt_timeout_s:.0f}s on GPU{worker.gpu_index}."
        )

    async def _submit_and_track_prompt(
        self,
        worker: WorkerSlot,
        job: GatewayJob,
        prompt_graph: dict[str, Any],
        *,
        stage_label: str,
        emit_previews: bool,
        release_lock_on_sampling: asyncio.Lock | None = None,
        phase_timings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        client_id = uuid.uuid4().hex
        ws_url = (
            worker.base_url.replace("http://", "ws://", 1).replace("https://", "wss://", 1)
            + f"/ws?clientId={client_id}"
        )
        last_preview_at = 0.0
        owns_load_lock = release_lock_on_sampling is not None
        lock_acquired_at = time.monotonic()
        history_entry: dict[str, Any] | None = None

        def maybe_release_load_lock() -> None:
            nonlocal owns_load_lock
            if owns_load_lock and release_lock_on_sampling is not None:
                owns_load_lock = False
                if release_lock_on_sampling.locked():
                    release_lock_on_sampling.release()

        try:
            if job.cancel_requested:
                raise asyncio.CancelledError("Job cancelled by user")

            async with self._http_client.ws_connect(ws_url, heartbeat=20) as ws:
                self._active_ws[worker.gpu_index] = ws
                payload = {"prompt": prompt_graph, "client_id": client_id}
                async with self._http_client.post(f"{worker.base_url}/prompt", json=payload) as resp:
                    resp_text = await resp.text()
                    if resp.status >= 400:
                        raise RuntimeError(f"ComfyUI /prompt rejected ({resp.status}): {resp_text}")
                    prompt_id = str(json.loads(resp_text)["prompt_id"])

                prompt_sent_mono = time.monotonic()
                if phase_timings is not None:
                    phase_timings["prompt_sent_at"] = prompt_sent_mono

                self._update_job(job, stage=stage_label)

                while True:
                    if job.cancel_requested:
                        await self._interrupt_worker(worker, clear_queue=True, wait_for_idle_s=12.0)
                        raise asyncio.CancelledError("Job cancelled by user")

                    if owns_load_lock and (time.monotonic() - lock_acquired_at) >= self.config.max_load_lock_hold_s:
                        maybe_release_load_lock()

                    timed_out_entry = await self._check_prompt_timeout(
                        worker, prompt_id, prompt_sent_mono, phase_timings
                    )
                    if timed_out_entry is not None:
                        history_entry = timed_out_entry
                        break

                    recv_timeout = self.config.ws_poll_interval_s
                    if self.config.prompt_timeout_s > 0:
                        remaining = self.config.prompt_timeout_s - (time.monotonic() - prompt_sent_mono)
                        recv_timeout = min(recv_timeout, max(0.05, remaining))

                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=recv_timeout)
                    except asyncio.TimeoutError:
                        if job.cancel_requested:
                            await self._interrupt_worker(worker, clear_queue=True, wait_for_idle_s=12.0)
                            raise asyncio.CancelledError("Job cancelled by user")
                        if owns_load_lock and (time.monotonic() - lock_acquired_at) >= self.config.max_load_lock_hold_s:
                            maybe_release_load_lock()
                        if worker.managed_process and worker.proc is not None and worker.proc.poll() is not None:
                            raise RuntimeError(f"ComfyUI GPU{worker.gpu_index} exited unexpectedly during execution.")
                        timed_out_entry = await self._check_prompt_timeout(
                            worker, prompt_id, prompt_sent_mono, phase_timings
                        )
                        if timed_out_entry is not None:
                            history_entry = timed_out_entry
                            break
                        probed = await self._probe_history_entry(worker, prompt_id)
                        if probed is not None:
                            history_entry = probed
                            if phase_timings is not None:
                                phase_timings.setdefault("prompt_done_at", time.monotonic())
                            break
                        continue

                    if msg.type == WSMsgType.TEXT:
                        event = json.loads(msg.data)
                        ev_type = event.get("type")
                        ev_data = event.get("data") or {}
                        if ev_data.get("prompt_id") not in (None, prompt_id):
                            continue
                        if ev_type == "progress":
                            val = int(ev_data.get("value") or 0)
                            mx = int(ev_data.get("max") or 0)
                            pct = int(round((val / mx) * 100)) if mx > 0 else job.percent
                            if phase_timings is not None and val >= 1:
                                if "first_step_done_at" not in phase_timings:
                                    phase_timings["first_step_done_at"] = time.monotonic()
                                    phase_timings["first_step_val"] = val
                                if mx > 0:
                                    phase_timings["last_step_max"] = mx
                            # Once sampler steps begin, heavy disk->RAM loading is finished.
                            maybe_release_load_lock()
                            self._update_job(
                                job,
                                stage=stage_label,
                                progress=val,
                                max_steps=mx,
                                percent=min(99, max(job.percent, pct)),
                            )
                        elif ev_type == "executing":
                            node_id = ev_data.get("node")
                            if node_id is None:
                                if phase_timings is not None:
                                    phase_timings.setdefault("prompt_done_at", time.monotonic())
                                break
                            if str(node_id) == "11":
                                if phase_timings is not None:
                                    phase_timings.setdefault("sampler_enter_at", time.monotonic())
                                maybe_release_load_lock()
                            if str(node_id) == "12":
                                if phase_timings is not None:
                                    phase_timings.setdefault("vae_enter_at", time.monotonic())
                                maybe_release_load_lock()
                                self._update_job(job, stage="decoding", percent=max(job.percent, 96))
                        elif ev_type == "execution_error":
                            maybe_release_load_lock()
                            err_msg = ev_data.get("exception_message") or json.dumps(ev_data)
                            raise RuntimeError(f"ComfyUI execution error: {err_msg}")
                        elif ev_type == "execution_interrupted":
                            maybe_release_load_lock()
                            raise asyncio.CancelledError("Execution interrupted")
                    elif msg.type == WSMsgType.BINARY and emit_previews:
                        now = time.monotonic()
                        if now - last_preview_at >= PREVIEW_MIN_INTERVAL_S:
                            parsed = parse_comfy_preview_frame(bytes(msg.data))
                            if parsed is not None:
                                mime, img_bytes = parsed
                                b64 = base64.b64encode(img_bytes).decode("ascii")
                                last_preview_at = now
                                self._update_job(
                                    job,
                                    preview_data_url=f"data:{mime};base64,{b64}",
                                )
                    elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSED, WSMsgType.CLOSING):
                        break
        finally:
            self._active_ws.pop(worker.gpu_index, None)
            maybe_release_load_lock()

        if job.cancel_requested:
            await self._wait_worker_queue_idle(worker, timeout_s=12.0)
            raise asyncio.CancelledError("Job cancelled by user")

        entry = history_entry
        if entry is None:
            async with self._http_client.get(f"{worker.base_url}/history/{prompt_id}") as hist_resp:
                hist_text = await hist_resp.text()
                if hist_resp.status >= 400:
                    raise RuntimeError(f"Failed fetching /history/{prompt_id}: {hist_text}")
                hist_json = json.loads(hist_text)
            entry = hist_json.get(prompt_id)

        if not entry:
            raise RuntimeError(f"ComfyUI prompt {prompt_id} finished without history entry.")
        status_info = entry.get("status") or {}
        if status_info.get("status_str") == "error":
            raise RuntimeError(f"ComfyUI execution failed: {status_info.get('messages')}")
        return dict(entry.get("outputs") or {})

    async def _wait_worker_queue_idle(self, worker: WorkerSlot, timeout_s: float = 12.0) -> None:
        if timeout_s <= 0 or self._http_client is None:
            return
        deadline = time.monotonic() + timeout_s
        while not self._shutting_down:
            if worker.managed_process and worker.proc is not None and worker.proc.poll() is not None:
                return
            try:
                async with self._http_client.get(
                    f"{worker.base_url}/queue",
                    timeout=ClientTimeout(total=5),
                ) as resp:
                    if resp.status != 200:
                        return
                    payload = json.loads(await resp.text())
                running = payload.get("queue_running") or []
                pending = payload.get("queue_pending") or []
                if not running and not pending:
                    return
            except Exception:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            await asyncio.sleep(min(0.25, remaining))

    async def _interrupt_worker(
        self,
        worker: WorkerSlot,
        *,
        clear_queue: bool = True,
        wait_for_idle_s: float = 0.0,
    ) -> None:
        if self._http_client is not None:
            ctrl_timeout = ClientTimeout(total=10)
            try:
                async with self._http_client.post(
                    f"{worker.base_url}/interrupt",
                    json={},
                    timeout=ctrl_timeout,
                ) as resp:
                    await resp.read()
            except Exception:
                pass
            if clear_queue:
                try:
                    async with self._http_client.post(
                        f"{worker.base_url}/queue",
                        json={"clear": True},
                        timeout=ctrl_timeout,
                    ) as resp:
                        await resp.read()
                except Exception:
                    pass
        active_ws = self._active_ws.get(worker.gpu_index)
        if active_ws is not None and not getattr(active_ws, "closed", True):
            try:
                await active_ws.close()
            except Exception:
                pass
        if wait_for_idle_s > 0:
            await self._wait_worker_queue_idle(worker, timeout_s=wait_for_idle_s)

    def _update_job(self, job: GatewayJob, **changes: Any) -> None:
        self.touch_activity()
        for k, v in changes.items():
            setattr(job, k, v)
        snapshot = job.to_public_dict()
        for q in list(self._subscribers.get(job.id, ())):
            try:
                q.put_nowait(snapshot)
            except asyncio.QueueFull:
                pass

    def _prune_retention(self) -> None:
        while len(self._image_order) > MAX_RETAINED_IMAGES:
            old_id = self._image_order.popleft()
            old_job = self._jobs.get(old_id)
            if old_job is not None:
                old_job.image_bytes = None
        while len(self._job_order) > MAX_RETAINED_JOBS:
            old_id = self._job_order.popleft()
            old_job = self._jobs.get(old_id)
            if old_job is not None and old_job.status in ("completed", "failed", "cancelled"):
                self._jobs.pop(old_id, None)
                self._subscribers.pop(old_id, None)

    async def _take_next_job(self, worker: WorkerSlot) -> GatewayJob | None:
        async with self._dispatch_cond:
            while not self._shutting_down:
                for idx, job_id in enumerate(self._pending_jobs):
                    job = self._jobs.get(job_id)
                    if job is None or job.cancel_requested:
                        continue
                    if job.preferred_gpu in (None, worker.gpu_index):
                        del self._pending_jobs[idx]
                        return job
                await self._dispatch_cond.wait()
            return None

    async def _worker_loop(self, worker: WorkerSlot) -> None:
        while not self._shutting_down:
            job = await self._take_next_job(worker)
            if job is None:
                break

            worker.state = "busy"
            worker.current_job_id = job.id
            job.gpu_index = worker.gpu_index
            started_ms = int(time.time() * 1000)
            job.started_at_ms = started_ms
            job.queue_wait_s = round(max(0.0, (started_ms - job.created_at_ms) / 1000.0), 2)
            self._update_job(job, status="running", stage="preparing", percent=0, gpu_index=worker.gpu_index)
            t0 = time.monotonic()

            try:
                await self._ensure_worker_alive(worker)
                await self._execute_job(worker, job)
                elapsed = round(time.monotonic() - t0, 2)
                worker.completed_count += 1
                self._update_job(
                    job,
                    status="completed",
                    stage="completed",
                    progress=job.max_steps or 1,
                    max_steps=job.max_steps or 1,
                    percent=100,
                    finished_at_ms=int(time.time() * 1000),
                    elapsed_s=elapsed,
                )
            except asyncio.CancelledError:
                elapsed = round(time.monotonic() - t0, 2)
                self._update_job(
                    job,
                    status="cancelled",
                    stage="cancelled",
                    finished_at_ms=int(time.time() * 1000),
                    elapsed_s=elapsed,
                    error="Cancelled by user",
                )
            except Exception as exc:
                elapsed = round(time.monotonic() - t0, 2)
                worker.failed_count += 1
                worker.last_error = str(exc)
                self._update_job(
                    job,
                    status="failed",
                    stage="failed",
                    finished_at_ms=int(time.time() * 1000),
                    elapsed_s=elapsed,
                    error=str(exc),
                )
            finally:
                worker.current_job_id = None
                worker.state = "idle"
                await self._refresh_vram_once()

    async def _execute_job(self, worker: WorkerSlot, job: GatewayJob) -> None:
        uploaded_names: list[str] = []
        if job.images:
            self._update_job(job, stage="uploading_images")
            t_up = time.monotonic()
            images_to_upload = job.images
            if job.normalize_face_crops and len(job.images) >= 2:
                normalized_imgs, faces_detected = await asyncio.to_thread(
                    normalize_reference_face_crops,
                    job.images,
                )
                images_to_upload = normalized_imgs
                if faces_detected > 0:
                    job.face_crops_applied = True
            for idx, (fname, raw_bytes) in enumerate(images_to_upload):
                uploaded = await self._upload_image_to_worker(worker, f"job_{job.id}_{idx}_{fname}", raw_bytes)
                uploaded_names.append(uploaded)
            job.upload_s = round(max(0.0, time.monotonic() - t_up), 2)

        # Stage 1: Prompt Enhancer (qwen3.5_9b) serialized under load_lock + explicit POST /free
        if job.action == "refine" or job.enhance_prompt:
            t_s1 = time.monotonic()
            await self._acquire_load_lock(job)
            try:
                pe_graph = build_refine_prompt_graph(
                    job.prompt,
                    mode=job.mode,
                    image_names=tuple(uploaded_names),
                    seed=job.seed,
                    thinking=job.thinking,
                    system_prompt=job.system_prompt,
                )
                self._update_job(job, stage="refining_prompt", percent=5)
                pe_outputs = await self._submit_and_track_prompt(
                    worker,
                    job,
                    pe_graph,
                    stage_label="refining_prompt",
                    emit_previews=False,
                )
                enhanced = self._extract_preview_text(pe_outputs, "3") or job.prompt
                reasoning = self._extract_preview_text(pe_outputs, "4")
                self._update_job(
                    job,
                    stage="freeing_vram",
                    enhanced_prompt=enhanced,
                    thinking_text=reasoning or None,
                    percent=100 if job.action == "refine" else 15,
                )
                await self._post_free(worker)
            finally:
                if self._load_lock.locked():
                    self._load_lock.release()
            job.stage1_s = round(max(0.0, time.monotonic() - t_s1), 2)

        if job.action == "refine":
            return

        # Stage 2: Diffusion image generation / editing (parallel across GPU 0 & GPU 1 after initial load)
        job.warm_worker = worker.last_loaded_profile == job.profile
        effective_prompt = (job.enhanced_prompt or job.prompt).strip()
        installed = inspect_installed_models(self.config.comfy_dir)
        base_unet_key = "unet_turbo" if job.profile == "fast" else "unet_base"
        bf16_unet_key = f"{base_unet_key}_bf16"
        if installed.get(bf16_unet_key, {}).get("installed") and (
            self.config.accelerator == "tpu" or not installed.get(base_unet_key, {}).get("installed")
        ):
            chosen_unet_key: str | None = bf16_unet_key
        else:
            chosen_unet_key = base_unet_key

        if installed.get("clip_vl_bf16", {}).get("installed") and (
            self.config.accelerator == "tpu" or not installed.get("clip_vl", {}).get("installed")
        ):
            chosen_clip_key: str | None = "clip_vl_bf16"
        else:
            chosen_clip_key = "clip_vl"

        effective_cache_dev = "off" if self.config.accelerator == "tpu" else job.cache_device
        req = GenerationRequest(
            prompt=effective_prompt,
            mode=job.mode,
            profile=job.profile,
            negative_prompt=job.negative_prompt,
            width=job.width,
            height=job.height,
            seed=job.seed,
            steps=job.steps,
            cfg=job.cfg,
            sampler_name=job.sampler_name,
            scheduler=job.scheduler,
            resolution=job.resolution,
            image_names=tuple(uploaded_names),
            use_image_latent=job.use_image_latent,
            identity_blend_mode=job.identity_blend_mode,
            identity_blend_ratio=job.identity_blend_ratio,
            cache_device=effective_cache_dev,
            cache_dtype=job.cache_dtype,
            filename_prefix=f"kaggle_gpu{worker.gpu_index}_{job.id[:8]}",
            unet_key_override=chosen_unet_key,
            clip_key_override=chosen_clip_key,
        )
        prompt_graph = build_qwen21_api_prompt(req)

        phase_timings: dict[str, Any] = {}
        # Acquire _load_lock for initial disk read -> VRAM staging, released automatically once sampling starts.
        await self._acquire_load_lock(job)
        stage2_outputs = await self._submit_and_track_prompt(
            worker,
            job,
            prompt_graph,
            stage_label="generating",
            emit_previews=True,
            release_lock_on_sampling=self._load_lock,
            phase_timings=phase_timings,
        )

        image_meta = self._extract_first_saved_image(stage2_outputs)
        if image_meta is None:
            raise RuntimeError(f"ComfyUI finished without SaveImage output: {stage2_outputs}")

        view_qs = urllib.parse.urlencode(
            {
                "filename": image_meta["filename"],
                "subfolder": image_meta.get("subfolder", ""),
                "type": image_meta.get("type", "output"),
            }
        )
        async with self._http_client.get(f"{worker.base_url}/view?{view_qs}") as img_resp:
            if img_resp.status >= 400:
                raise RuntimeError(f"Failed fetching output image ({img_resp.status})")
            png_bytes = await img_resp.read()

        t_stage2_end = time.monotonic()
        t_sent = phase_timings.get("prompt_sent_at")
        t_vae = phase_timings.get("vae_enter_at")
        t_first = phase_timings.get("first_step_done_at")
        t_sampler_enter = phase_timings.get("sampler_enter_at")
        first_val = int(phase_timings.get("first_step_val") or 0)
        last_max = int(phase_timings.get("last_step_max") or job.max_steps or job.effective_steps or 0)

        if isinstance(t_sent, float):
            if isinstance(t_vae, float):
                job.vae_s = round(max(0.0, t_stage2_end - t_vae), 2)
                sampling_end = t_vae
            else:
                sampling_end = float(phase_timings.get("prompt_done_at") or t_stage2_end)

            total_pre_vae = max(0.0, sampling_end - t_sent)
            if isinstance(t_first, float) and last_max > first_val >= 1 and sampling_end >= t_first:
                remaining_steps = last_max - first_val
                per_step = (sampling_end - t_first) / max(1, remaining_steps)
                sampling_total = min(total_pre_vae, max(0.0, per_step * last_max))
                load_total = max(0.0, total_pre_vae - sampling_total)
            elif isinstance(t_sampler_enter, float) and sampling_end >= t_sampler_enter >= t_sent:
                load_total = t_sampler_enter - t_sent
                sampling_total = sampling_end - t_sampler_enter
            else:
                load_total = 0.0
                sampling_total = total_pre_vae

            job.load_s = round(load_total, 2)
            job.sampling_s = round(sampling_total, 2)

        dims = parse_png_dimensions(png_bytes)
        out_w, out_h = dims if dims is not None else (job.width, job.height)
        job.output_width = out_w
        job.output_height = out_h
        worker.last_loaded_profile = job.profile

        steps_for_rate = job.max_steps or job.effective_steps
        if steps_for_rate > 0 and job.sampling_s is not None and out_w > 0 and out_h > 0:
            job.sec_per_step = round(job.sampling_s / steps_for_rate, 3)
            mp = (out_w * out_h) / 1_000_000.0
            if mp > 0:
                job.ms_per_step_mp = round((job.sampling_s * 1000.0) / (steps_for_rate * mp), 2)

        job.image_bytes = png_bytes
        job.output_filename = str(image_meta["filename"])
        self._image_order.append(job.id)
        self._prune_retention()

    @staticmethod
    def _extract_preview_text(outputs: dict[str, Any], node_id: str) -> str:
        node_out = outputs.get(node_id)
        if not isinstance(node_out, dict):
            return ""
        text_val = node_out.get("text")
        if isinstance(text_val, list) and text_val:
            return str(text_val[0]).strip()
        if isinstance(text_val, str):
            return text_val.strip()
        return ""

    @staticmethod
    def _extract_first_saved_image(outputs: dict[str, Any]) -> dict[str, Any] | None:
        preferred = outputs.get("13")
        if isinstance(preferred, dict) and preferred.get("images"):
            return dict(preferred["images"][0])
        for node_out in outputs.values():
            if isinstance(node_out, dict) and node_out.get("images"):
                return dict(node_out["images"][0])
        return None

    async def enqueue_job(self, body: dict[str, Any]) -> GatewayJob:
        action = str(body.get("action") or "generate").strip().lower()
        if action not in ("generate", "refine"):
            raise ValueError("action must be 'generate' or 'refine'.")

        mode = str(body.get("mode") or "t2i").strip().lower()
        if mode not in ("t2i", "edit"):
            raise ValueError("mode must be 't2i' or 'edit'.")

        profile = str(body.get("profile") or "fast").strip().lower()
        if profile not in ("fast", "quality"):
            raise ValueError("profile must be 'fast' or 'quality'.")

        prompt = str(body.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("prompt cannot be empty.")

        negative_prompt = str(body.get("negative_prompt") or "").strip()
        width = int(body.get("width") or 1024)
        height = int(body.get("height") or 1024)
        if not (256 <= width <= 2048 and 256 <= height <= 2048):
            raise ValueError("width and height must be between 256 and 2048.")

        raw_steps = body.get("steps")
        steps = int(raw_steps) if raw_steps is not None else (25 if profile == "quality" else None)
        if steps is not None and not (1 <= steps <= 60):
            raise ValueError("steps must be between 1 and 60.")

        raw_cfg = body.get("cfg")
        cfg = float(raw_cfg) if raw_cfg is not None else 1.0
        if not (0.5 <= cfg <= 15.0):
            raise ValueError("cfg must be between 0.5 and 15.0.")

        raw_seed = body.get("seed")
        seed = int(raw_seed) if raw_seed is not None and int(raw_seed) >= 0 else random.randint(1, 2_147_483_647)

        enhance_prompt = bool(body.get("enhance_prompt", False))
        thinking = bool(body.get("thinking", False))
        system_prompt = str(body.get("system_prompt") or "").strip()
        use_image_latent = bool(body.get("use_image_latent", True))
        identity_blend_mode = str(body.get("identity_blend_mode") or "none").strip().lower()
        if identity_blend_mode not in IDENTITY_BLEND_MODES:
            raise ValueError(f"identity_blend_mode must be one of {IDENTITY_BLEND_MODES}.")
        raw_blend_ratio = body.get("identity_blend_ratio")
        if raw_blend_ratio is None:
            identity_blend_ratio = 0.5
        else:
            if isinstance(raw_blend_ratio, bool) or not isinstance(raw_blend_ratio, (int, float)):
                raise ValueError("identity_blend_ratio must be a number between 0.1 and 0.9.")
            identity_blend_ratio = float(raw_blend_ratio)
            if not math.isfinite(identity_blend_ratio) or not (0.1 <= identity_blend_ratio <= 0.9):
                raise ValueError("identity_blend_ratio must be between 0.1 and 0.9.")
        normalize_face_crops = bool(body.get("normalize_face_crops", False))
        cache_device = str(body.get("cache_device") or "cpu").strip()
        cache_dtype = str(body.get("cache_dtype") or "int8").strip()
        sampler_name = str(body.get("sampler_name") or "euler").strip()
        scheduler = str(body.get("scheduler") or "simple").strip()
        resolution = int(body.get("resolution") or 1024)

        raw_pref = body.get("preferred_gpu")
        preferred_gpu = int(raw_pref) if raw_pref is not None else None
        if preferred_gpu is not None and not (0 <= preferred_gpu < len(self.workers)):
            raise ValueError(f"preferred_gpu={preferred_gpu} is out of range.")

        raw_images = body.get("images") or []
        if not isinstance(raw_images, list):
            raise ValueError("images must be a list.")
        if len(raw_images) > MAX_REFERENCE_IMAGES:
            raise ValueError(f"Maximum {MAX_REFERENCE_IMAGES} images allowed.")
        decoded_images = [_decode_uploaded_image(item, idx) for idx, item in enumerate(raw_images)]

        if mode == "edit" and not decoded_images:
            raise ValueError("Edit mode requires at least one reference image.")

        installed = inspect_installed_models(self.config.comfy_dir)
        needed = required_model_keys(
            action=action,
            mode=mode,
            profile=profile,
            enhance_prompt=enhance_prompt,
        )
        missing: list[str] = []
        for k in needed:
            ok = bool(installed.get(k, {}).get("installed"))
            bf16_alt = f"{k}_bf16"
            if not ok and bf16_alt in installed and installed[bf16_alt]["installed"]:
                ok = True
            if not ok:
                missing.append(k)
        if missing:
            raise MissingModelsError(missing)

        effective_steps = 0 if action == "refine" else (8 if profile == "fast" else (steps or 25))
        effective_cfg = float(cfg)

        job = GatewayJob(
            id=uuid.uuid4().hex[:16],
            action=action,
            mode=mode,
            profile=profile,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            seed=seed,
            sampler_name=sampler_name,
            scheduler=scheduler,
            resolution=resolution,
            enhance_prompt=enhance_prompt,
            thinking=thinking,
            system_prompt=system_prompt,
            use_image_latent=use_image_latent,
            identity_blend_mode=identity_blend_mode,
            identity_blend_ratio=identity_blend_ratio,
            normalize_face_crops=normalize_face_crops,
            cache_device=cache_device,
            cache_dtype=cache_dtype,
            preferred_gpu=preferred_gpu,
            accelerator=("tpu_v5e_8" if self.config.accelerator == "tpu" else "gpu_t4_x2"),
            images=decoded_images,
            effective_steps=effective_steps,
            effective_cfg=effective_cfg,
        )
        self._jobs[job.id] = job
        self._job_order.append(job.id)
        async with self._dispatch_cond:
            self._pending_jobs.append(job.id)
            self._dispatch_cond.notify_all()
        return job

    def build_status_payload(self) -> dict[str, Any]:
        diag = collect_system_diagnostics()
        models = inspect_installed_models(self.config.comfy_dir)
        # Allow BF16 weights to light up the standard UI badges (`unet_turbo`, `unet_base`, `clip_vl`)
        for base_k in ("unet_turbo", "unet_base", "clip_vl"):
            bf16_k = f"{base_k}_bf16"
            if base_k in models and bf16_k in models and not models[base_k]["installed"] and models[bf16_k]["installed"]:
                models[base_k] = {**models[base_k], "installed": True, "size_gib": models[bf16_k]["size_gib"]}
        return {
            "ok": True,
            "service": "qwen-kaggle-gateway",
            "accelerator": "tpu_v5e_8" if self.config.accelerator == "tpu" else "gpu_t4_x2",
            "uptime_s": round((time.time() * 1000 - self.started_at_ms) / 1000, 1),
            "queue_depth": len(self._pending_jobs),
            "workers": [w.to_public_dict() for w in self.workers],
            "models": models,
            "model_sync": dict(self._model_sync_state),
            "idle": self.build_idle_payload(),
            "capabilities": {
                "identity_blend_modes": list(IDENTITY_BLEND_MODES),
                "face_scale_normalization": True,
                "idle_extend": True,
            },
            "system": {
                "ram_total_gb": diag["memory"].get("ram_total_gb"),
                "ram_available_gb": diag["memory"].get("ram_available_gb"),
                "tmp_free_gb": (diag["disks"].get("tmp") or {}).get("free_gb"),
            },
        }


class MissingModelsError(RuntimeError):
    """Raised when a job asks for weights not yet present in ComfyUI models directory."""

    def __init__(self, missing_keys: list[str]) -> None:
        self.missing_keys = missing_keys
        names = ", ".join(MODELS[k].filename for k in missing_keys)
        super().__init__(f"Missing required model weights on Kaggle: {names}")


def create_gateway_app(gateway: QwenGateway) -> web.Application:
    """Create the authenticated aiohttp Web Application for `QwenGateway`."""

    @web.middleware
    async def auth_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
        if request.path == "/health":
            return await handler(request)
        if request.path.startswith("/v1/"):
            auth_header = request.headers.get("Authorization", "")
            prefix = "Bearer "
            provided = auth_header[len(prefix) :].strip() if auth_header.startswith(prefix) else ""
            if not provided or not hmac.compare_digest(provided, gateway.config.bearer_token):
                return web.json_response(
                    {"error": "UNAUTHORIZED", "message": "Missing or invalid Bearer token."},
                    status=401,
                    headers={"Cache-Control": "no-store"},
                )
        return await handler(request)

    app = web.Application(
        middlewares=[auth_middleware],
        client_max_size=128 * 1024 * 1024,
    )

    async def handle_health(_request: web.Request) -> web.Response:
        return web.json_response(
            {"ok": True, "service": "qwen-kaggle-gateway", "auth_required": True},
            headers={"Cache-Control": "no-store"},
        )

    async def handle_status(_request: web.Request) -> web.Response:
        await gateway._refresh_vram_once()
        return web.json_response(gateway.build_status_payload(), headers={"Cache-Control": "no-store"})

    async def handle_ensure_models(request: web.Request) -> web.Response:
        body = await request.json() if request.can_read_body else {}
        preset = str((body or {}).get("preset") or gateway.config.model_preset).strip()
        if preset not in PRESETS:
            return web.json_response({"error": "INVALID_PRESET", "presets": list(PRESETS)}, status=400)
        if gateway._model_sync_state["running"]:
            return web.json_response({"ok": True, "already_running": True, "model_sync": gateway._model_sync_state})

        async def _sync_worker() -> None:
            gateway._model_sync_state.update({"running": True, "preset": preset, "last_error": None})
            try:
                await asyncio.to_thread(
                    ensure_models_and_templates,
                    preset,
                    comfy_dir=gateway.config.comfy_dir,
                    venv_dir=gateway.config.venv_dir,
                    input_root=gateway.config.input_dir,
                )
                gateway._model_sync_state["completed_at_ms"] = int(time.time() * 1000)
            except Exception as exc:
                gateway._model_sync_state["last_error"] = str(exc)
            finally:
                gateway._model_sync_state["running"] = False

        gateway._model_sync_task = asyncio.create_task(_sync_worker())
        return web.json_response(
            {"ok": True, "model_sync": gateway._model_sync_state},
            status=202,
            headers={"Cache-Control": "no-store"},
        )

    async def handle_free(request: web.Request) -> web.Response:
        body = await request.json() if request.can_read_body else {}
        force = bool((body or {}).get("force", False))
        freed: list[int] = []
        skipped: list[int] = []
        for w in gateway.workers:
            if w.state == "busy":
                if not force:
                    skipped.append(w.gpu_index)
                    continue
                if w.current_job_id:
                    active_job = gateway._jobs.get(w.current_job_id)
                    if active_job is not None:
                        active_job.cancel_requested = True
                await gateway._interrupt_worker(w)
            await gateway._post_free(w)
            freed.append(w.gpu_index)
        return web.json_response({"ok": True, "freed_gpus": freed, "skipped_busy_gpus": skipped})

    async def handle_list_jobs(_request: web.Request) -> web.Response:
        jobs = [gateway._jobs[jid].to_public_dict() for jid in reversed(gateway._job_order) if jid in gateway._jobs]
        return web.json_response({"jobs": jobs}, headers={"Cache-Control": "no-store"})

    async def handle_create_job(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "INVALID_JSON"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"error": "INVALID_BODY"}, status=400)
        try:
            job = await gateway.enqueue_job(body)
        except MissingModelsError as exc:
            return web.json_response(
                {"error": "MISSING_MODELS", "message": str(exc), "missing_models": exc.missing_keys},
                status=400,
            )
        except ValueError as exc:
            return web.json_response({"error": "VALIDATION_ERROR", "message": str(exc)}, status=400)

        return web.json_response(
            {"job": job.to_public_dict()},
            status=202,
            headers={"Cache-Control": "no-store"},
        )

    async def handle_get_job(request: web.Request) -> web.Response:
        job_id = request.match_info["id"]
        job = gateway._jobs.get(job_id)
        if job is None:
            return web.json_response({"error": "NOT_FOUND"}, status=404)
        return web.json_response({"job": job.to_public_dict()}, headers={"Cache-Control": "no-store"})

    async def handle_job_image(request: web.Request) -> web.Response:
        job_id = request.match_info["id"]
        job = gateway._jobs.get(job_id)
        if job is None or job.image_bytes is None:
            return web.json_response({"error": "IMAGE_NOT_FOUND"}, status=404)
        fname = re.sub(r"[^a-zA-Z0-9._-]+", "_", job.output_filename or f"{job_id}.png")
        return web.Response(
            body=job.image_bytes,
            content_type="image/png",
            headers={
                "Content-Disposition": f'inline; filename="{fname}"',
                "Cache-Control": "no-store",
            },
        )

    async def handle_cancel_job(request: web.Request) -> web.Response:
        job_id = request.match_info["id"]
        job = gateway._jobs.get(job_id)
        if job is None:
            return web.json_response({"error": "NOT_FOUND"}, status=404)
        job.cancel_requested = True
        if job.status == "queued":
            async with gateway._dispatch_cond:
                try:
                    gateway._pending_jobs.remove(job_id)
                except ValueError:
                    pass
                gateway._dispatch_cond.notify_all()
            gateway._update_job(job, status="cancelled", stage="cancelled", error="Cancelled in queue")
        elif job.status == "running" and job.gpu_index is not None:
            for w in gateway.workers:
                if w.gpu_index == job.gpu_index:
                    await gateway._interrupt_worker(w)
                    break
        return web.json_response({"job": job.to_public_dict()})

    async def handle_job_events(request: web.Request) -> web.StreamResponse:
        job_id = request.match_info["id"]
        job = gateway._jobs.get(job_id)
        if job is None:
            return web.json_response({"error": "NOT_FOUND"}, status=404)

        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-store",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        await resp.prepare(request)

        sub_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=64)
        subs = gateway._subscribers.setdefault(job_id, [])
        subs.append(sub_queue)
        try:
            initial = json.dumps({"job": job.to_public_dict()})
            await resp.write(f"event: update\ndata: {initial}\n\n".encode("utf-8"))
            if job.status in ("completed", "failed", "cancelled"):
                return resp

            while True:
                try:
                    snapshot = await asyncio.wait_for(sub_queue.get(), timeout=10.0)
                    payload = json.dumps({"job": snapshot})
                    await resp.write(f"event: update\ndata: {payload}\n\n".encode("utf-8"))
                    if snapshot.get("status") in ("completed", "failed", "cancelled"):
                        break
                except asyncio.TimeoutError:
                    await resp.write(b": keepalive\n\n")
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            if sub_queue in subs:
                subs.remove(sub_queue)
            if not subs:
                gateway._subscribers.pop(job_id, None)
        return resp

    app.router.add_get("/health", handle_health)
    app.router.add_get("/v1/status", handle_status)
    app.router.add_post("/v1/models/ensure", handle_ensure_models)
    app.router.add_post("/v1/free", handle_free)
    app.router.add_get("/v1/jobs", handle_list_jobs)
    app.router.add_post("/v1/jobs", handle_create_job)
    app.router.add_get("/v1/jobs/{id}", handle_get_job)
    app.router.add_get("/v1/jobs/{id}/events", handle_job_events)
    app.router.add_get("/v1/jobs/{id}/image", handle_job_image)
    app.router.add_post("/v1/jobs/{id}/cancel", handle_cancel_job)

    async def handle_shutdown(request: web.Request) -> web.Response:
        """Stop button: cancel queued/running jobs, then stop the gateway, tunnel and ComfyUI workers."""
        reason = "user_stop"
        if request.can_read_body:
            try:
                body = await request.json()
                if isinstance(body, dict) and str(body.get("reason") or "").strip():
                    reason = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(body["reason"]).strip())[:40]
            except Exception:
                pass
        for job in gateway._jobs.values():
            if job.status in ("queued", "running"):
                job.cancel_requested = True
        accepted = gateway.request_shutdown(reason)
        return web.json_response(
            {"ok": True, "shutting_down": True, "already_requested": not accepted, "reason": gateway.shutdown_reason},
            status=202,
            headers={"Cache-Control": "no-store"},
        )

    async def handle_idle_extend(request: web.Request) -> web.Response:
        """Extend the idle auto-shutdown countdown by `seconds` (default 600)."""
        body: dict[str, Any] = {}
        if request.can_read_body:
            raw_text = (await request.text()).strip()
            if raw_text:
                try:
                    parsed = json.loads(raw_text)
                except Exception:
                    return web.json_response({"error": "INVALID_JSON"}, status=400)
                if not isinstance(parsed, dict):
                    return web.json_response({"error": "INVALID_JSON"}, status=400)
                body = parsed

        raw_seconds = body.get("seconds", 600.0)
        if (
            isinstance(raw_seconds, bool)
            or not isinstance(raw_seconds, (int, float))
            or not math.isfinite(float(raw_seconds))
            or float(raw_seconds) <= 0
            or float(raw_seconds) > 7200
        ):
            return web.json_response(
                {"error": "INVALID_SECONDS", "message": "seconds must be between 0 and 7200."},
                status=400,
            )

        seconds = float(raw_seconds)
        accepted = gateway.extend_idle(seconds)
        if not accepted:
            return web.json_response(
                {"ok": False, "error": "SHUTTING_DOWN", "idle": gateway.build_idle_payload()},
                status=409,
                headers={"Cache-Control": "no-store"},
            )
        return web.json_response(
            {"ok": True, "extended_s": round(seconds, 1), "idle": gateway.build_idle_payload()},
            status=200,
            headers={"Cache-Control": "no-store"},
        )

    app.router.add_post("/v1/idle/extend", handle_idle_extend)
    app.router.add_post("/v1/shutdown", handle_shutdown)

    async def on_startup(_app: web.Application) -> None:
        await gateway.start()

    async def on_cleanup(_app: web.Application) -> None:
        await gateway.stop()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def _resolve_bearer_token(explicit: str = "") -> str:
    if explicit.strip():
        return explicit.strip()
    env_tok = os.environ.get("QWEN_GATEWAY_TOKEN", "").strip()
    if env_tok:
        return env_tok
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        secret = str(UserSecretsClient().get_secret("QWEN_GATEWAY_TOKEN") or "").strip()
        if secret:
            return secret
    except Exception:
        pass
    return secrets.token_urlsafe(24)


async def run_gateway_server(
    config: GatewayConfig,
    *,
    use_tunnel: bool = True,
) -> None:
    """Start `QwenGateway`, aiohttp server, and optional `cloudflared` tunnel until interrupted."""
    gateway = QwenGateway(config)
    app = create_gateway_app(gateway)
    tunnel: CloudflaredTunnel | None = (
        CloudflaredTunnel(config.port, input_root=config.input_dir) if use_tunnel else None
    )
    tunnel_task: asyncio.Task[str] | None = (
        asyncio.create_task(tunnel.start()) if tunnel is not None else None
    )
    runner = web.AppRunner(app, access_log=None)
    runner_ready = False
    try:
        await runner.setup()
        runner_ready = True
        site = web.TCPSite(runner, config.host, config.port)
        await site.start()
        print(f"[GATEWAY] Listening on http://{config.host}:{config.port} ({len(gateway.workers)} GPU workers ready)", flush=True)
        if tunnel_task is not None:
            public_url = await tunnel_task
            pair_payload = json.dumps(
                {
                    "url": public_url,
                    "token": config.bearer_token,
                    "accelerator": "tpu_v5e_8" if config.accelerator == "tpu" else "gpu_t4_x2",
                }
            )
            banner = "\n".join(
                [
                    "",
                    "=" * 72,
                    "KAGGLE QWEN 3.5 GATEWAY GOTOWY DO PRACY Z BARTOSZ OS",
                    f"TUNNEL_URL   : {public_url}",
                    f"BEARER_TOKEN : {config.bearer_token}",
                    f"BARTOSZ_OS_JSON: {pair_payload}",
                    "=" * 72,
                    "",
                ]
            )
            print(banner, flush=True)
        else:
            pair_payload = json.dumps(
                {
                    "url": f"http://{config.host}:{config.port}",
                    "token": config.bearer_token,
                    "accelerator": "tpu_v5e_8" if config.accelerator == "tpu" else "gpu_t4_x2",
                }
            )
            print(f"BARTOSZ_OS_JSON: {pair_payload}", flush=True)

        gateway.touch_activity()
        watchdog = asyncio.create_task(gateway.idle_watchdog())
        if config.idle_timeout_s > 0:
            print(
                f"[GATEWAY] Auto-shutdown after {config.idle_timeout_s:.0f}s idle; "
                "extend anytime via POST /v1/idle/extend or stop via POST /v1/shutdown",
                flush=True,
            )
        try:
            await gateway.shutdown_event.wait()
        finally:
            watchdog.cancel()
        print(f"[GATEWAY] Shutting down ({gateway.shutdown_reason}).", flush=True)
        await asyncio.sleep(1.0)  # let the /v1/shutdown HTTP response flush before the tunnel closes
    finally:
        if tunnel_task is not None and not tunnel_task.done():
            tunnel_task.cancel()
            try:
                await tunnel_task
            except (asyncio.CancelledError, Exception):
                pass
        if tunnel is not None:
            await tunnel.stop()
        if runner_ready:
            await runner.cleanup()


def prepare_and_launch_gateway_subprocess(
    *,
    comfy_dir: Path = DEFAULT_COMFY_DIR,
    venv_dir: Path = DEFAULT_VENV_DIR,
    input_dir: Path = DEFAULT_INPUT_DIR,
    preset: str = "full_i2i",
    port: int = 8000,
    num_workers: int = 2,
    force_hf: bool = False,
    bearer_token: str = "",
    use_tunnel: bool = True,
    accelerator: str = "gpu",
    idle_timeout_s: float = 600.0,
    prompt_timeout_s: float = 1800.0,
) -> int:
    """Prepare `/tmp/ComfyUI` + `/tmp/venv` + models, then stream the isolated `/tmp/venv` gateway process."""
    diag = collect_system_diagnostics(input_dir=input_dir)
    print("[DIAG]", json.dumps(diag, indent=2), flush=True)
    resolved_accel = (
        ("tpu" if diag.get("tpu_devices") else "gpu")
        if accelerator == "auto"
        else accelerator
    )
    py_exe = setup_comfy_and_venv(
        comfy_dir,
        venv_dir,
        accelerator=resolved_accel,
        input_root=input_dir,
    )
    ensure_models_and_templates(
        preset,
        comfy_dir=comfy_dir,
        venv_dir=venv_dir,
        input_root=input_dir,
        force_hf=force_hf,
    )
    if resolved_accel == "tpu":
        effective_workers = 1
    else:
        gpu_count = len(diag.get("nvidia_smi") or [])
        effective_workers = min(max(1, num_workers), gpu_count) if gpu_count else 1
    token = _resolve_bearer_token(bearer_token)

    pkg_root = str(Path(__file__).resolve().parent.parent)
    env = dict(os.environ)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{pkg_root}{os.pathsep}{existing_pp}" if existing_pp else pkg_root
    env["PYTHONUNBUFFERED"] = "1"
    env["QWEN_GATEWAY_TOKEN"] = token

    cmd = [
        str(py_exe),
        "-m",
        "qwen_kaggle.gateway",
        "--comfy-dir",
        str(comfy_dir),
        "--venv-dir",
        str(venv_dir),
        "--input-dir",
        str(input_dir),
        "--preset",
        preset,
        "--port",
        str(port),
        "--num-workers",
        str(effective_workers),
        "--accelerator",
        resolved_accel,
        "--idle-timeout",
        str(idle_timeout_s),
        "--prompt-timeout",
        str(prompt_timeout_s),
    ]
    if not use_tunnel:
        cmd.append("--no-tunnel")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
        return proc.wait()
    except KeyboardInterrupt:
        print("\n[GATEWAY] Stopping server...", flush=True)
        proc.terminate()
        try:
            return proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            return proc.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen-Image 2.1 Kaggle Gateway Server (Dual-T4 & TPU v5e-8)")
    parser.add_argument("--comfy-dir", type=Path, default=DEFAULT_COMFY_DIR)
    parser.add_argument("--venv-dir", type=Path, default=DEFAULT_VENV_DIR)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--preset", default="full_i2i", choices=tuple(PRESETS))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--accelerator", default="gpu", choices=("gpu", "tpu"))
    parser.add_argument("--no-tunnel", action="store_true")
    parser.add_argument("--idle-timeout", type=float, default=600.0, help="Seconds idle before auto-shutdown (0 disables).")
    parser.add_argument("--prompt-timeout", type=float, default=1800.0, help="Max seconds per ComfyUI prompt before interrupt (0 disables).")
    args = parser.parse_args()

    cfg = GatewayConfig(
        comfy_dir=args.comfy_dir,
        venv_dir=args.venv_dir,
        input_dir=args.input_dir,
        host=args.host,
        port=args.port,
        num_workers=1 if args.accelerator == "tpu" else args.num_workers,
        model_preset=args.preset,
        accelerator=args.accelerator,
        extra_comfy_flags=TPU_SAFE_COMFY_FLAGS if args.accelerator == "tpu" else SAFE_COMFY_FLAGS,
        bearer_token=_resolve_bearer_token(),
        prompt_timeout_s=args.prompt_timeout,
        idle_timeout_s=args.idle_timeout,
    )
    try:
        asyncio.run(run_gateway_server(cfg, use_tunnel=not args.no_tunnel))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
