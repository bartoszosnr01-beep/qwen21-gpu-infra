"""Authoritative constants, model catalog, and Kaggle runtime defaults for Qwen-Image 2.1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    """Metadata describing a single Qwen-Image 2.1 asset in Comfy-Org/Qwen-Image-2.1."""

    key: str
    hf_path: str
    subdir: str
    filename: str
    approx_gib: float
    description: str


HF_REPO_ID = "Comfy-Org/Qwen-Image-2.1"

MODELS: dict[str, ModelSpec] = {
    "unet_turbo": ModelSpec(
        key="unet_turbo",
        hf_path="diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",
        subdir="diffusion_models",
        filename="qwen_image_2.1_turbo_int8_convrot.safetensors",
        approx_gib=6.76,
        description="Qwen-Image 2.1 Turbo UNet (INT8 ConvRot, 8 steps, CFG 1.0)",
    ),
    "unet_base": ModelSpec(
        key="unet_base",
        hf_path="diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
        subdir="diffusion_models",
        filename="qwen_image_2.1_int8_convrot.safetensors",
        approx_gib=6.76,
        description="Qwen-Image 2.1 Base UNet (INT8 ConvRot, ~25-40 steps)",
    ),
    "clip_vl": ModelSpec(
        key="clip_vl",
        hf_path="text_encoders/qwen3vl_8b_int8_convrot.safetensors",
        subdir="text_encoders",
        filename="qwen3vl_8b_int8_convrot.safetensors",
        approx_gib=8.71,
        description="Qwen3-VL 8B text/vision encoder (INT8 ConvRot)",
    ),
    "pe_i2i": ModelSpec(
        key="pe_i2i",
        hf_path="text_encoders/qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors",
        subdir="text_encoders",
        filename="qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors",
        approx_gib=8.82,
        description="Qwen3.5 9B Prompt Enhancer for Image Edit (INT8 ConvRot)",
    ),
    "pe_t2i": ModelSpec(
        key="pe_t2i",
        hf_path="text_encoders/qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors",
        subdir="text_encoders",
        filename="qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors",
        approx_gib=8.82,
        description="Qwen3.5 9B Prompt Enhancer for Text-to-Image (INT8 ConvRot)",
    ),
    "vae": ModelSpec(
        key="vae",
        hf_path="vae/qwen_image_2.1_vae_bf16.safetensors",
        subdir="vae",
        filename="qwen_image_2.1_vae_bf16.safetensors",
        approx_gib=0.63,
        description="Qwen-Image 2.1 VAE (BF16)",
    ),
    "unet_turbo_bf16": ModelSpec(
        key="unet_turbo_bf16",
        hf_path="diffusion_models/qwen_image_2.1_turbo_bf16.safetensors",
        subdir="diffusion_models",
        filename="qwen_image_2.1_turbo_bf16.safetensors",
        approx_gib=13.25,
        description="Qwen-Image 2.1 Turbo UNet (Native BF16 for TPU v5e-8, 8 steps)",
    ),
    "unet_base_bf16": ModelSpec(
        key="unet_base_bf16",
        hf_path="diffusion_models/qwen_image_2.1_bf16.safetensors",
        subdir="diffusion_models",
        filename="qwen_image_2.1_bf16.safetensors",
        approx_gib=13.25,
        description="Qwen-Image 2.1 Base UNet (Native BF16 for TPU v5e-8, ~25-40 steps)",
    ),
    "clip_vl_bf16": ModelSpec(
        key="clip_vl_bf16",
        hf_path="text_encoders/qwen3vl_8b_bf16.safetensors",
        subdir="text_encoders",
        filename="qwen3vl_8b_bf16.safetensors",
        approx_gib=16.33,
        description="Qwen3-VL 8B text/vision encoder (Native BF16 for TPU v5e-8)",
    ),
}

FILENAME_TO_SPEC: dict[str, ModelSpec] = {spec.filename: spec for spec in MODELS.values()}

PRESETS: dict[str, tuple[str, ...]] = {
    # Fast Phase 0 smoke test (~16.1 GiB)
    "smoke": ("unet_turbo", "clip_vl", "vae"),
    # Core generation without 9B prompt enhancer (~22.9 GiB)
    "core": ("unet_turbo", "unet_base", "clip_vl", "vae"),
    # Full editing + generation with I2I 9B enhancer (~31.7 GiB)
    "full_i2i": ("unet_turbo", "unet_base", "clip_vl", "pe_i2i", "vae"),
    # Complete set including both T2I and I2I 9B enhancers (~40.5 GiB)
    "all": ("unet_turbo", "unet_base", "clip_vl", "pe_i2i", "pe_t2i", "vae"),
    # Native BF16 core set for TPU v5e-8 128 GiB HBM (~43.5 GiB)
    "tpu_bf16_core": ("unet_turbo_bf16", "unet_base_bf16", "clip_vl_bf16", "vae"),
    # Native BF16 full set + 9B I2I enhancer for TPU v5e-8 (~52.3 GiB)
    "tpu_bf16": ("unet_turbo_bf16", "unet_base_bf16", "clip_vl_bf16", "pe_i2i", "vae"),
}

TAESD_DECODER_URL = (
    "https://raw.githubusercontent.com/madebyollin/taesd/main/taeqi2_1_decoder.pth"
)
TAESD_DECODER_SUBDIR = "vae_approx"
TAESD_DECODER_FILENAME = "taeqi2_1_decoder.pth"

WORKFLOW_TEMPLATE_URLS: dict[str, str] = {
    "image_qwen_image_2_1_t2i.json": (
        "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/"
        "image_qwen_image_2_1_t2i.json"
    ),
    "image_qwen_image_2_1_turbo_t2i.json": (
        "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/"
        "image_qwen_image_2_1_turbo_t2i.json"
    ),
    "image_qwen_image_2_1_image_edit.json": (
        "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/"
        "image_qwen_image_2_1_image_edit.json"
    ),
    "image_qwen_image_2_1_turbo_image_edit.json": (
        "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/"
        "image_qwen_image_2_1_turbo_image_edit.json"
    ),
}

COMFYUI_REPO_URL = "https://github.com/Comfy-Org/ComfyUI.git"
CLOUDFLARED_LINUX_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
)
RUNTIME_CACHE_SLUG = "qwen-t4-runtime-cache"
RUNTIME_WHEELS_ARCHIVE = "qwen_t4_wheels.tar"
RUNTIME_COMFY_ARCHIVE = "comfyui_repo.tar"
RUNTIME_CLOUDFLARED_FILENAME = "cloudflared"
RUNTIME_META_FILENAME = "qwen_t4_runtime_meta.json"
GPU_EXTRA_PIP_PACKAGES: tuple[str, ...] = (
    "huggingface-hub[hf_xet]>=0.32.0",
    "comfy-kitchen",
    "aiohttp",
    "pillow",
    "numpy",
    "packaging",
    "requests",
    "wrapt",
)
DEFAULT_COMFY_DIR = Path("/tmp/ComfyUI")
DEFAULT_VENV_DIR = Path("/tmp/venv")
DEFAULT_HF_CACHE_DIR = Path("/tmp/hf_cache")
DEFAULT_WORKING_DIR = Path("/kaggle/working")
DEFAULT_INPUT_DIR = Path("/kaggle/input")

STANDARD_MODEL_SUBDIRS: tuple[str, ...] = (
    "checkpoints",
    "clip",
    "controlnet",
    "diffusion_models",
    "embeddings",
    "loras",
    "model_patches",
    "text_encoders",
    "unet",
    "upscale_models",
    "vae",
    "vae_approx",
)

# Verified from official Comfy-Org/workflow_templates Qwen-Image 2.1 Turbo templates
TURBO_MANUAL_SIGMAS = (
    "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0"
)

# Safe flags for Kaggle T4 (16 GB VRAM per card, ~31 GB system RAM across 2 ComfyUI workers, Turing sm_75).
# Deliberately omits --force-fp16 to avoid FP16 NaN / black-image overflows on Qwen/VAE.
# Explicit --cache-ram thresholds prevent 2 parallel ComfyUI workers from racing on default ~3.1 GB free-RAM eviction.
SAFE_COMFY_FLAGS: tuple[str, ...] = (
    "--disable-pinned-memory",
    "--disable-async-offload",
    "--cache-ram",
    "6.0",
    "8.0",
    "--preview-method",
    "taesd",
    "--disable-auto-launch",
)

TPU_SAFE_COMFY_FLAGS: tuple[str, ...] = (
    "--bf16-unet",
    "--bf16-vae",
    "--disable-pinned-memory",
    "--disable-async-offload",
    "--preview-method",
    "taesd",
    "--disable-auto-launch",
)

TPU_CONTEXT_BUCKETS: tuple[int, ...] = (256, 512, 1024, 2048)
