"""Authoritative constants, model catalog, and Kaggle runtime defaults for MiniMax-H3-Image.

No built-in Prompt Enhancer (PE) weights are included: rich prompts are authored externally
(e.g. via Codex) and passed directly into `MiniMaxH3ImageConditioning`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    """Metadata describing a single MiniMax-H3-Image asset across Hugging Face repos."""

    key: str
    hf_repo_id: str
    hf_path: str
    subdir: str
    filename: str
    approx_gib: float
    loader_kind: str  # "safetensors" | "gguf"
    description: str


HF_COMFY_IMAGE_REPO = "Comfy-Org/MiniMax-H3-Image"
HF_COMFY_H3_REPO = "Comfy-Org/MiniMax-H3"
HF_GGUF_DIT_REPO = "Abiray/MiniMax-H3-Image-GGUF"
HF_GGUF_CLIP_REPO = "nif0/Qwen3-VL-32B-Instruct-MiniMax-H3-GGUF"

MODELS: dict[str, ModelSpec] = {
    # --- Native Comfy-Org safetensors (TPU v5e-8 128 GiB HBM + ~330 GiB host RAM) ---
    "dit_int8": ModelSpec(
        key="dit_int8",
        hf_repo_id=HF_COMFY_IMAGE_REPO,
        hf_path="diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors",
        subdir="diffusion_models",
        filename="minimax_h3_image_pruned_int8_convrot.safetensors",
        approx_gib=19.54,
        loader_kind="safetensors",
        description="MiniMax-H3-Image Pruned 33.1B DiT (INT8 ConvRot, dequantized to BF16 on TPU v5e-8)",
    ),
    "dit_bf16": ModelSpec(
        key="dit_bf16",
        hf_repo_id=HF_COMFY_IMAGE_REPO,
        hf_path="diffusion_models/minimax_h3_image_pruned_bf16.safetensors",
        subdir="diffusion_models",
        filename="minimax_h3_image_pruned_bf16.safetensors",
        approx_gib=37.47,
        loader_kind="safetensors",
        description="MiniMax-H3-Image Pruned 33.1B DiT (Native BF16 for TPU v5e-8)",
    ),
    "clip_int8": ModelSpec(
        key="clip_int8",
        hf_repo_id=HF_COMFY_IMAGE_REPO,
        hf_path="text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        subdir="text_encoders",
        filename="qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        approx_gib=25.28,
        loader_kind="safetensors",
        description="Qwen3-VL-32B (L0-49) MiniMax-H3 text/vision encoder (INT8 ConvRot)",
    ),
    "clip_bf16": ModelSpec(
        key="clip_bf16",
        hf_repo_id=HF_COMFY_H3_REPO,
        hf_path="text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors",
        subdir="text_encoders",
        filename="qwen3vl_32b_minimax_h3_bf16.safetensors",
        approx_gib=47.97,
        loader_kind="safetensors",
        description="Qwen3-VL-32B (L0-49) MiniMax-H3 text/vision encoder (Native BF16)",
    ),
    "vae": ModelSpec(
        key="vae",
        hf_repo_id=HF_COMFY_IMAGE_REPO,
        hf_path="vae/minimax_h3_image_vae_fp16.safetensors",
        subdir="vae",
        filename="minimax_h3_image_vae_fp16.safetensors",
        approx_gib=1.09,
        loader_kind="safetensors",
        description="MiniMax-H3-Image 24-ch 16x Image VAE (FP16)",
    ),
    # --- GGUF community quants (Kaggle GPU T4 x2 fallback: 16 GiB VRAM / ~29 GiB RAM) ---
    "dit_gguf_q3": ModelSpec(
        key="dit_gguf_q3",
        hf_repo_id=HF_GGUF_DIT_REPO,
        hf_path="MiniMax-H3-Image-Pruned-Q3_K_M.gguf",
        subdir="diffusion_models",
        filename="MiniMax-H3-Image-Pruned-Q3_K_M.gguf",
        approx_gib=8.30,
        loader_kind="gguf",
        description="MiniMax-H3-Image Pruned DiT GGUF Q3_K_M (FP32 sensitive layers preserved)",
    ),
    "dit_gguf_q4": ModelSpec(
        key="dit_gguf_q4",
        hf_repo_id=HF_GGUF_DIT_REPO,
        hf_path="MiniMax-H3-Image-Pruned-Q4_K_M.gguf",
        subdir="diffusion_models",
        filename="MiniMax-H3-Image-Pruned-Q4_K_M.gguf",
        approx_gib=10.78,
        loader_kind="gguf",
        description="MiniMax-H3-Image Pruned DiT GGUF Q4_K_M (FP32 sensitive layers preserved)",
    ),
    "clip_gguf_iq3": ModelSpec(
        key="clip_gguf_iq3",
        hf_repo_id=HF_GGUF_CLIP_REPO,
        hf_path="Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-UD-IQ3_XXS.gguf",
        subdir="text_encoders",
        filename="Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-UD-IQ3_XXS.gguf",
        approx_gib=9.10,
        loader_kind="gguf",
        description="Qwen3-VL-32B (L0-49) MiniMax-H3 text encoder GGUF UD-IQ3_XXS",
    ),
    "clip_gguf_iq4": ModelSpec(
        key="clip_gguf_iq4",
        hf_repo_id=HF_GGUF_CLIP_REPO,
        hf_path="Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-IQ4_XS.gguf",
        subdir="text_encoders",
        filename="Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-IQ4_XS.gguf",
        approx_gib=12.52,
        loader_kind="gguf",
        description="Qwen3-VL-32B (L0-49) MiniMax-H3 text encoder GGUF IQ4_XS",
    ),
    "clip_gguf_mmproj": ModelSpec(
        key="clip_gguf_mmproj",
        hf_repo_id=HF_GGUF_CLIP_REPO,
        hf_path="Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-mmproj-BF16.gguf",
        subdir="text_encoders",
        filename="Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-mmproj-BF16.gguf",
        approx_gib=1.12,
        loader_kind="gguf",
        description="Qwen3-VL-32B MiniMax-H3 multimodal vision projector (auto-loaded by Nif00/ComfyUI-GGUF)",
    ),
}

FILENAME_TO_SPEC: dict[str, ModelSpec] = {spec.filename: spec for spec in MODELS.values()}

PRESETS: dict[str, tuple[str, ...]] = {
    # Primary TPU v5e-8 set (~45.91 GiB): INT8 ConvRot DiT (dequantized to BF16 across 8 chips) + INT8 Qwen3-VL-32B + VAE
    "tpu_int8": ("dit_int8", "clip_int8", "vae"),
    # Native BF16 DiT for TPU v5e-8 (~63.84 GiB) paired with INT8 Qwen3-VL-32B on 330 GiB host RAM
    "tpu_bf16": ("dit_bf16", "clip_int8", "vae"),
    # T4 x2 GGUF balanced fallback (~22.09 GiB): Q4_K_M DiT + IQ3_XXS Qwen3-VL-32B + mmproj + VAE
    "t4_gguf": ("dit_gguf_q4", "clip_gguf_iq3", "clip_gguf_mmproj", "vae"),
    # T4 x2 GGUF lightest RAM footprint (~19.61 GiB): Q3_K_M DiT + IQ3_XXS Qwen3-VL-32B + mmproj + VAE
    "t4_gguf_lite": ("dit_gguf_q3", "clip_gguf_iq3", "clip_gguf_mmproj", "vae"),
    # T4 x2 GGUF higher-quality text encoder (~25.51 GiB; tight on 29 GiB host RAM)
    "t4_gguf_hq": ("dit_gguf_q4", "clip_gguf_iq4", "clip_gguf_mmproj", "vae"),
}

GGUF_PRESETS: frozenset[str] = frozenset({"t4_gguf", "t4_gguf_lite", "t4_gguf_hq"})

WORKFLOW_TEMPLATE_URLS: dict[str, str] = {
    "image_minimax_h3_image_t2i.json": (
        "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/"
        "image_minimax_h3_image_t2i.json"
    ),
    "image_minimax_h3_image_edit.json": (
        "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/"
        "image_minimax_h3_image_edit.json"
    ),
}

COMFYUI_REPO_URL = "https://github.com/Comfy-Org/ComfyUI.git"
# Fork of city96/ComfyUI-GGUF with explicit truncated Qwen3-VL-32B (L0-49) + mmproj support for MiniMax-H3.
GGUF_NODE_REPO_URL = "https://github.com/Nif00/ComfyUI-GGUF.git"
GGUF_NODE_DIRNAME = "ComfyUI-GGUF"

CLOUDFLARED_LINUX_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
)
RUNTIME_CACHE_SLUG = "minimax-h3-runtime-cache"
RUNTIME_WHEELS_ARCHIVE = "minimax_t4_wheels.tar"
RUNTIME_COMFY_ARCHIVE = "comfyui_repo.tar"
RUNTIME_GGUF_ARCHIVE = "comfyui_gguf_repo.tar"
RUNTIME_CLOUDFLARED_FILENAME = "cloudflared"
RUNTIME_META_FILENAME = "minimax_t4_runtime_meta.json"

GPU_EXTRA_PIP_PACKAGES: tuple[str, ...] = (
    "huggingface-hub[hf_xet]>=0.32.0",
    "comfy-kitchen",
    "gguf>=0.13.0",
    "sentencepiece",
    "protobuf",
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

# Official MiniMax-H3-Image sampling contract (CFG-distilled 6-step PDD checkpoint)
CANVAS_MULTIPLE = 32
MINIMAX_DEFAULT_STEPS = 6
MINIMAX_DEFAULT_CFG = 1.0
MINIMAX_DEFAULT_SAMPLER = "euler"
MINIMAX_DEFAULT_SCHEDULER = "simple"

# Mask inpainting & outpainting defaults
DEFAULT_INPAINT_MASK_EXPAND = 16
DEFAULT_INPAINT_MASK_BLUR = 12
DEFAULT_OUTPAINT_FEATHER = 80

# Safe resolution bounds:
# - T4 (16 GiB VRAM) is capped around 4 MP (2048x2048) to prevent OOM.
# - TPU v5e-8 (128 GiB HBM) defaults to 2K/4K (16 MP) and allows up to ~48 MP for multi-person group fidelity.
T4_DEFAULT_SIZE = 1024
T4_QUALITY_SIZE = 1536
T4_MAX_DIMENSION = 2048
TPU_DEFAULT_SIZE = 2048
TPU_QUALITY_SIZE = 4096
TPU_MAX_DIMENSION = 8192
TPU_MAX_PIXELS = 49_000_000  # up to ~48 MP (e.g. 6912x6912=47.8MP, 8192x5856=48.0MP, 4096x4096=16.8MP)

# Safe flags for Kaggle T4 (16 GiB VRAM, ~31 GiB system RAM, Turing sm_75).
# Relies on ComfyUI DynamicVRAM (--lowvram + comfy-aimdo) to stream GGUF layers without
# bloating host RAM (~3.8 GiB RSS observed live), while keeping ComfyUI's default node cache
# so 22 GiB of GGUF weights and token_embd do not reload from scratch on every prompt.
T4_SAFE_COMFY_FLAGS: tuple[str, ...] = (
    "--lowvram",
    "--disable-pinned-memory",
    "--disable-async-offload",
    "--preview-method",
    "latent2rgb",
    "--disable-auto-launch",
)

# Safe flags for Kaggle TPU v5e-8 (128 GiB HBM across 8 chips + ~330 GiB host RAM).
TPU_SAFE_COMFY_FLAGS: tuple[str, ...] = (
    "--bf16-unet",
    "--disable-pinned-memory",
    "--disable-async-offload",
    "--preview-method",
    "latent2rgb",
    "--disable-auto-launch",
)

TPU_CONTEXT_BUCKETS: tuple[int, ...] = (256, 512, 1024, 2048)
