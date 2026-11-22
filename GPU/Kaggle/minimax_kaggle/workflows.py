"""ComfyUI API prompt graph builders for MiniMax-H3-Image (T2I, Edit, Inpaint + Refs, Outpaint; zero PE).

Verified directly against:
- `Comfy-Org/ComfyUI` `comfy_extras/nodes_minimax_h3.py` (`MiniMaxH3ImageConditioning`)
- `Comfy-Org/ComfyUI` stock mask/outpaint nodes (`LoadImageMask`, `GrowMask`, `ImageBlur`,
  `MaskToImage`, `ImageToMask`, `VAEEncode`, `SetLatentNoiseMask`, `ImagePadForOutpaint`,
  `ImageCompositeMasked`, `ImageScale`, `GetImageSize`)
- `Comfy-Org/workflow_templates` (`image_minimax_h3_image_t2i.json`, `image_minimax_h3_image_edit.json`)
- `Nif00/ComfyUI-GGUF` (`UnetLoaderGGUF`, `CLIPLoaderGGUF` with `type="minimax"` + auto `mmproj`)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from minimax_kaggle.constants import (
    CANVAS_MULTIPLE,
    DEFAULT_INPAINT_MASK_BLUR,
    DEFAULT_INPAINT_MASK_EXPAND,
    DEFAULT_OUTPAINT_FEATHER,
    MINIMAX_DEFAULT_CFG,
    MINIMAX_DEFAULT_SAMPLER,
    MINIMAX_DEFAULT_SCHEDULER,
    MINIMAX_DEFAULT_STEPS,
    MODELS,
)

MAX_REFERENCE_IMAGES = 16
SUPPORTED_MODES = ("t2i", "edit", "inpaint", "outpaint")
MASK_CHANNELS = ("red", "alpha", "green", "blue")
IDENTITY_BLEND_MODES = ("none", "velocity_blend", "step_swap")

_QWEN_IMAGE_TAG_RE = re.compile(r"<image([1-9]|1[0-6])>", flags=re.IGNORECASE)
_MULTI_PIC_TAG_LIST_RE = re.compile(
    r"<Picture\s+\d+>(?:\s*(?:,|&|\+|and|or)\s*<Picture\s+\d+>)+",
    flags=re.IGNORECASE,
)
_SECONDARY_PIC_TAG_RE = re.compile(r"<Picture\s+(?:[2-9]|1[0-6])>", flags=re.IGNORECASE)
_EITHER_REF_RE = re.compile(
    r"\bfrom\s+(?:either|both)\s+(?:side\s+of\s+<Picture\s+1>|references?|reference\s+images?)\b",
    flags=re.IGNORECASE,
)
_BOTH_PARENTS_POSS_RE = re.compile(r"\bboth\s+(?:parents|mothers|fathers)'", flags=re.IGNORECASE)


def round_to_canvas_multiple(value: int, multiple: int = CANVAS_MULTIPLE) -> int:
    """Round a positive pixel dimension to the nearest non-zero multiple of `CANVAS_MULTIPLE` (32)."""
    if value <= 0:
        return 0
    return max(multiple, ((int(value) + multiple // 2) // multiple) * multiple)


def normalize_minimax_reference_tags(prompt: str) -> str:
    """Convert Qwen-style `<imageN>` tags to MiniMax-H3 `<Picture N>` tags while preserving native prompts."""
    return _QWEN_IMAGE_TAG_RE.sub(lambda m: f"<Picture {m.group(1)}>", prompt.strip())


def prepare_single_branch_prompt(prompt: str) -> str:
    """Normalize a multi-image fusion prompt for an isolated single-reference `<Picture 1>` branch."""
    cleaned = normalize_minimax_reference_tags(prompt)
    cleaned = _MULTI_PIC_TAG_LIST_RE.sub("<Picture 1>", cleaned)
    cleaned = _SECONDARY_PIC_TAG_RE.sub("<Picture 1>", cleaned)
    cleaned = _EITHER_REF_RE.sub("from <Picture 1>", cleaned)
    cleaned = _BOTH_PARENTS_POSS_RE.sub("<Picture 1>'s", cleaned)
    if "<picture 1>" not in cleaned.lower():
        cleaned = (
            f"Using the subject features of <Picture 1>: {cleaned}"
            if cleaned
            else "Portrait of the subject in <Picture 1>."
        )
    return cleaned


@dataclass(frozen=True)
class GenerationRequest:
    """Normalized generation parameters shared by smoke tests, TPU runs, and API gateway."""

    prompt: str
    mode: str = "t2i"  # "t2i" | "edit" | "inpaint" | "outpaint"
    profile: str = "fast"  # "fast" | "quality" (controls default resolution tier; model is 6-step distilled)
    width: int = 2048
    height: int = 2048
    seed: int = 424242
    steps: int = MINIMAX_DEFAULT_STEPS
    cfg: float = MINIMAX_DEFAULT_CFG
    sampler_name: str = MINIMAX_DEFAULT_SAMPLER
    scheduler: str = MINIMAX_DEFAULT_SCHEDULER
    image_names: tuple[str, ...] = field(default_factory=tuple)
    use_image_latent: bool = True  # when True in edit mode & width/height==0, MiniMax derives canvas from image_1
    mask_image_name: str | None = None  # white=regenerate, black=preserve (required for mode="inpaint")
    mask_channel: str = "red"  # "red" | "alpha" | "green" | "blue"
    mask_expand: int = DEFAULT_INPAINT_MASK_EXPAND
    mask_blur: int = DEFAULT_INPAINT_MASK_BLUR
    outpaint_left: int = 0
    outpaint_top: int = 0
    outpaint_right: int = 0
    outpaint_bottom: int = 0
    outpaint_feather: int = DEFAULT_OUTPAINT_FEATHER
    identity_blend_mode: str = "none"  # "none" | "velocity_blend" | "step_swap"
    identity_blend_ratio: float = 0.5
    filename_prefix: str = "minimax_h3"
    dit_key: str = "dit_int8"
    clip_key: str = "clip_int8"
    vae_key: str = "vae"


def _validate_request(req: GenerationRequest) -> None:
    if req.mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode={req.mode!r}; expected one of {SUPPORTED_MODES}.")
    if req.profile not in ("fast", "quality"):
        raise ValueError(f"Unsupported profile={req.profile!r}; expected 'fast' or 'quality'.")
    if not str(req.prompt or "").strip():
        raise ValueError("prompt cannot be empty.")
    if req.identity_blend_mode not in IDENTITY_BLEND_MODES:
        raise ValueError(
            f"Invalid identity_blend_mode={req.identity_blend_mode!r}; expected one of {IDENTITY_BLEND_MODES}."
        )
    if not (0.1 <= float(req.identity_blend_ratio) <= 0.9):
        raise ValueError("identity_blend_ratio must be between 0.1 and 0.9.")
    if req.mode in ("edit", "inpaint", "outpaint") and not req.image_names:
        raise ValueError(f"{req.mode} mode requires at least one image name in image_names.")
    if len(req.image_names) > MAX_REFERENCE_IMAGES:
        raise ValueError(f"Too many reference images ({len(req.image_names)} > {MAX_REFERENCE_IMAGES}).")
    if req.mode == "inpaint":
        if not (req.mask_image_name and str(req.mask_image_name).strip()):
            raise ValueError("Inpaint mode requires mask_image_name.")
        if req.mask_channel not in MASK_CHANNELS:
            raise ValueError(f"Invalid mask_channel={req.mask_channel!r}; expected one of {MASK_CHANNELS}.")
        if not (-256 <= int(req.mask_expand) <= 256):
            raise ValueError("mask_expand must be between -256 and 256.")
        if not (0 <= int(req.mask_blur) <= 31):
            raise ValueError("mask_blur must be between 0 and 31.")
    if req.mode == "outpaint":
        pads = (req.outpaint_left, req.outpaint_top, req.outpaint_right, req.outpaint_bottom)
        if any(int(p) < 0 for p in pads):
            raise ValueError("Outpaint padding values cannot be negative.")
        if not any(int(p) > 0 for p in pads):
            raise ValueError("Outpaint mode requires at least one positive padding side (left/top/right/bottom).")
        if not (0 <= int(req.outpaint_feather) <= 512):
            raise ValueError("outpaint_feather must be between 0 and 512.")
    if req.steps < 1 or req.steps > 30:
        raise ValueError(f"Invalid steps={req.steps}; expected 1..30 (MiniMax-H3-Image is distilled for 6 steps).")
    if not (0.5 <= float(req.cfg) <= 10.0):
        raise ValueError(f"Invalid cfg={req.cfg}; MiniMax-H3-Image expects cfg=1.0.")
    if req.width < 0 or req.height < 0:
        raise ValueError(f"Invalid negative dimensions: {req.width}x{req.height}.")
    if req.mode == "t2i" and (req.width <= 0 or req.height <= 0):
        raise ValueError(f"T2I mode requires positive dimensions: {req.width}x{req.height}.")
    for key in (req.dit_key, req.clip_key, req.vae_key):
        if key not in MODELS:
            raise ValueError(f"Unknown model key {key!r}; expected one of {tuple(MODELS)}.")


def build_minimax_h3_api_prompt(req: GenerationRequest) -> dict[str, dict[str, Any]]:
    """Build a ComfyUI API prompt graph for MiniMax-H3-Image (safetensors or GGUF, zero PE).

    Node ID conventions preserved for gateway progress/timing instrumentation:
      - "1": DiT loader (`UNETLoader` or `UnetLoaderGGUF`)
      - "3": Text encoder loader (`CLIPLoader` or `CLIPLoaderGGUF`, `type="minimax"`)
      - "4": `VAELoader`
      - "5": `MiniMaxH3ImageConditioning` (in standard non-blended mode)
      - "6": `ConditioningZeroOut`
      - "11": `KSampler` (6 steps, CFG 1.0, `euler` + `simple`)
      - "12": `VAEDecode`
      - "13": `SaveImage`
      - "14": `ImageCompositeMasked` (only for `inpaint` / `outpaint`, locking untouched pixels)
    """
    _validate_request(req)

    dit_spec = MODELS[req.dit_key]
    clip_spec = MODELS[req.clip_key]
    vae_spec = MODELS[req.vae_key]

    graph: dict[str, dict[str, Any]] = {}

    if dit_spec.loader_kind == "gguf":
        graph["1"] = {
            "class_type": "UnetLoaderGGUF",
            "inputs": {
                "unet_name": dit_spec.filename,
            },
        }
    else:
        graph["1"] = {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": dit_spec.filename,
                "weight_dtype": "default",
            },
        }

    if clip_spec.loader_kind == "gguf":
        graph["3"] = {
            "class_type": "CLIPLoaderGGUF",
            "inputs": {
                "clip_name": clip_spec.filename,
                "type": "minimax",
            },
        }
    else:
        graph["3"] = {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": clip_spec.filename,
                "type": "minimax",
                "device": "default",
            },
        }

    graph["4"] = {
        "class_type": "VAELoader",
        "inputs": {
            "vae_name": vae_spec.filename,
        },
    }

    target_w = round_to_canvas_multiple(req.width) if req.width > 0 else 0
    target_h = round_to_canvas_multiple(req.height) if req.height > 0 else 0
    if req.mode == "edit" and req.use_image_latent and req.width == 0 and req.height == 0:
        target_w, target_h = 0, 0

    composite_dest_ref: list[Any] | None = None
    composite_mask_ref: list[Any] | None = None

    use_diffusion_blend = (
        req.mode == "edit"
        and len(req.image_names) >= 2
        and req.identity_blend_mode in ("velocity_blend", "step_swap")
    )

    if req.mode == "inpaint":
        normalized_prompt = normalize_minimax_reference_tags(req.prompt)
        graph["20"] = {
            "class_type": "LoadImage",
            "inputs": {"image": req.image_names[0]},
        }
        base_img_ref: list[Any] = ["20", 0]
        if target_w > 0 and target_h > 0:
            graph["30"] = {
                "class_type": "ImageScale",
                "inputs": {
                    "image": ["20", 0],
                    "upscale_method": "lanczos",
                    "width": target_w,
                    "height": target_h,
                    "crop": "disabled",
                },
            }
            base_img_ref = ["30", 0]
            cond_w: Any = target_w
            cond_h: Any = target_h
        else:
            graph["31"] = {
                "class_type": "GetImageSize",
                "inputs": {"image": ["20", 0]},
            }
            cond_w = ["31", 0]
            cond_h = ["31", 1]

        graph["40"] = {
            "class_type": "LoadImageMask",
            "inputs": {
                "image": req.mask_image_name,
                "channel": req.mask_channel,
            },
        }
        mask_ref: list[Any] = ["40", 0]
        if int(req.mask_expand) != 0:
            graph["41"] = {
                "class_type": "GrowMask",
                "inputs": {
                    "mask": mask_ref,
                    "expand": int(req.mask_expand),
                    "tapered_corners": True,
                },
            }
            mask_ref = ["41", 0]
        if int(req.mask_blur) > 0:
            blur_radius = int(req.mask_blur)
            sigma = round(max(0.5, min(10.0, blur_radius / 2.0)), 2)
            graph["42"] = {
                "class_type": "MaskToImage",
                "inputs": {"mask": mask_ref},
            }
            graph["43"] = {
                "class_type": "ImageBlur",
                "inputs": {
                    "image": ["42", 0],
                    "blur_radius": blur_radius,
                    "sigma": sigma,
                },
            }
            graph["44"] = {
                "class_type": "ImageToMask",
                "inputs": {
                    "image": ["43", 0],
                    "channel": "red",
                },
            }
            mask_ref = ["44", 0]

        cond_inputs: dict[str, Any] = {
            "clip": ["3", 0],
            "vae": ["4", 0],
            "prompt": normalized_prompt,
            "width": cond_w,
            "height": cond_h,
            "images.image_1": base_img_ref,
        }
        next_id = 21
        for idx, extra_img in enumerate(req.image_names[1:], start=2):
            load_id = str(next_id)
            next_id += 1
            graph[load_id] = {
                "class_type": "LoadImage",
                "inputs": {"image": extra_img},
            }
            cond_inputs[f"images.image_{idx}"] = [load_id, 0]

        graph["5"] = {
            "class_type": "MiniMaxH3ImageConditioning",
            "inputs": cond_inputs,
        }
        graph["7"] = {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": base_img_ref,
                "vae": ["4", 0],
            },
        }
        graph["8"] = {
            "class_type": "SetLatentNoiseMask",
            "inputs": {
                "samples": ["7", 0],
                "mask": mask_ref,
            },
        }
        positive_ref: list[Any] = ["5", 0]
        latent_ref: list[Any] = ["8", 0]
        composite_dest_ref = base_img_ref
        composite_mask_ref = mask_ref

    elif req.mode == "outpaint":
        normalized_prompt = normalize_minimax_reference_tags(req.prompt)
        graph["20"] = {
            "class_type": "LoadImage",
            "inputs": {"image": req.image_names[0]},
        }
        src_img_ref: list[Any] = ["20", 0]
        if target_w > 0 and target_h > 0:
            graph["29"] = {
                "class_type": "ImageScale",
                "inputs": {
                    "image": ["20", 0],
                    "upscale_method": "lanczos",
                    "width": target_w,
                    "height": target_h,
                    "crop": "disabled",
                },
            }
            src_img_ref = ["29", 0]

        pad_l = round_to_canvas_multiple(req.outpaint_left) if req.outpaint_left > 0 else 0
        pad_t = round_to_canvas_multiple(req.outpaint_top) if req.outpaint_top > 0 else 0
        pad_r = round_to_canvas_multiple(req.outpaint_right) if req.outpaint_right > 0 else 0
        pad_b = round_to_canvas_multiple(req.outpaint_bottom) if req.outpaint_bottom > 0 else 0

        graph["30"] = {
            "class_type": "ImagePadForOutpaint",
            "inputs": {
                "image": src_img_ref,
                "left": pad_l,
                "top": pad_t,
                "right": pad_r,
                "bottom": pad_b,
                "feathering": int(req.outpaint_feather),
            },
        }
        graph["31"] = {
            "class_type": "GetImageSize",
            "inputs": {"image": ["30", 0]},
        }
        cond_inputs = {
            "clip": ["3", 0],
            "vae": ["4", 0],
            "prompt": normalized_prompt,
            "width": ["31", 0],
            "height": ["31", 1],
            "images.image_1": ["30", 0],
        }
        next_id = 21
        for idx, extra_img in enumerate(req.image_names[1:], start=2):
            load_id = str(next_id)
            next_id += 1
            graph[load_id] = {
                "class_type": "LoadImage",
                "inputs": {"image": extra_img},
            }
            cond_inputs[f"images.image_{idx}"] = [load_id, 0]

        graph["5"] = {
            "class_type": "MiniMaxH3ImageConditioning",
            "inputs": cond_inputs,
        }
        graph["7"] = {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["30", 0],
                "vae": ["4", 0],
            },
        }
        graph["8"] = {
            "class_type": "SetLatentNoiseMask",
            "inputs": {
                "samples": ["7", 0],
                "mask": ["30", 1],
            },
        }
        positive_ref = ["5", 0]
        latent_ref = ["8", 0]
        composite_dest_ref = ["30", 0]
        composite_mask_ref = ["30", 1]

    elif use_diffusion_blend:
        blend_w = target_w or 2048
        blend_h = target_h or 2048
        n_refs = len(req.image_names)
        branch_prompt = prepare_single_branch_prompt(req.prompt)
        ratio = round(float(req.identity_blend_ratio), 3)
        if n_refs == 2:
            weights = [round(1.0 - ratio, 3), ratio]
            ranges = [(0.0, round(1.0 - ratio, 3)), (round(1.0 - ratio, 3), 1.0)]
        else:
            weights = [round(1.0 / n_refs, 4) for _ in range(n_refs)]
            ranges = [
                (round(i / n_refs, 4), 1.0 if i == n_refs - 1 else round((i + 1) / n_refs, 4))
                for i in range(n_refs)
            ]

        next_blend_id = 50
        pos_branches: list[list[Any]] = []
        blend_latent_ref: list[Any] | None = None

        for idx, img_name in enumerate(req.image_names):
            load_id = str(20 + idx)
            graph[load_id] = {
                "class_type": "LoadImage",
                "inputs": {"image": img_name},
            }
            enc_id = str(next_blend_id)
            next_blend_id += 1
            graph[enc_id] = {
                "class_type": "MiniMaxH3ImageConditioning",
                "inputs": {
                    "clip": ["3", 0],
                    "vae": ["4", 0],
                    "prompt": branch_prompt,
                    "width": blend_w,
                    "height": blend_h,
                    "images.image_1": [load_id, 0],
                },
            }
            if blend_latent_ref is None:
                blend_latent_ref = [enc_id, 1]

            pos_mod_id = str(next_blend_id)
            next_blend_id += 1
            if req.identity_blend_mode == "velocity_blend":
                graph[pos_mod_id] = {
                    "class_type": "ConditioningSetAreaStrength",
                    "inputs": {"conditioning": [enc_id, 0], "strength": weights[idx]},
                }
            else:
                start_pct, end_pct = ranges[idx]
                graph[pos_mod_id] = {
                    "class_type": "ConditioningSetTimestepRange",
                    "inputs": {"conditioning": [enc_id, 0], "start": start_pct, "end": end_pct},
                }
            pos_branches.append([pos_mod_id, 0])

        positive_ref = pos_branches[0]
        for nxt in pos_branches[1:]:
            comb_id = str(next_blend_id)
            next_blend_id += 1
            graph[comb_id] = {
                "class_type": "ConditioningCombine",
                "inputs": {"conditioning_1": positive_ref, "conditioning_2": nxt},
            }
            positive_ref = [comb_id, 0]

        assert blend_latent_ref is not None
        latent_ref = blend_latent_ref

    else:
        normalized_prompt = normalize_minimax_reference_tags(req.prompt)
        cond_inputs = {
            "clip": ["3", 0],
            "vae": ["4", 0],
            "prompt": normalized_prompt,
            "width": target_w,
            "height": target_h,
        }
        if req.mode == "edit":
            next_id = 20
            for idx, img_name in enumerate(req.image_names, start=1):
                load_id = str(next_id)
                next_id += 1
                graph[load_id] = {
                    "class_type": "LoadImage",
                    "inputs": {"image": img_name},
                }
                cond_inputs[f"images.image_{idx}"] = [load_id, 0]

        graph["5"] = {
            "class_type": "MiniMaxH3ImageConditioning",
            "inputs": cond_inputs,
        }
        positive_ref = ["5", 0]
        latent_ref = ["5", 1]

    # CFG-distilled 6-step checkpoint ignores negative conditioning; Comfy-Org templates wire ConditioningZeroOut.
    graph["6"] = {
        "class_type": "ConditioningZeroOut",
        "inputs": {
            "conditioning": positive_ref,
        },
    }
    negative_ref: list[Any] = ["6", 0]

    graph["11"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["1", 0],
            "seed": int(req.seed),
            "steps": int(req.steps),
            "cfg": float(req.cfg),
            "sampler_name": req.sampler_name,
            "scheduler": req.scheduler,
            "positive": positive_ref,
            "negative": negative_ref,
            "latent_image": latent_ref,
            "denoise": 1.0,
        },
    }
    graph["12"] = {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["11", 0],
            "vae": ["4", 0],
        },
    }

    final_image_ref: list[Any] = ["12", 0]
    if composite_dest_ref is not None and composite_mask_ref is not None:
        graph["14"] = {
            "class_type": "ImageCompositeMasked",
            "inputs": {
                "destination": composite_dest_ref,
                "source": ["12", 0],
                "x": 0,
                "y": 0,
                "resize_source": True,
                "mask": composite_mask_ref,
            },
        }
        final_image_ref = ["14", 0]

    graph["13"] = {
        "class_type": "SaveImage",
        "inputs": {
            "images": final_image_ref,
            "filename_prefix": req.filename_prefix,
        },
    }
    return graph
