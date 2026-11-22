"""ComfyUI API prompt graph builders for Qwen-Image 2.1 (T2I, Edit, Turbo, Base, and Stage-1 PE)."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from qwen_kaggle.constants import MODELS, TURBO_MANUAL_SIGMAS

MAX_REFERENCE_IMAGES = 16
IDENTITY_BLEND_MODES = ("none", "velocity_blend", "step_swap")
_MULTI_IMAGE_TAG_LIST_RE = re.compile(
    r"<image\d+>(?:\s*(?:,|&|\+|and|or)\s*<image\d+>)+",
    flags=re.IGNORECASE,
)
_SECONDARY_IMAGE_TAG_RE = re.compile(r"<image(?:[2-9]|1[0-6])>", flags=re.IGNORECASE)
_EITHER_REF_RE = re.compile(
    r"\bfrom\s+(?:either|both)\s+(?:side\s+of\s+<image1>|references?|reference\s+images?)\b",
    flags=re.IGNORECASE,
)
_BOTH_PARENTS_POSS_RE = re.compile(r"\bboth\s+(?:parents|mothers|fathers)'", flags=re.IGNORECASE)


def prepare_single_branch_prompt(prompt: str) -> str:
    """Normalize a multi-image fusion prompt for an isolated single-reference `<image1>` branch."""
    cleaned = prompt.strip()
    cleaned = _MULTI_IMAGE_TAG_LIST_RE.sub("<image1>", cleaned)
    cleaned = _SECONDARY_IMAGE_TAG_RE.sub("<image1>", cleaned)
    cleaned = _EITHER_REF_RE.sub("from <image1>", cleaned)
    cleaned = _BOTH_PARENTS_POSS_RE.sub("<image1>'s", cleaned)
    if "<image1>" not in cleaned.lower():
        cleaned = f"Using the facial features of <image1>: {cleaned}" if cleaned else "Portrait of the person in <image1>."
    return cleaned


@dataclass(frozen=True)
class GenerationRequest:
    """Normalized generation parameters shared by smoke tests, batch runs, and API gateway."""

    prompt: str
    mode: str = "t2i"  # "t2i" | "edit"
    profile: str = "fast"  # "fast" (Turbo) | "quality" (Base)
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    seed: int = 424242
    steps: int | None = None
    cfg: float | None = None
    sampler_name: str = "euler"
    scheduler: str = "simple"
    resolution: int = 1024
    image_names: tuple[str, ...] = field(default_factory=tuple)
    use_image_latent: bool = True
    identity_blend_mode: str = "none"  # "none" | "velocity_blend" | "step_swap"
    identity_blend_ratio: float = 0.5  # weight of <image2> (0.1..0.9) for 2-image fusion
    cache_device: str = "cpu"  # "auto" | "gpu" | "cpu" | "off"
    cache_dtype: str = "int8"  # "default" | "int8" | "int4"
    filename_prefix: str = "qwen21"
    unet_key_override: str | None = None
    clip_key_override: str | None = None


def _validate_request(req: GenerationRequest) -> None:
    if req.mode not in ("t2i", "edit"):
        raise ValueError(f"Unsupported mode={req.mode!r}; expected 't2i' or 'edit'.")
    if req.profile not in ("fast", "quality"):
        raise ValueError(f"Unsupported profile={req.profile!r}; expected 'fast' or 'quality'.")
    if req.identity_blend_mode not in IDENTITY_BLEND_MODES:
        raise ValueError(
            f"Invalid identity_blend_mode={req.identity_blend_mode!r}; expected one of {IDENTITY_BLEND_MODES}."
        )
    if not (0.1 <= float(req.identity_blend_ratio) <= 0.9):
        raise ValueError("identity_blend_ratio must be between 0.1 and 0.9.")
    if req.cache_device not in ("auto", "gpu", "cpu", "off"):
        raise ValueError(f"Invalid cache_device={req.cache_device!r}.")
    if req.cache_dtype not in ("default", "int8", "int4"):
        raise ValueError(f"Invalid cache_dtype={req.cache_dtype!r}.")
    if req.mode == "edit" and not req.image_names:
        raise ValueError("Edit mode requires at least one image name in image_names.")
    if len(req.image_names) > MAX_REFERENCE_IMAGES:
        raise ValueError(f"Too many reference images ({len(req.image_names)} > {MAX_REFERENCE_IMAGES}).")
    if req.width <= 0 or req.height <= 0:
        raise ValueError(f"Invalid dimensions: {req.width}x{req.height}.")


def build_qwen21_api_prompt(req: GenerationRequest) -> dict[str, dict[str, Any]]:
    """Build a ComfyUI API-format prompt graph verified against Comfy-Org/ComfyUI nodes."""
    _validate_request(req)

    default_unet_key = "unet_turbo" if req.profile == "fast" else "unet_base"
    unet_key = req.unet_key_override or default_unet_key
    clip_key = req.clip_key_override or "clip_vl"
    unet_name = MODELS[unet_key].filename
    clip_name = MODELS[clip_key].filename
    vae_name = MODELS["vae"].filename

    graph: dict[str, dict[str, Any]] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": unet_name,
                "weight_dtype": "default",
            },
        },
        "2": {
            "class_type": "QwenImage21Cache",
            "inputs": {
                "model": ["1", 0],
                "device": req.cache_device,
                "dtype": req.cache_dtype,
            },
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": clip_name,
                "type": "qwen_image",
            },
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": vae_name,
            },
        },
    }

    use_diffusion_blend = (
        req.mode == "edit"
        and len(req.image_names) >= 2
        and req.identity_blend_mode in ("velocity_blend", "step_swap")
    )

    if use_diffusion_blend:
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
        neg_branches: list[list[Any]] = []

        for idx, img_name in enumerate(req.image_names):
            load_id = str(20 + idx)
            graph[load_id] = {
                "class_type": "LoadImage",
                "inputs": {"image": img_name},
            }
            enc_id = str(next_blend_id)
            next_blend_id += 1
            graph[enc_id] = {
                "class_type": "TextEncodeQwenImage21",
                "inputs": {
                    "clip": ["3", 0],
                    "prompt": branch_prompt,
                    "negative_prompt": req.negative_prompt,
                    "resolution": req.resolution,
                    "vae": ["4", 0],
                    "images.image_1": [load_id, 0],
                },
            }

            pos_mod_id = str(next_blend_id)
            next_blend_id += 1
            neg_mod_id = str(next_blend_id)
            next_blend_id += 1
            if req.identity_blend_mode == "velocity_blend":
                strength = weights[idx]
                graph[pos_mod_id] = {
                    "class_type": "ConditioningSetAreaStrength",
                    "inputs": {"conditioning": [enc_id, 0], "strength": strength},
                }
                graph[neg_mod_id] = {
                    "class_type": "ConditioningSetAreaStrength",
                    "inputs": {"conditioning": [enc_id, 1], "strength": strength},
                }
            else:
                start_pct, end_pct = ranges[idx]
                graph[pos_mod_id] = {
                    "class_type": "ConditioningSetTimestepRange",
                    "inputs": {"conditioning": [enc_id, 0], "start": start_pct, "end": end_pct},
                }
                graph[neg_mod_id] = {
                    "class_type": "ConditioningSetTimestepRange",
                    "inputs": {"conditioning": [enc_id, 1], "start": start_pct, "end": end_pct},
                }
            pos_branches.append([pos_mod_id, 0])
            neg_branches.append([neg_mod_id, 0])

        positive_ref = pos_branches[0]
        for nxt in pos_branches[1:]:
            comb_id = str(next_blend_id)
            next_blend_id += 1
            graph[comb_id] = {
                "class_type": "ConditioningCombine",
                "inputs": {"conditioning_1": positive_ref, "conditioning_2": nxt},
            }
            positive_ref = [comb_id, 0]

        negative_ref = neg_branches[0]
        for nxt in neg_branches[1:]:
            comb_id = str(next_blend_id)
            next_blend_id += 1
            graph[comb_id] = {
                "class_type": "ConditioningCombine",
                "inputs": {"conditioning_1": negative_ref, "conditioning_2": nxt},
            }
            negative_ref = [comb_id, 0]

        # Diffusion-level identity fusion always generates on a fresh EmptyLatentImage so neither reference locks framing.
        graph["6"] = {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": req.width,
                "height": req.height,
                "batch_size": 1,
            },
        }
        latent_ref: list[Any] = ["6", 0]
    else:
        encode_inputs: dict[str, Any] = {
            "clip": ["3", 0],
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "resolution": req.resolution,
        }

        if req.mode == "edit":
            encode_inputs["vae"] = ["4", 0]
            next_id = 20
            for idx, img_name in enumerate(req.image_names, start=1):
                node_id = str(next_id)
                next_id += 1
                graph[node_id] = {
                    "class_type": "LoadImage",
                    "inputs": {"image": img_name},
                }
                encode_inputs[f"images.image_{idx}"] = [node_id, 0]

        graph["5"] = {
            "class_type": "TextEncodeQwenImage21",
            "inputs": encode_inputs,
        }
        positive_ref = ["5", 0]
        negative_ref = ["5", 1]

        if req.mode == "edit" and req.use_image_latent:
            latent_ref = ["5", 2]
        else:
            graph["6"] = {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": req.width,
                    "height": req.height,
                    "batch_size": 1,
                },
            }
            latent_ref = ["6", 0]

    if req.profile == "fast":
        cfg_val = 1.0 if req.cfg is None else float(req.cfg)
        graph["7"] = {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": int(req.seed)},
        }
        graph["8"] = {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": req.sampler_name},
        }
        graph["9"] = {
            "class_type": "ManualSigmas",
            "inputs": {"sigmas": TURBO_MANUAL_SIGMAS},
        }
        graph["10"] = {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["2", 0],
                "positive": positive_ref,
                "negative": negative_ref,
                "cfg": cfg_val,
            },
        }
        graph["11"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["7", 0],
                "guider": ["10", 0],
                "sampler": ["8", 0],
                "sigmas": ["9", 0],
                "latent_image": latent_ref,
            },
        }
        samples_ref: list[Any] = ["11", 0]
    else:
        steps_val = 25 if req.steps is None else int(req.steps)
        cfg_val = 1.0 if req.cfg is None else float(req.cfg)
        graph["11"] = {
            "class_type": "KSampler",
            "inputs": {
                "model": ["2", 0],
                "seed": int(req.seed),
                "steps": steps_val,
                "cfg": cfg_val,
                "sampler_name": req.sampler_name,
                "scheduler": req.scheduler,
                "positive": positive_ref,
                "negative": negative_ref,
                "latent_image": latent_ref,
                "denoise": 1.0,
            },
        }
        samples_ref = ["11", 0]

    graph["12"] = {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": samples_ref,
            "vae": ["4", 0],
        },
    }
    graph["13"] = {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["12", 0],
            "filename_prefix": req.filename_prefix,
        },
    }
    return graph


def build_refine_prompt_graph(
    prompt: str,
    *,
    mode: str = "edit",
    image_names: tuple[str, ...] = (),
    seed: int = 424242,
    thinking: bool = True,
    max_length: int = 4096,
    system_prompt: str = "",
) -> dict[str, dict[str, Any]]:
    """Build a decoupled Stage-1 Prompt Enhancer graph (run with POST /free before Stage 2)."""
    if mode not in ("t2i", "edit"):
        raise ValueError(f"Unsupported mode={mode!r}.")

    pe_key = "pe_i2i" if mode == "edit" else "pe_t2i"
    clip_type = "qwen_image" if mode == "edit" else "stable_diffusion"

    graph: dict[str, dict[str, Any]] = {
        "1": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": MODELS[pe_key].filename,
                "type": clip_type,
            },
        }
    }

    textgen_inputs: dict[str, Any] = {
        "clip": ["1", 0],
        "prompt": prompt,
        "max_length": int(max_length),
        "sampling_mode": "on",
        "sampling_mode.temperature": 1.0,
        "sampling_mode.top_k": 20,
        "sampling_mode.top_p": 0.95,
        "sampling_mode.min_p": 0.0,
        "sampling_mode.repetition_penalty": 1.0,
        "sampling_mode.presence_penalty": 1.5 if mode == "t2i" else 0.0,
        "sampling_mode.seed": int(seed),
        "thinking": bool(thinking),
        "use_default_template": not bool(system_prompt),
        "mtp": "auto",
    }
    if system_prompt:
        textgen_inputs["system_prompt"] = system_prompt

    next_id = 10
    for idx, img_name in enumerate(image_names, start=1):
        node_id = str(next_id)
        next_id += 1
        graph[node_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": img_name},
        }
        textgen_inputs[f"images.image{idx}"] = [node_id, 0]

    graph["2"] = {
        "class_type": "TextGenerateV2",
        "inputs": textgen_inputs,
    }
    graph["3"] = {
        "class_type": "PreviewAny",
        "inputs": {"source": ["2", 0]},
    }
    graph["4"] = {
        "class_type": "PreviewAny",
        "inputs": {"source": ["2", 1]},
    }
    return graph
