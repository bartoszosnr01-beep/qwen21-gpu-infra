"""TPU v5e-8 (8x16 GiB = 128 GiB HBM) PyTorch/XLA SPMD runtime & static-shape bridge for MiniMax-H3-Image.

Patches `comfy.ldm.minimax.model`:
  - Replaces `comfy_kitchen` CUDA/Triton `rms_rope_split_half_` with pure-PyTorch XLA-traceable
    per-head RMSNorm + partial split-half RoPE (matching `comfy_kitchen.backends.eager.rope`).
  - Replaces in-place `_mod_scale_shift` / `_mod_gate` slice mutations with out-of-place XLA ops.
  - Pads variable-length text context to static buckets (`256/512/1024/2048`) while preserving
    exact unpadded RoPE coordinates for reference/target tokens and masking padded key columns
    in both `TokenRefiner` and all 50 `DiTBlock` attention layers.
  - Dequantizes `int8_convrot` weights to native `torch.bfloat16` (preserving FP32 islands:
    `RMSNorm`, `video_patch_proj`, `final_layer.video_out`, `adaln_t_table`, `rope.inv_freq`)
    and shards 2D+ matrices across all 8 TPU v5e chips via 1D SPMD `('model', None)` / `(None, 'model')`
    while keeping `final_layer.video_out` replicated so 6-step PDD head reshaping is conflict-free.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from minimax_kaggle.constants import (
    DEFAULT_COMFY_DIR,
    TPU_CONTEXT_BUCKETS,
    TPU_SAFE_COMFY_FLAGS,
)

TPU_TOTAL_HBM_BYTES = 128 * (1024**3)  # 8 x 16 GiB HBM on Kaggle TPU v5e-8
TPU_TOTAL_HBM_MB = 131072.0
MIN_SHARD_NUMEL = 65536


def is_tpu_environment() -> bool:
    """Detect whether the current container has Google Cloud TPU v5e devices."""
    if os.environ.get("MINIMAX_FORCE_TPU") == "1":
        return True
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
        return True
    if any(k.startswith("TPU_") for k in os.environ):
        return True
    if Path("/dev/accel0").exists() or Path("/dev/vfio/0").exists():
        return True
    try:
        import torch_xla.runtime as xr  # type: ignore[import-not-found]

        return xr.device_type().upper() == "TPU"
    except Exception:
        return False


def choose_static_context_bucket(
    seq_len: int,
    buckets: tuple[int, ...] = TPU_CONTEXT_BUCKETS,
) -> int:
    """Round prompt sequence length up to a fixed bucket to prevent XLA recompilation."""
    if seq_len <= 0:
        return buckets[0]
    for b in buckets:
        if seq_len <= b:
            return b
    step = buckets[-1]
    return ((seq_len + step - 1) // step) * step


def pad_minimax_context(
    context: torch.Tensor,
    text_token_tags: Any = None,
    buckets: tuple[int, ...] = TPU_CONTEXT_BUCKETS,
) -> tuple[torch.Tensor, Any, int, int]:
    """Right-pad `[1, L_real, D]` MiniMax text context (and optional `text_token_tags`) to `L_bucket`.

    Returns `(padded_context, padded_tags, real_len, bucket_len)`.
    """
    real_len = int(context.shape[1])
    bucket_len = choose_static_context_bucket(real_len, buckets=buckets)
    pad_len = bucket_len - real_len
    if pad_len <= 0:
        return context, text_token_tags, real_len, real_len

    pad = torch.zeros(
        (context.shape[0], pad_len, *context.shape[2:]),
        dtype=context.dtype,
        device=context.device,
    )
    padded_ctx = torch.cat([context, pad], dim=1)

    padded_tags = text_token_tags
    if text_token_tags is not None:
        if isinstance(text_token_tags, torch.Tensor):
            tail_val = text_token_tags[-1] if text_token_tags.numel() > 0 else torch.tensor(0, dtype=text_token_tags.dtype, device=text_token_tags.device)
            tag_pad = tail_val.expand(pad_len).to(device=text_token_tags.device, dtype=text_token_tags.dtype)
            padded_tags = torch.cat([text_token_tags, tag_pad], dim=0)
        elif isinstance(text_token_tags, (list, tuple)):
            tail_val = text_token_tags[-1] if len(text_token_tags) > 0 else 0
            padded_tags = list(text_token_tags) + [tail_val] * pad_len

    return padded_ctx, padded_tags, real_len, bucket_len


def build_static_tpu_layout_and_masks(
    real_len: int,
    bucket_len: int,
    latent_t: int,
    lat_h: int,
    lat_w: int,
    refs: list[dict[str, Any]] | None,
    packed_layout_cls: Any,
    device: torch.device,
) -> tuple[Any, torch.Tensor | None, torch.Tensor | None]:
    """Build a static-bucket `PackedLayout` preserving exact unpadded RoPE positions and key masks.

    - Shifts post-text `position_ids[:, 0]` back by `(bucket_len - real_len)` so reference and
      target image RoPE coordinates match unpadded execution bit-for-bit.
    - Returns `(layout, refiner_attn_mask, dit_attn_mask)` where boolean `False` masks out padded
      text keys `[real_len:bucket_len]` in both `TokenRefiner` and `DiTBlock` self-attention.
    """
    layout = packed_layout_cls(bucket_len, latent_t, lat_h, lat_w, 0, refs=refs)
    pad_len = bucket_len - real_len
    if pad_len <= 0:
        return layout, None, None

    if hasattr(layout, "position_ids") and layout.position_ids.shape[0] > bucket_len:
        pos = layout.position_ids.clone()
        pos[bucket_len:, 0] = pos[bucket_len:, 0] - float(pad_len)
        layout.position_ids = pos

    refiner_mask = torch.ones((1, 1, bucket_len, bucket_len), dtype=torch.bool, device=device)
    refiner_mask[:, :, :, real_len:bucket_len] = False

    seq_len = int(layout.seq_len)
    dit_mask = torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device=device)
    dit_mask[:, :, :, real_len:bucket_len] = False
    return layout, refiner_mask, dit_mask


def xla_apply_rope_split_half1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch out-of-place split-half RoPE matching `comfy_kitchen.backends.eager.rope.apply_rope_split_half1`."""
    t = x
    if torch.promote_types(x.dtype, freqs_cis.dtype) != freqs_cis.dtype:
        t = x.to(dtype=freqs_cis.dtype)
    if freqs_cis.ndim == 2:
        freqs_cis = freqs_cis.unsqueeze(0)
    half = x.shape[-1] // 2
    x0 = t[..., :half]
    x1 = t[..., half:]
    out0 = x0 * freqs_cis[..., 0, 0] + x1 * freqs_cis[..., 0, 1]
    out1 = x1 * freqs_cis[..., 1, 1] + x0 * freqs_cis[..., 1, 0]
    return torch.cat((out0, out1), dim=-1).to(dtype=x.dtype)


def xla_rms_rope_split_half(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-5,
    rot_dim: int = 0,
) -> torch.Tensor:
    """Pure-PyTorch XLA-traceable per-head RMSNorm + partial split-half RoPE."""
    x_norm = F.rms_norm(x, (x.shape[-1],), weight=weight, eps=epsilon)
    if rot_dim and rot_dim != x.shape[-1]:
        rotated = xla_apply_rope_split_half1(x_norm[..., :rot_dim], freqs_cis)
        return torch.cat((rotated, x_norm[..., rot_dim:]), dim=-1)
    return xla_apply_rope_split_half1(x_norm, freqs_cis)


def _is_quantized_weight(obj: Any) -> bool:
    """Return True only for `comfy_kitchen` `QuantizedTensor` or quantized tensors, not plain float Tensors."""
    if obj is None:
        return False
    if hasattr(obj, "_qdata") or obj.__class__.__name__ == "QuantizedTensor":
        return True
    if isinstance(obj, torch.Tensor):
        return bool(getattr(obj, "is_quantized", False))
    return hasattr(obj, "dequantize")


def dequantize_module_weights_to_bf16(module: nn.Module) -> int:
    """Convert any `QuantizedTensor` parameters/buffers inside `module` to native `torch.bfloat16`.

    Returns the number of dequantized tensors. Allows reusing `int8_convrot` checkpoints on TPU v5e-8.
    """
    converted = 0
    for sub in module.modules():
        for name, param in list(sub._parameters.items()):
            if param is None:
                continue
            data = getattr(param, "data", param)
            if _is_quantized_weight(param) or _is_quantized_weight(data):
                q_obj = param if _is_quantized_weight(param) else data
                dq = q_obj.dequantize().to(dtype=torch.bfloat16)
                sub._parameters[name] = nn.Parameter(dq, requires_grad=False)
                converted += 1
        for name, buf in list(sub._buffers.items()):
            if _is_quantized_weight(buf):
                sub._buffers[name] = buf.dequantize().to(dtype=torch.bfloat16)
                converted += 1
    return converted


def choose_sharding_spec(
    shape: tuple[int, ...],
    num_chips: int,
    min_numel: int = MIN_SHARD_NUMEL,
) -> tuple[str | None, ...] | None:
    """Compute a 1D SPMD `('model', ...)` partition spec for a weight tensor of `shape`."""
    if num_chips <= 1 or len(shape) < 2:
        return None
    numel = 1
    for dim in shape:
        numel *= dim
    if numel < min_numel:
        return None
    if shape[0] % num_chips == 0:
        return ("model",) + (None,) * (len(shape) - 1)
    if shape[1] % num_chips == 0:
        return (None, "model") + (None,) * (len(shape) - 2)
    return None


# Submodule path fragments that must stay in FP32 and/or replicated for MiniMax-H3-Image fidelity
_FP32_ISLAND_FRAGMENTS = (
    "video_patch_proj",
    "audio_patch_proj",
    "final_layer.video_out",
    "final_layer.audio_out",
    "time_embedder",
)
_REPLICATE_ONLY_FRAGMENTS = (
    "final_layer.video_out",
    "final_layer.audio_out",
    "video_patch_proj",
    "audio_patch_proj",
)


def shard_module_for_spmd(
    module: nn.Module,
    xla_device: torch.device,
    mesh: Any,
    num_chips: int,
    mark_sharding_fn: Any,
) -> dict[str, int]:
    """Move `module` weights to `xla_device` (BF16 for DiT blocks, FP32 for sensitive islands) and shard 2D+ matrices."""
    dequantized = dequantize_module_weights_to_bf16(module)
    sharded = 0
    replicated = 0

    for mod_path, sub in module.named_modules():
        cls_name = sub.__class__.__name__
        is_norm = "RMSNorm" in cls_name or "LayerNorm" in cls_name
        is_fp32_island = is_norm or any(frag in mod_path for frag in _FP32_ISLAND_FRAGMENTS)
        replicate_only = any(frag in mod_path for frag in _REPLICATE_ONLY_FRAGMENTS)

        for name, param in list(sub._parameters.items()):
            if param is None:
                continue
            keep_fp32 = is_fp32_island or ("adaln" in mod_path and param.dtype == torch.float32)
            target_dtype = (
                torch.float32
                if (keep_fp32 and param.dtype == torch.float32)
                else (torch.bfloat16 if param.is_floating_point() else param.dtype)
            )
            moved = param.detach().to(device=xla_device, dtype=target_dtype)
            spec = None if replicate_only else choose_sharding_spec(tuple(moved.shape), num_chips)
            if spec is not None and mesh is not None and mark_sharding_fn is not None:
                mark_sharding_fn(moved, mesh, spec)
                sharded += 1
            else:
                replicated += 1
            sub._parameters[name] = nn.Parameter(moved, requires_grad=False)

        for name, buf in list(sub._buffers.items()):
            if buf is None:
                continue
            target_dtype = (
                torch.float32
                if (buf.dtype == torch.float32 and buf.is_floating_point())
                else (torch.bfloat16 if buf.is_floating_point() else buf.dtype)
            )
            moved_buf = buf.detach().to(device=xla_device, dtype=target_dtype)
            spec = None if replicate_only else choose_sharding_spec(tuple(moved_buf.shape), num_chips)
            if spec is not None and mesh is not None and mark_sharding_fn is not None:
                mark_sharding_fn(moved_buf, mesh, spec)
                sharded += 1
            sub._buffers[name] = moved_buf

    return {"dequantized": dequantized, "sharded": sharded, "replicated": replicated}


class TpuXlaContext:
    """Holds initialized PyTorch/XLA SPMD device, 8-chip mesh, and compilation cache state."""

    def __init__(self, cache_dir: Path = Path("/tmp/minimax_xla_cache")) -> None:
        self.cache_dir = cache_dir
        self.device: torch.device = torch.device("cpu")
        self.num_chips: int = 1
        self.mesh: Any = None
        self.xm: Any = None
        self.xr: Any = None
        self.xs: Any = None
        self._sharded_models: set[int] = set()

    def initialize(self) -> None:
        os.environ.pop("TPU_LIBRARY_PATH", None)
        os.environ.setdefault("PJRT_DEVICE", "TPU")
        os.environ.setdefault("XLA_USE_BF16", "1")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        import numpy as np
        import torch_xla.core.xla_model as xm  # type: ignore[import-not-found]
        import torch_xla.distributed.spmd as xs  # type: ignore[import-not-found]
        import torch_xla.runtime as xr  # type: ignore[import-not-found]

        try:
            xr.initialize_cache(str(self.cache_dir), readonly=False)
        except Exception as exc:
            print(f"[TPU] XLA persistent cache init warning: {exc}", flush=True)

        xr.use_spmd()
        self.xm = xm
        self.xr = xr
        self.xs = xs
        self.num_chips = int(xr.global_runtime_device_count())
        self.device = xm.xla_device()
        if self.num_chips > 1:
            device_ids = np.arange(self.num_chips)
            self.mesh = xs.Mesh(device_ids, (self.num_chips,), ("model",))
            xs.set_global_mesh(self.mesh)
        print(
            f"[TPU] Initialized PyTorch/XLA SPMD on {self.device} across {self.num_chips} TPU v5e chips "
            f"({self.num_chips * 16} GiB HBM total, cache={self.cache_dir})",
            flush=True,
        )

    def mark_step(self) -> None:
        if self.xm is not None:
            self.xm.mark_step()

    def ensure_model_on_tpu(self, module: nn.Module) -> None:
        mid = id(module)
        if mid in self._sharded_models:
            return
        stats = shard_module_for_spmd(
            module,
            xla_device=self.device,
            mesh=self.mesh,
            num_chips=self.num_chips,
            mark_sharding_fn=self.xs.mark_sharding if self.xs is not None else None,
        )
        self.mark_step()
        self._sharded_models.add(mid)
        print(
            f"[TPU] Sharded {module.__class__.__name__} across {self.num_chips} chips: "
            f"sharded={stats['sharded']}, replicated={stats['replicated']}, dequantized={stats['dequantized']}",
            flush=True,
        )


def _move_payload_to_tpu(payload: dict[str, Any], device: torch.device, padded_tags: Any) -> dict[str, Any]:
    """Shallow-copy `minimax_payload`, moving reference latents and tags onto XLA TPU device."""
    out = dict(payload)
    if padded_tags is not None:
        if isinstance(padded_tags, torch.Tensor):
            out["text_token_tags"] = padded_tags.to(device=device)
        else:
            out["text_token_tags"] = padded_tags
    refs = payload.get("refs")
    if refs:
        moved_refs: list[dict[str, Any]] = []
        for blk in refs:
            blk_copy = dict(blk)
            if isinstance(blk_copy.get("latent"), torch.Tensor):
                blk_copy["latent"] = blk_copy["latent"].to(device=device, dtype=torch.float32)
            moved_refs.append(blk_copy)
        out["refs"] = moved_refs
    return out


def install_comfy_tpu_patches(ctx: TpuXlaContext) -> None:
    """Patch ComfyUI's `model_management`, `comfy_kitchen`, and `MiniMaxH3ImageModel` for TPU v5e-8."""
    try:
        import comfy_kitchen as ck  # type: ignore[import-not-found]

        ck.registry.disable("cuda")
        ck.registry.disable("triton")
    except Exception:
        pass

    import comfy.ldm.minimax.model as minimax_mod  # type: ignore[import-not-found]
    import comfy.model_management as mm  # type: ignore[import-not-found]

    mm.vram_state = mm.VRAMState.HIGH_VRAM
    mm.set_vram_to = mm.VRAMState.HIGH_VRAM
    mm.total_vram = TPU_TOTAL_HBM_BYTES // (1024 * 1024)

    _orig_get_total_memory = mm.get_total_memory
    _orig_get_free_memory = mm.get_free_memory

    def _tpu_get_total_memory(dev: Any = None, torch_total_too: bool = False) -> Any:
        if dev is not None and str(dev).startswith("xla"):
            return (TPU_TOTAL_HBM_BYTES, TPU_TOTAL_HBM_BYTES) if torch_total_too else TPU_TOTAL_HBM_BYTES
        return _orig_get_total_memory(dev, torch_total_too=torch_total_too)

    def _tpu_get_free_memory(dev: Any = None, torch_free_too: bool = False) -> Any:
        if dev is not None and str(dev).startswith("xla"):
            free_b = int(TPU_TOTAL_HBM_BYTES * 0.85)
            return (free_b, free_b) if torch_free_too else free_b
        return _orig_get_free_memory(dev, torch_free_too=torch_free_too)

    mm.get_total_memory = _tpu_get_total_memory
    mm.get_free_memory = _tpu_get_free_memory
    mm.should_use_bf16 = lambda device=None, model_params=0, prioritize_performance=True, manual_cast=False: True

    # Out-of-place XLA-safe AdaLN modulation helpers (replaces in-place slice mutations)
    def _xla_mod_scale_shift(h: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, segments: list[Any]) -> torch.Tensor:
        if len(segments) == 1:
            a, b, row = segments[0]
            return h[a:b] * (1.0 + minimax_mod._mod_row(scale, row, h.dtype)) + minimax_mod._mod_row(shift, row, h.dtype)
        parts = [
            h[a:b] * (1.0 + minimax_mod._mod_row(scale, row, h.dtype)) + minimax_mod._mod_row(shift, row, h.dtype)
            for a, b, row in segments
        ]
        return torch.cat(parts, dim=0)

    def _xla_mod_gate(x: torch.Tensor, gate: torch.Tensor, other: torch.Tensor, segments: list[Any]) -> torch.Tensor:
        if len(segments) == 1:
            a, b, row = segments[0]
            return x[a:b] + other[a:b] * minimax_mod._mod_row(gate, row, x.dtype)
        parts = [
            x[a:b] + other[a:b] * minimax_mod._mod_row(gate, row, x.dtype)
            for a, b, row in segments
        ]
        return torch.cat(parts, dim=0)

    # Pure-PyTorch XLA Attention.forward replacing `comfy.quant_ops.ck.rms_rope_split_half_`
    def _xla_attention_forward(
        self: Any,
        x: torch.Tensor,
        rope_freqs: torch.Tensor | None = None,
        transformer_options: dict[str, Any] = {},
    ) -> torch.Tensor:
        s = x.shape[0]
        q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
        v = v.view(s, self.heads, self.head_dim)
        if rope_freqs is not None:
            q_4d = q.view(1, s, self.heads, self.head_dim)
            k_4d = k.view(1, s, self.heads, self.head_dim)
            qw = mm.cast_to(self.q_norm.weight, device=x.device, dtype=x.dtype)
            kw = mm.cast_to(self.k_norm.weight, device=x.device, dtype=x.dtype)
            rot = rope_freqs.shape[-3] * 2
            q = xla_rms_rope_split_half(q_4d, rope_freqs, qw, epsilon=self.q_norm.eps, rot_dim=rot)[0]
            k = xla_rms_rope_split_half(k_4d, rope_freqs, kw, epsilon=self.k_norm.eps, rot_dim=rot)[0]
            attn_mask = transformer_options.get("minimax_tpu_dit_mask")
        else:
            q = self.q_norm(q.view(s, self.heads, self.head_dim))
            k = self.k_norm(k.view(s, self.heads, self.head_dim))
            attn_mask = transformer_options.get("minimax_tpu_refiner_mask")

        q_c = minimax_mod.AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
        k_c = minimax_mod.AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
        v_c = minimax_mod.AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
        out = minimax_mod.optimized_attention(
            q_c,
            k_c,
            v_c,
            self.heads,
            preferred_attention=self.comfy_attention,
            mask=attn_mask,
            skip_reshape=True,
            transformer_options=transformer_options,
        )
        return self.out_proj(out.squeeze(0))

    minimax_mod._mod_scale_shift = _xla_mod_scale_shift
    minimax_mod._mod_gate = _xla_mod_gate
    minimax_mod.Attention.forward = _xla_attention_forward

    _orig_image_forward = minimax_mod.MiniMaxH3ImageModel._forward

    def _tpu_minimax_image_forward(
        self: Any,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        transformer_options: dict[str, Any] = {},
        minimax_payload: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        ctx.ensure_model_on_tpu(self)
        orig_device = x.device
        orig_dtype = x.dtype

        payload = dict(minimax_payload or {})
        raw_tags = payload.get("text_token_tags")
        padded_ctx, padded_tags, real_len, bucket_len = pad_minimax_context(
            context,
            text_token_tags=raw_tags,
            buckets=TPU_CONTEXT_BUCKETS,
        )

        # Compute padded x spatial dimensions matching MiniMaxH3ImageModel._forward
        import comfy.ldm.common_dit as common_dit  # type: ignore[import-not-found]

        x_padded = common_dit.pad_to_patch_size(x, self.patch_size)
        latent_t, lat_h, lat_w = int(x_padded.shape[2]), int(x_padded.shape[3]), int(x_padded.shape[4])
        layout, refiner_mask, dit_mask = build_static_tpu_layout_and_masks(
            real_len=real_len,
            bucket_len=bucket_len,
            latent_t=latent_t,
            lat_h=lat_h,
            lat_w=lat_w,
            refs=payload.get("refs"),
            packed_layout_cls=minimax_mod.PackedLayout,
            device=ctx.device,
        )

        tpu_payload = _move_payload_to_tpu(payload, ctx.device, padded_tags)
        tpu_payload["layout"] = layout

        x_tpu = x.to(device=ctx.device, dtype=torch.float32)
        c_tpu = padded_ctx.to(device=ctx.device, dtype=torch.bfloat16)

        t_opts = dict(transformer_options)
        if refiner_mask is not None:
            t_opts["minimax_tpu_refiner_mask"] = refiner_mask
        else:
            t_opts.pop("minimax_tpu_refiner_mask", None)
        if dit_mask is not None:
            t_opts["minimax_tpu_dit_mask"] = dit_mask
        else:
            t_opts.pop("minimax_tpu_dit_mask", None)

        out_tpu = _orig_image_forward(
            self,
            x_tpu,
            timestep,
            c_tpu,
            transformer_options=t_opts,
            minimax_payload=tpu_payload,
            **kwargs,
        )
        ctx.mark_step()
        return out_tpu.to(device=orig_device, dtype=orig_dtype)

    minimax_mod.MiniMaxH3ImageModel._forward = _tpu_minimax_image_forward
    print("[TPU] Installed ComfyUI MiniMaxH3ImageModel static-shape & 8-chip SPMD patches.", flush=True)


def launch_comfy_on_tpu(comfy_dir: Path, comfy_args: list[str]) -> None:
    """Initialize TPU v5e-8 SPMD context, patch ComfyUI, and run ComfyUI `main.py`."""
    comfy_dir = comfy_dir.resolve()
    if str(comfy_dir) not in sys.path:
        sys.path.insert(0, str(comfy_dir))

    ctx = TpuXlaContext()
    ctx.initialize()

    # Run ComfyUI host orchestration with --cpu so Qwen3-VL-32B text/vision encoding and VAE
    # stay on Kaggle TPU VM's ~330 GiB host RAM while MiniMaxH3ImageModel (33.1B DiT, 6 steps)
    # executes sharded in BF16 across all 8 TPU v5e chips (128 GiB HBM).
    sys.argv = [str(comfy_dir / "main.py"), "--cpu", *comfy_args]
    install_comfy_tpu_patches(ctx)

    main_py = comfy_dir / "main.py"
    code = compile(main_py.read_text(encoding="utf-8"), str(main_py), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(main_py)})


def run_tpu_smoke_suite(
    *,
    comfy_dir: Path = DEFAULT_COMFY_DIR,
    venv_dir: Path = Path("/tmp/venv"),
    input_dir: Path = Path("/kaggle/input"),
    out_dir: Path = Path("/kaggle/working/minimax_tpu_smoke_results"),
    preset: str = "tpu_int8",
    width: int = 2048,
    height: int = 2048,
) -> dict[str, Any]:
    """End-to-end TPU v5e-8 qualification test: bootstrap, shard to 8 chips, verify static-shape warmup."""
    import json
    import subprocess
    from minimax_kaggle.bootstrap import (
        collect_system_diagnostics,
        ensure_models_and_templates,
        ensure_physical_tpu_or_raise,
        setup_comfy_and_venv,
    )
    from minimax_kaggle.smoke_test import _run_prompt_and_collect, _wait_for_comfy
    from minimax_kaggle.workflows import GenerationRequest, build_minimax_h3_api_prompt

    out_dir.mkdir(parents=True, exist_ok=True)
    diag = collect_system_diagnostics(input_dir=input_dir)
    print("[TPU-SMOKE] Diagnostics:", json.dumps(diag, indent=2), flush=True)

    sources: dict[str, str] = {}
    results: list[dict[str, Any]] = []
    proc: subprocess.Popen[str] | None = None
    log_path = out_dir / "comfy_tpu0.log"
    log_fh = None

    dit_key = "dit_bf16" if preset == "tpu_bf16" else "dit_int8"
    clip_key = "clip_int8"

    try:
        ensure_physical_tpu_or_raise(diag)
        py_exe = setup_comfy_and_venv(comfy_dir, venv_dir, accelerator="tpu", install_gguf_node=False)
        sources = ensure_models_and_templates(
            preset,
            comfy_dir=comfy_dir,
            venv_dir=venv_dir,
            input_root=input_dir,
        )
        print("[TPU-SMOKE] Model sources:", json.dumps(sources, indent=2), flush=True)

        pkg_root = str(Path(__file__).resolve().parent.parent)
        env = dict(os.environ)
        env.pop("TPU_LIBRARY_PATH", None)
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{pkg_root}{os.pathsep}{existing_pp}" if existing_pp else pkg_root
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("PJRT_DEVICE", "TPU")

        log_fh = log_path.open("w", encoding="utf-8")
        cmd = [
            str(py_exe),
            "-m",
            "minimax_kaggle.tpu_runtime",
            "--comfy-dir",
            str(comfy_dir),
            "--listen",
            "127.0.0.1",
            "--port",
            "8188",
            *TPU_SAFE_COMFY_FLAGS,
        ]
        print(f"[TPU-SMOKE] Launching TPU ComfyUI worker: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(
            cmd,
            cwd=str(comfy_dir),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
        base_url = "http://127.0.0.1:8188"
        _wait_for_comfy(base_url, proc, timeout_s=240)
        print("[TPU-SMOKE] ComfyUI TPU worker ready at", base_url, flush=True)

        cases = [
            (
                f"tpu_cold_compile_{width}x{height}",
                GenerationRequest(
                    prompt=(
                        "Studio product photograph of a sleek obsidian glass sphere resting on a matte "
                        "dark graphite pedestal, soft warm rim lighting from upper left, subtle reflections, "
                        "crisp macro focus, neutral charcoal backdrop."
                    ),
                    mode="t2i",
                    profile="fast",
                    width=width,
                    height=height,
                    seed=101,
                    dit_key=dit_key,
                    clip_key=clip_key,
                    filename_prefix="minimax_tpu_cold",
                ),
            ),
            (
                f"tpu_warm_cached_diff_len_{width}x{height}",
                GenerationRequest(
                    prompt=(
                        "Photorealistic architectural photograph of a minimalist poured-concrete and warm "
                        "cedar lakeside pavilion at blue hour, serene mirror reflections across calm alpine "
                        "water, soft interior tungsten glow through floor-to-ceiling frameless glass, "
                        "35mm tilt-shift lens, fine natural film grain and crisp structural geometry."
                    ),
                    mode="t2i",
                    profile="fast",
                    width=width,
                    height=height,
                    seed=202,
                    dit_key=dit_key,
                    clip_key=clip_key,
                    filename_prefix="minimax_tpu_warm",
                ),
            ),
        ]

        for name, req in cases:
            print(f"[TPU-SMOKE] Running {name} (prompt_chars={len(req.prompt)})...", flush=True)
            graph = build_minimax_h3_api_prompt(req)
            res = _run_prompt_and_collect(
                base_url,
                py_exe,
                comfy_dir,
                out_dir,
                name,
                graph,
                timeout_s=2400,
            )
            print(f"[TPU-SMOKE] Completed {name}: {json.dumps(res)}", flush=True)
            results.append(res)
    except Exception as exc:
        print(f"[TPU-SMOKE] ERROR: {exc}", flush=True)
        results.append({"name": "tpu_smoke_error", "status": "FAIL", "error": str(exc)})
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except Exception:
                proc.kill()
        if log_fh is not None:
            log_fh.close()
        if log_path.exists():
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-100:])
            print("=== [TPU WORKER LOG TAIL] ===\n" + tail + "\n=== [END TPU LOG] ===", flush=True)

    report = {"diagnostics": diag, "sources": sources, "results": results}
    (out_dir / "tpu_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="ComfyUI MiniMax-H3-Image TPU v5e-8 SPMD Worker Launcher")
    parser.add_argument("--comfy-dir", type=Path, default=DEFAULT_COMFY_DIR)
    args, remaining = parser.parse_known_args()
    if not remaining:
        remaining = ["--listen", "127.0.0.1", "--port", "8188", *TPU_SAFE_COMFY_FLAGS]
    launch_comfy_on_tpu(args.comfy_dir, remaining)


if __name__ == "__main__":
    main()
