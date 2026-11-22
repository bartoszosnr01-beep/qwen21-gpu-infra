"""TPU v5e-8 (8x16 GiB = 128 GiB HBM) PyTorch/XLA SPMD runtime & static-shape bridge for ComfyUI."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn

from qwen_kaggle.constants import DEFAULT_COMFY_DIR, TPU_CONTEXT_BUCKETS, TPU_SAFE_COMFY_FLAGS

TPU_TOTAL_HBM_BYTES = 128 * (1024**3)  # 8 x 16 GiB HBM on Kaggle TPU v5e-8
TPU_TOTAL_HBM_MB = 131072.0
MIN_SHARD_NUMEL = 65536


def is_tpu_environment() -> bool:
    """Detect whether the current container has Google Cloud TPU v5e devices."""
    if os.environ.get("QWEN_FORCE_TPU") == "1":
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


def pad_context_tensor(
    context: torch.Tensor,
    buckets: tuple[int, ...] = TPU_CONTEXT_BUCKETS,
) -> tuple[torch.Tensor, int, int]:
    """Right-pad `[B, L_real, D]` context tensor with zeros to `[B, L_bucket, D]`.

    Returns `(padded_context, real_len, bucket_len)`.
    """
    real_len = int(context.shape[1])
    bucket_len = choose_static_context_bucket(real_len, buckets=buckets)
    pad_len = bucket_len - real_len
    if pad_len <= 0:
        return context, real_len, real_len
    pad = torch.zeros(
        (context.shape[0], pad_len, *context.shape[2:]),
        dtype=context.dtype,
        device=context.device,
    )
    return torch.cat([context, pad], dim=1), real_len, bucket_len


def build_static_padded_sequence(
    model: Any,
    x: torch.Tensor,
    context: torch.Tensor,
    ref_latents: list[torch.Tensor],
    image_slots: list[int],
    buckets: tuple[int, ...] = TPU_CONTEXT_BUCKETS,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int, torch.Tensor | None]]]:
    """Static-shape replacement for `QwenImage21Transformer2DModel.build_sequence`.

    Pads the trailing text segment up to `L_bucket` tokens so `hidden_states` and `pe`
    have fixed shapes across prompts, while masking out padded positions `(real_len .. L_bucket)`
    in subsequent image attention masks and keeping exact RoPE `pos = real_len` coordinates.
    """
    padded_context, real_len, bucket_len = pad_context_tensor(context, buckets=buckets)
    pad_len = bucket_len - real_len

    txt = model.txt_in(padded_context)
    slots = (image_slots + [real_len] * len(ref_latents))[: len(ref_latents)]
    # Final bound uses `bucket_len` so the trailing text chunk carries the static padding
    bounds = [0] + slots + [bucket_len]

    parts: list[torch.Tensor] = []
    ids: list[torch.Tensor] = []
    segments: list[tuple[int, int, torch.Tensor | None]] = []
    # Track padded vs real token positions separately so RoPE `pos` matches unpadded execution
    pos_real = 0
    length = 0
    padded_text_span: tuple[int, int] | None = None

    all_imgs = list(ref_latents) + [x]
    for seg_idx, ((start, end), img) in enumerate(zip(zip(bounds[:-1], bounds[1:]), all_imgs)):
        is_last_text_seg = seg_idx == len(all_imgs) - 1
        n_padded = end - start
        n_real = (n_padded - pad_len) if is_last_text_seg else n_padded

        if n_padded > 0:
            parts.append(txt[:, start:end])
            ids.append(
                torch.arange(pos_real, pos_real + n_padded, device=x.device, dtype=torch.float32)
                .unsqueeze(1)
                .expand(n_padded, 3)
            )
            causal_mask = torch.ones((n_padded, length + n_padded), dtype=torch.bool, device=x.device).tril(length)
            if padded_text_span is not None:
                p_start, p_end = padded_text_span
                if p_end > p_start:
                    causal_mask[:, p_start:p_end] = False
            segments.append((length, length + n_padded, causal_mask))
            if is_last_text_span := (is_last_text_seg and pad_len > 0):
                _ = is_last_text_span
                padded_text_span = (length + n_real, length + n_padded)
            pos_real += max(0, n_real)
            length += n_padded

        h, w = img.shape[-2:]
        img_tokens = h * w
        parts.append(model.img_in(img.flatten(2).transpose(1, 2)))
        hh = (
            torch.arange(h, device=x.device, dtype=torch.float32)
            - (h - h // 2)
            + 0.5 * (h % 2 - x.shape[-2] % 2)
        )
        ww = (
            torch.arange(w, device=x.device, dtype=torch.float32)
            - (w - w // 2)
            + 0.5 * (w % 2 - x.shape[-1] % 2)
        )
        ids.append(
            torch.stack(
                [
                    torch.full((h, w), float(pos_real), device=x.device, dtype=torch.float32),
                    hh[:, None].expand(h, w),
                    ww[None, :].expand(h, w),
                ],
                dim=-1,
            ).flatten(0, 1)
        )
        if padded_text_span is not None and padded_text_span[1] > padded_text_span[0]:
            img_mask = torch.ones((img_tokens, length + img_tokens), dtype=torch.bool, device=x.device)
            img_mask[:, padded_text_span[0] : padded_text_span[1]] = False
        else:
            img_mask = None
        segments.append((length, length + img_tokens, img_mask))
        pos_real += max(h, w)
        length += img_tokens

    pe = model.pe_embedder(torch.cat(ids, dim=0).unsqueeze(0)).transpose(1, 2).contiguous()
    return torch.cat(parts, dim=1), pe, segments


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


def shard_module_for_spmd(
    module: nn.Module,
    xla_device: torch.device,
    mesh: Any,
    num_chips: int,
    mark_sharding_fn: Any,
) -> dict[str, int]:
    """Move `module` weights to `xla_device` in BF16 and annotate 2D+ matrices for 8-chip SPMD."""
    dequantized = dequantize_module_weights_to_bf16(module)
    sharded = 0
    replicated = 0

    for sub in module.modules():
        cls_name = sub.__class__.__name__
        keep_fp32 = "RMSNorm" in cls_name or "LayerNorm" in cls_name
        for name, param in list(sub._parameters.items()):
            if param is None:
                continue
            target_dtype = (
                torch.float32
                if (keep_fp32 and param.dtype == torch.float32)
                else (torch.bfloat16 if param.is_floating_point() else param.dtype)
            )
            moved = param.detach().to(device=xla_device, dtype=target_dtype)
            spec = choose_sharding_spec(tuple(moved.shape), num_chips)
            if spec is not None and mesh is not None and mark_sharding_fn is not None:
                mark_sharding_fn(moved, mesh, spec)
                sharded += 1
            else:
                replicated += 1
            sub._parameters[name] = nn.Parameter(moved, requires_grad=False)

        for name, buf in list(sub._buffers.items()):
            if buf is None:
                continue
            target_dtype = torch.bfloat16 if buf.is_floating_point() and buf.dtype != torch.float32 else buf.dtype
            moved_buf = buf.detach().to(device=xla_device, dtype=target_dtype)
            spec = choose_sharding_spec(tuple(moved_buf.shape), num_chips)
            if spec is not None and mesh is not None and mark_sharding_fn is not None:
                mark_sharding_fn(moved_buf, mesh, spec)
                sharded += 1
            sub._buffers[name] = moved_buf

    return {"dequantized": dequantized, "sharded": sharded, "replicated": replicated}


class TpuXlaContext:
    """Holds initialized PyTorch/XLA SPMD device, 8-chip mesh, and compilation cache state."""

    def __init__(self, cache_dir: Path = Path("/tmp/qwen_xla_cache")) -> None:
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


def install_comfy_tpu_patches(ctx: TpuXlaContext) -> None:
    """Patch ComfyUI's model_management, comfy_kitchen ops, and QwenImage21 DiT for TPU v5e-8."""
    try:
        import comfy_kitchen as ck  # type: ignore[import-not-found]

        ck.registry.disable("cuda")
        ck.registry.disable("triton")
    except Exception:
        pass

    import comfy.ldm.qwen_image21.model as qwen21_mod  # type: ignore[import-not-found]
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

    # Pure-PyTorch XLA-traceable replacements for comfy_kitchen custom_ops inside QwenImage21
    def _xla_modulated_norm(norm: nn.Module, x: torch.Tensor, scale: tuple[torch.Tensor, torch.Tensor], prefix_len: int, _zero: torch.Tensor) -> torch.Tensor:
        s_prefix, s_target = scale
        out = norm(x)
        if prefix_len > 0:
            return torch.cat(
                [out[:, :prefix_len] * (1.0 + s_prefix), out[:, prefix_len:] * (1.0 + s_target)],
                dim=1,
            )
        return out * (1.0 + s_target)

    def _xla_last_layer_forward(self: Any, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale = self.linear(torch.nn.functional.silu(temb)).unsqueeze(1)
        return self.norm(x) * (1.0 + scale)

    def _xla_attention_forward(
        self: Any,
        x: torch.Tensor,
        pe: torch.Tensor,
        attn_fn: Any,
        prefix_len: int,
        transformer_options: dict[str, Any] = {},
    ) -> torch.Tensor:
        b, n, _ = x.shape
        q = self.to_q(x).view(b, n, self.heads, -1)
        k = self.to_k(x).view(b, n, self.heads, -1)
        v = self.to_v(x).view(b, n, self.heads, -1)
        q, k = self.norm_q(q), self.norm_k(k)
        q = qwen21_mod.apply_rope1(q, pe)
        k = qwen21_mod.apply_rope1(k, pe)
        q_c = qwen21_mod.AttentionTensorContainer(q)
        k_c = qwen21_mod.AttentionTensorContainer(k)
        v_c = qwen21_mod.AttentionTensorContainer(v)
        return self.to_out[0](attn_fn(q_c, k_c, v_c, self.heads, preferred_attention=self.comfy_attention))

    qwen21_mod._modulated_norm = _xla_modulated_norm
    qwen21_mod.LastLayer.forward = _xla_last_layer_forward
    qwen21_mod.Attention.forward = _xla_attention_forward

    _orig_dit_forward = qwen21_mod.QwenImage21Transformer2DModel._forward

    def _tpu_dit_forward(
        self: Any,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        ref_latents: list[torch.Tensor] | None = None,
        image_slots: list[int] | None = None,
        transformer_options: dict[str, Any] = {},
        prefix_cache: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        ctx.ensure_model_on_tpu(self)
        orig_device = x.device
        orig_dtype = x.dtype

        x_tpu = x.to(device=ctx.device, dtype=torch.bfloat16)
        t_tpu = timesteps.to(device=ctx.device, dtype=torch.bfloat16)
        c_tpu = context.to(device=ctx.device, dtype=torch.bfloat16)
        refs_tpu = [r.to(device=ctx.device, dtype=torch.bfloat16) for r in (ref_latents or [])]

        # Disable prefix cache splitting on TPU so all steps 1..N reuse one static-shape XLA graph
        t_opts = dict(transformer_options)
        t_opts["qwen_image21_cache"] = {"device": "off", "dtype": "default"}
        self.prefix_cache_enabled = False

        out_tpu = _orig_dit_forward(
            self,
            x_tpu,
            t_tpu,
            c_tpu,
            ref_latents=refs_tpu,
            image_slots=image_slots,
            transformer_options=t_opts,
            prefix_cache=None,
            **kwargs,
        )
        ctx.mark_step()
        return out_tpu.to(device=orig_device, dtype=orig_dtype)

    def _tpu_build_sequence(
        self: Any,
        x: torch.Tensor,
        context: torch.Tensor,
        ref_latents: list[torch.Tensor],
        image_slots: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int, torch.Tensor | None]]]:
        return build_static_padded_sequence(self, x, context, ref_latents, image_slots)

    qwen21_mod.QwenImage21Transformer2DModel.build_sequence = _tpu_build_sequence
    qwen21_mod.QwenImage21Transformer2DModel._forward = _tpu_dit_forward
    print("[TPU] Installed ComfyUI QwenImage21 static-shape & 8-chip SPMD patches.", flush=True)


def launch_comfy_on_tpu(comfy_dir: Path, comfy_args: list[str]) -> None:
    """Initialize TPU v5e-8 SPMD context, patch ComfyUI, and run ComfyUI `main.py`."""
    comfy_dir = comfy_dir.resolve()
    if str(comfy_dir) not in sys.path:
        sys.path.insert(0, str(comfy_dir))

    ctx = TpuXlaContext()
    ctx.initialize()

    # Run ComfyUI host orchestration with --cpu so dynamic tokenizers/Qwen3.5-9B stay on 330 GB host RAM
    # while QwenImage21Transformer2DModel executes sharded across all 8 TPU v5e chips in BF16.
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
    out_dir: Path = Path("/kaggle/working/tpu_smoke_results"),
    preset: str = "smoke",
) -> dict[str, Any]:
    """End-to-end TPU v5e-8 qualification test: bootstrap, shard to 8 chips, verify static-shape warmup."""
    import json
    import subprocess
    import time
    import urllib.request
    from qwen_kaggle.bootstrap import (
        collect_system_diagnostics,
        ensure_models_and_templates,
        ensure_physical_tpu_or_raise,
        setup_comfy_and_venv,
    )
    from qwen_kaggle.smoke_test import _run_prompt_and_collect, _wait_for_comfy
    from qwen_kaggle.workflows import GenerationRequest, build_qwen21_api_prompt

    out_dir.mkdir(parents=True, exist_ok=True)
    diag = collect_system_diagnostics(input_dir=input_dir)
    print("[TPU-SMOKE] Diagnostics:", json.dumps(diag, indent=2), flush=True)

    sources: dict[str, str] = {}
    results: list[dict[str, Any]] = []
    proc: subprocess.Popen[str] | None = None
    log_path = out_dir / "comfy_tpu0.log"
    log_fh = None

    try:
        ensure_physical_tpu_or_raise(diag)
        py_exe = setup_comfy_and_venv(comfy_dir, venv_dir, accelerator="tpu")
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
            "qwen_kaggle.tpu_runtime",
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
                "tpu_cold_compile_1024",
                GenerationRequest(
                    prompt="A sleek obsidian glass sphere on a dark graphite pedestal, studio rim light",
                    mode="t2i",
                    profile="fast",
                    width=1024,
                    height=1024,
                    seed=101,
                    cache_device="off",
                    filename_prefix="tpu_cold",
                ),
            ),
            (
                "tpu_warm_cached_diff_len_1024",
                GenerationRequest(
                    prompt=(
                        "Photorealistic architectural photograph of a minimalist concrete and warm cedar "
                        "pavilion beside a calm mountain lake at dusk, crisp reflections, 35mm lens, "
                        "natural atmospheric haze, balanced shadows"
                    ),
                    mode="t2i",
                    profile="fast",
                    width=1024,
                    height=1024,
                    seed=202,
                    cache_device="off",
                    filename_prefix="tpu_warm",
                ),
            ),
        ]

        for name, req in cases:
            print(f"[TPU-SMOKE] Running {name} (prompt_chars={len(req.prompt)})...", flush=True)
            graph = build_qwen21_api_prompt(req)
            res = _run_prompt_and_collect(
                base_url,
                py_exe,
                comfy_dir,
                out_dir,
                name,
                graph,
                timeout_s=1800,
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
    parser = argparse.ArgumentParser(description="ComfyUI TPU v5e-8 SPMD Worker Launcher")
    parser.add_argument("--comfy-dir", type=Path, default=DEFAULT_COMFY_DIR)
    args, remaining = parser.parse_known_args()
    if not remaining:
        remaining = ["--listen", "127.0.0.1", "--port", "8188", *TPU_SAFE_COMFY_FLAGS]
    launch_comfy_on_tpu(args.comfy_dir, remaining)


if __name__ == "__main__":
    main()
