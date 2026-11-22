"""Phase 0 live verification and benchmark harness for MiniMax-H3-Image GGUF on Kaggle T4 x2."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import struct
import subprocess
import threading
import time
import urllib.request
import uuid
import zlib
from pathlib import Path
from typing import Any

from minimax_kaggle.bootstrap import (
    collect_system_diagnostics,
    ensure_models_and_templates,
    setup_comfy_and_venv,
)
from minimax_kaggle.constants import (
    DEFAULT_COMFY_DIR,
    DEFAULT_INPUT_DIR,
    DEFAULT_VENV_DIR,
    DEFAULT_WORKING_DIR,
    GGUF_PRESETS,
    PRESETS,
    T4_SAFE_COMFY_FLAGS,
)
from minimax_kaggle.workflows import GenerationRequest, build_minimax_h3_api_prompt


def _make_synthetic_png(path: Path, width: int = 512, height: int = 512) -> None:
    """Write a crisp RGB test PNG without requiring PIL in the outer Python interpreter."""
    rows = bytearray()
    cx, cy = width // 2, height // 2
    r2 = (min(width, height) // 4) ** 2
    for y in range(height):
        rows.append(0)
        for x in range(width):
            dx, dy = x - cx, y - cy
            if dx * dx + dy * dy <= r2:
                rows.extend((40, 120, 235))
            elif (x // 32 + y // 32) % 2 == 0:
                rows.extend((235, 235, 235))
            else:
                rows.extend((205, 210, 215))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(rows), level=6)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def _http_json(url: str, *, payload: dict[str, Any] | None = None, timeout: int = 30) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8").strip()
        return json.loads(raw) if raw else {}


def _upload_image(base_url: str, image_path: Path) -> str:
    boundary = f"----MinimaxKaggleBoundary{uuid.uuid4().hex}"
    img_bytes = image_path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode("utf-8") + img_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    return str(res["name"])


def _analyze_png_via_venv(py_exe: Path, png_path: Path) -> dict[str, Any]:
    code = (
        "import json, numpy as np; from PIL import Image; "
        f"arr = np.asarray(Image.open({str(png_path)!r}).convert('RGB'), dtype=np.float32); "
        "stats = {"
        "'width': int(arr.shape[1]), 'height': int(arr.shape[0]), "
        "'min': float(arr.min()), 'max': float(arr.max()), "
        "'mean': round(float(arr.mean()), 2), 'std': round(float(arr.std()), 2), "
        "'is_black': bool(arr.max() < 2.0), 'is_flat': bool(arr.std() < 1.0)}; "
        "print(json.dumps(stats))"
    )
    proc = subprocess.run([str(py_exe), "-c", code], capture_output=True, text=True, check=True)
    return json.loads(proc.stdout.strip())


class ResourceMonitor:
    """Sample system RAM and NVIDIA GPU memory usage in the background."""

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_ram_used_gb = 0.0
        self.peak_cgroup_ram_gb = 0.0
        self.peak_gpu_vram_mb: dict[int, float] = {}

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _sample_once(self) -> None:
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            vals: dict[str, int] = {}
            for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split(":", 1)
                if len(parts) == 2:
                    toks = parts[1].strip().split()
                    if toks and toks[0].isdigit():
                        vals[parts[0].strip()] = int(toks[0])
            if "MemTotal" in vals and "MemAvailable" in vals:
                used_gb = (vals["MemTotal"] - vals["MemAvailable"]) / (1024 * 1024)
                if used_gb > self.peak_ram_used_gb:
                    self.peak_ram_used_gb = used_gb

        for cg_path in (
            Path("/sys/fs/cgroup/memory.current"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ):
            if cg_path.exists():
                try:
                    raw = cg_path.read_text(encoding="utf-8", errors="replace").strip()
                    if raw.isdigit():
                        cg_gb = int(raw) / (1024**3)
                        if cg_gb > self.peak_cgroup_ram_gb:
                            self.peak_cgroup_ram_gb = cg_gb
                        break
                except OSError:
                    pass

        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            proc = subprocess.run(
                [nvidia_smi, "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                for line in proc.stdout.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 2 and parts[0].isdigit():
                        idx = int(parts[0])
                        mb = float(parts[1])
                        if mb > self.peak_gpu_vram_mb.get(idx, 0.0):
                            self.peak_gpu_vram_mb[idx] = mb

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_s)


def _start_comfy(
    py_exe: Path,
    comfy_dir: Path,
    *,
    gpu_index: int,
    port: int,
    extra_flags: tuple[str, ...] = (),
    log_path: Path,
) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        str(py_exe),
        "main.py",
        "--listen",
        "127.0.0.1",
        "--port",
        str(port),
        *T4_SAFE_COMFY_FLAGS,
        *extra_flags,
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    print(f"[COMFY:{gpu_index}] Starting on port {port}: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(comfy_dir),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proc._minimax_log_file = log_file  # type: ignore[attr-defined]
    return proc


def _stop_comfy(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    log_file = getattr(proc, "_minimax_log_file", None)
    if log_file is not None:
        log_file.close()


def _wait_for_comfy(base_url: str, proc: subprocess.Popen[str], timeout_s: int = 180) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_err = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"ComfyUI exited early with code {proc.returncode}")
        try:
            return _http_json(f"{base_url}/system_stats", timeout=5)
        except Exception as exc:
            last_err = str(exc)
            time.sleep(1.5)
    raise TimeoutError(f"ComfyUI did not respond at {base_url} within {timeout_s}s ({last_err})")


def _run_prompt_and_collect(
    base_url: str,
    py_exe: Path,
    comfy_dir: Path,
    out_dir: Path,
    case_name: str,
    prompt_graph: dict[str, Any],
    timeout_s: int = 1200,
) -> dict[str, Any]:
    t0 = time.monotonic()
    resp = _http_json(
        f"{base_url}/prompt",
        payload={"prompt": prompt_graph, "client_id": f"minimax-smoke-{uuid.uuid4().hex[:8]}"},
        timeout=30,
    )
    prompt_id = resp["prompt_id"]
    deadline = time.monotonic() + timeout_s
    entry: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        hist = _http_json(f"{base_url}/history/{prompt_id}", timeout=15)
        if prompt_id in hist:
            entry = hist[prompt_id]
            break
        time.sleep(1.0)

    if entry is None:
        raise TimeoutError(f"Case {case_name} timed out after {timeout_s}s (prompt_id={prompt_id})")

    status = entry.get("status", {})
    if status.get("status_str") == "error":
        raise RuntimeError(f"Case {case_name} failed in ComfyUI: {json.dumps(status, ensure_ascii=False)}")

    elapsed = round(time.monotonic() - t0, 2)
    outputs = entry.get("outputs", {})
    saved_files: list[Path] = []
    for node_out in outputs.values():
        for img in node_out.get("images", []):
            src = comfy_dir / "output" / img.get("subfolder", "") / img["filename"]
            dst = out_dir / f"{case_name}_{img['filename']}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            saved_files.append(dst)

    if not saved_files:
        raise RuntimeError(f"Case {case_name} produced no output images.")

    stats = _analyze_png_via_venv(py_exe, saved_files[0])
    if stats["is_black"] or stats["is_flat"]:
        raise RuntimeError(f"Case {case_name} produced degenerate image: {stats}")

    return {
        "case": case_name,
        "status": "PASS",
        "elapsed_s": elapsed,
        "output_file": str(saved_files[0]),
        "image_stats": stats,
    }


def _tail_text(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    all_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(all_lines[-lines:])


def _preset_dit_and_clip_keys(preset: str) -> tuple[str, str]:
    keys = PRESETS[preset]
    dit_key = next(k for k in keys if k.startswith("dit_"))
    clip_key = next(k for k in keys if k.startswith("clip_") and "mmproj" not in k)
    return dit_key, clip_key


def run_smoke_suite(args: argparse.Namespace) -> dict[str, Any]:
    """Run Phase 0 diagnostics, GGUF setup, and live MiniMax-H3-Image T2I + Edit checks on Kaggle T4."""
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    diag = collect_system_diagnostics(working_dir=DEFAULT_WORKING_DIR, input_dir=args.input_dir)
    print("[DIAG]", json.dumps(diag, indent=2), flush=True)

    preset = args.preset
    if preset not in GGUF_PRESETS:
        raise ValueError(f"T4 smoke test expects a GGUF preset from {sorted(GGUF_PRESETS)}, got {preset!r}.")
    dit_key, clip_key = _preset_dit_and_clip_keys(preset)

    py_exe = setup_comfy_and_venv(
        args.comfy_dir,
        args.venv_dir,
        install_xformers=args.xformers,
        install_gguf_node=True,
        accelerator="gpu",
        input_root=args.input_dir,
    )
    model_sources = ensure_models_and_templates(
        preset=preset,
        comfy_dir=args.comfy_dir,
        venv_dir=args.venv_dir,
        input_root=args.input_dir,
        force_hf=args.force_hf,
    )

    extra_flags: list[str] = []
    if args.pytorch_sdpa:
        extra_flags.append("--use-pytorch-cross-attention")

    monitor = ResourceMonitor()
    monitor.start()

    log0 = out_dir / "comfy_gpu0.log"
    proc0 = _start_comfy(
        py_exe,
        args.comfy_dir,
        gpu_index=0,
        port=8188,
        extra_flags=tuple(extra_flags),
        log_path=log0,
    )
    base_url0 = "http://127.0.0.1:8188"
    results: list[dict[str, Any]] = []
    system_stats0: dict[str, Any] = {}
    required_nodes_ok = False

    try:
        system_stats0 = _wait_for_comfy(base_url0, proc0, timeout_s=240)
        obj_info = _http_json(f"{base_url0}/object_info", timeout=30)
        required_nodes = [
            "UnetLoaderGGUF",
            "CLIPLoaderGGUF",
            "VAELoader",
            "MiniMaxH3ImageConditioning",
            "ConditioningZeroOut",
            "KSampler",
            "VAEDecode",
            "SaveImage",
        ]
        missing = [n for n in required_nodes if n not in obj_info]
        if missing:
            raise RuntimeError(f"Missing required ComfyUI/GGUF nodes: {missing}")
        required_nodes_ok = True

        cases: list[tuple[str, GenerationRequest]] = [
            (
                "minimax_gguf_t2i_1024_short",
                GenerationRequest(
                    prompt=(
                        "A red glazed ceramic coffee mug resting on a dark oak desk beside a brass lamp, "
                        "soft directional studio light from the left, crisp sharp focus."
                    ),
                    mode="t2i",
                    profile="fast",
                    width=1024,
                    height=1024,
                    seed=1001,
                    dit_key=dit_key,
                    clip_key=clip_key,
                    filename_prefix="mm_smoke_t2i_short",
                ),
            ),
            (
                "minimax_gguf_t2i_1344x768_wide",
                GenerationRequest(
                    prompt=(
                        "Cinematic wide-angle architectural photograph of a vintage electric tram crossing "
                        "a rain-slicked cobblestone stone bridge at blue hour, warm interior cabin glow "
                        "reflecting across wet granite pavers, crisp detail."
                    ),
                    mode="t2i",
                    profile="fast",
                    width=1344,
                    height=768,
                    seed=1002,
                    dit_key=dit_key,
                    clip_key=clip_key,
                    filename_prefix="mm_smoke_t2i_wide",
                ),
            ),
        ]

        # Synthetic reference image for Edit smoke test (validates mmproj vision projector + <Picture 1>)
        synth_img = out_dir / "smoke_edit_input.png"
        _make_synthetic_png(synth_img)
        uploaded_name = _upload_image(base_url0, synth_img)
        cases.append(
            (
                "minimax_gguf_edit_1024",
                GenerationRequest(
                    prompt=(
                        "Replace the blue circle in the center of <Picture 1> with a brushed gold metallic "
                        "sphere catching soft studio reflections while keeping the grey checkerboard "
                        "background unchanged."
                    ),
                    mode="edit",
                    profile="fast",
                    width=1024,
                    height=1024,
                    image_names=(uploaded_name,),
                    use_image_latent=False,
                    seed=1003,
                    dit_key=dit_key,
                    clip_key=clip_key,
                    filename_prefix="mm_smoke_edit",
                ),
            )
        )

        if args.test_2k:
            cases.append(
                (
                    "minimax_gguf_t2i_2048",
                    GenerationRequest(
                        prompt=(
                            "Editorial studio product photograph of a matte anodized aluminum mechanical "
                            "keyboard on a warm walnut tabletop, crisp sculpted keycap legends, clean "
                            "softbox highlights and subtle shadow falloff."
                        ),
                        mode="t2i",
                        profile="quality",
                        width=2048,
                        height=2048,
                        seed=1004,
                        dit_key=dit_key,
                        clip_key=clip_key,
                        filename_prefix="mm_smoke_2048",
                    ),
                )
            )

        for name, req in cases:
            print(f"[RUN] {name} ...", flush=True)
            res = _run_prompt_and_collect(
                base_url0,
                py_exe,
                args.comfy_dir,
                out_dir,
                name,
                build_minimax_h3_api_prompt(req),
            )
            print(f"[PASS] {name}: {res['elapsed_s']}s | stats={res['image_stats']}", flush=True)
            results.append(res)

        gpu_list = diag.get("nvidia_smi") or []
        dual_concurrent_wall_s: float | None = None
        if getattr(args, "test_dual_concurrent", False) and len(gpu_list) >= 2:
            log1 = out_dir / "comfy_gpu1.log"
            proc1 = _start_comfy(
                py_exe,
                args.comfy_dir,
                gpu_index=1,
                port=8189,
                extra_flags=tuple(extra_flags),
                log_path=log1,
            )
            base_url1 = "http://127.0.0.1:8189"
            try:
                _wait_for_comfy(base_url1, proc1, timeout_s=240)
                print("[RUN] dual_concurrent_gpu0_and_gpu1 (2x T4 parallel) ...", flush=True)
                t_dual = time.monotonic()
                dual_jobs = [
                    (
                        base_url0,
                        "dual_concurrent_gpu0_t2i",
                        GenerationRequest(
                            prompt="A hand-blown emerald glass vase on a travertine plinth in morning sunbeams, photorealistic.",
                            mode="t2i",
                            profile="fast",
                            width=1024,
                            height=1024,
                            seed=3001,
                            dit_key=dit_key,
                            clip_key=clip_key,
                            filename_prefix="mm_smoke_dual_gpu0",
                        ),
                    ),
                    (
                        base_url1,
                        "dual_concurrent_gpu1_t2i",
                        GenerationRequest(
                            prompt="A sleek brushed-silver fountain pen resting on cream cotton paper, natural window light.",
                            mode="t2i",
                            profile="fast",
                            width=1024,
                            height=1024,
                            seed=3002,
                            dit_key=dit_key,
                            clip_key=clip_key,
                            filename_prefix="mm_smoke_dual_gpu1",
                        ),
                    ),
                ]
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    futs = [
                        pool.submit(
                            _run_prompt_and_collect,
                            url,
                            py_exe,
                            args.comfy_dir,
                            out_dir,
                            case_name,
                            build_minimax_h3_api_prompt(req),
                        )
                        for url, case_name, req in dual_jobs
                    ]
                    dual_results = [f.result() for f in futs]
                dual_concurrent_wall_s = round(time.monotonic() - t_dual, 2)
                for res_d in dual_results:
                    print(
                        f"[PASS] {res_d['case']}: {res_d['elapsed_s']}s (wall={dual_concurrent_wall_s}s) | stats={res_d['image_stats']}",
                        flush=True,
                    )
                    results.append(res_d)
                _http_json(f"{base_url1}/free", payload={"unload_models": True, "free_memory": True}, timeout=20)
            except Exception as exc:
                print(f"[FAIL] Dual-GPU concurrent test error: {exc}", flush=True)
                print("[LOG TAIL GPU1]\n" + _tail_text(log1), flush=True)
                raise
            finally:
                _stop_comfy(proc1)

        _http_json(f"{base_url0}/free", payload={"unload_models": True, "free_memory": True}, timeout=20)
    except Exception as exc:
        print(f"[FAIL] GPU0 smoke test error: {exc}", flush=True)
        print("[LOG TAIL GPU0]\n" + _tail_text(log0), flush=True)
        raise
    finally:
        _stop_comfy(proc0)

    # Sequential isolated check on GPU 1 (only when dual-concurrent was not enabled)
    gpu_list = diag.get("nvidia_smi") or []
    if not getattr(args, "test_dual_concurrent", False) and args.test_gpu1_isolated and len(gpu_list) >= 2:
        log1 = out_dir / "comfy_gpu1.log"
        proc1 = _start_comfy(
            py_exe,
            args.comfy_dir,
            gpu_index=1,
            port=8189,
            extra_flags=tuple(extra_flags),
            log_path=log1,
        )
        base_url1 = "http://127.0.0.1:8189"
        try:
            _wait_for_comfy(base_url1, proc1, timeout_s=240)
            res1 = _run_prompt_and_collect(
                base_url1,
                py_exe,
                args.comfy_dir,
                out_dir,
                "gpu1_isolated_minimax_t2i",
                build_minimax_h3_api_prompt(
                    GenerationRequest(
                        prompt="A sleek brushed-silver fountain pen resting on cream cotton paper, natural window light.",
                        mode="t2i",
                        profile="fast",
                        width=1024,
                        height=1024,
                        seed=2001,
                        dit_key=dit_key,
                        clip_key=clip_key,
                        filename_prefix="mm_smoke_gpu1",
                    )
                ),
            )
            print(f"[PASS] gpu1_isolated_minimax_t2i: {res1['elapsed_s']}s | stats={res1['image_stats']}", flush=True)
            results.append(res1)
            _http_json(f"{base_url1}/free", payload={"unload_models": True, "free_memory": True}, timeout=20)
        except Exception as exc:
            print(f"[FAIL] GPU1 isolated test error: {exc}", flush=True)
            print("[LOG TAIL GPU1]\n" + _tail_text(log1), flush=True)
            raise
        finally:
            _stop_comfy(proc1)

    monitor.stop()

    report = {
        "diagnostics": diag,
        "preset": preset,
        "model_sources": model_sources,
        "extra_flags": extra_flags,
        "required_nodes_ok": required_nodes_ok,
        "comfy_system_stats": system_stats0.get("system"),
        "peak_ram_used_gb": round(monitor.peak_ram_used_gb, 2),
        "peak_cgroup_ram_gb": round(monitor.peak_cgroup_ram_gb, 2),
        "peak_gpu_vram_mb": monitor.peak_gpu_vram_mb,
        "dual_concurrent_wall_s": dual_concurrent_wall_s,
        "results": results,
    }
    report_path = out_dir / "minimax_smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[DONE] Smoke report saved to {report_path}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 smoke test for MiniMax-H3-Image GGUF on Kaggle T4 x2.")
    parser.add_argument("--preset", default="t4_gguf", choices=sorted(GGUF_PRESETS))
    parser.add_argument("--comfy-dir", type=Path, default=DEFAULT_COMFY_DIR)
    parser.add_argument("--venv-dir", type=Path, default=DEFAULT_VENV_DIR)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_WORKING_DIR / "minimax_smoke_results")
    parser.add_argument("--force-hf", action="store_true")
    parser.add_argument("--xformers", action="store_true")
    parser.add_argument("--pytorch-sdpa", action="store_true")
    parser.add_argument("--test-2k", action="store_true")
    parser.add_argument("--skip-gpu1", dest="test_gpu1_isolated", action="store_false")
    parser.add_argument("--test-dual-concurrent", dest="test_dual_concurrent", action="store_true")
    parser.add_argument("--skip-dual-concurrent", dest="test_dual_concurrent", action="store_false")
    parser.set_defaults(test_gpu1_isolated=True, test_dual_concurrent=True)
    args = parser.parse_args()
    run_smoke_suite(args)


if __name__ == "__main__":
    main()
