"""ROCm environment probe and ComfyUI API latency/VRAM benchmark helper for AMD Instinct GPUs."""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def probe_rocm_environment() -> dict:
    """Inspect PyTorch HIP/ROCm availability, GPU architecture, and comfy-kitchen backend status."""
    report: dict = {
        "python": sys.version.split()[0],
        "torch_installed": False,
        "hip_version": None,
        "cuda_api_available": False,
        "devices": [],
        "triton_version": None,
        "comfy_kitchen_installed": False,
    }
    try:
        import torch  # type: ignore[import-not-found]

        report["torch_installed"] = True
        report["torch_version"] = torch.__version__
        report["hip_version"] = getattr(torch.version, "hip", None)
        report["cuda_api_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                report["devices"].append(
                    {
                        "index": idx,
                        "name": props.name,
                        "gcn_arch_name": getattr(props, "gcnArchName", None),
                        "total_vram_gib": round(props.total_memory / (1024**3), 2),
                    }
                )
    except ImportError:
        pass

    try:
        import triton  # type: ignore[import-not-found]

        report["triton_version"] = getattr(triton, "__version__", "unknown")
    except ImportError:
        pass

    try:
        import importlib.util

        report["comfy_kitchen_installed"] = importlib.util.find_spec("comfy_kitchen") is not None
    except Exception:
        pass

    return report


def fetch_comfy_system_stats(comfy_url: str, timeout_s: float = 5.0) -> dict:
    """Query ComfyUI /system_stats to verify active device VRAM and runtime flags."""
    base = comfy_url.rstrip("/")
    with urllib.request.urlopen(f"{base}/system_stats", timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_prompt_benchmark(comfy_url: str, prompt_json_path: Path, timeout_s: float = 300.0) -> dict:
    """Submit a ComfyUI API prompt JSON and measure wall-clock completion latency."""
    base = comfy_url.rstrip("/")
    prompt_payload = json.loads(prompt_json_path.read_text(encoding="utf-8"))
    if "prompt" not in prompt_payload:
        prompt_payload = {"prompt": prompt_payload}

    body = json.dumps(prompt_payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/prompt",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as resp:
        queued = json.loads(resp.read().decode("utf-8"))
    prompt_id = queued["prompt_id"]

    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        with urllib.request.urlopen(f"{base}/history/{prompt_id}", timeout=10) as resp:
            history = json.loads(resp.read().decode("utf-8"))
        if prompt_id in history:
            elapsed = time.perf_counter() - started
            entry = history[prompt_id]
            return {
                "prompt_id": prompt_id,
                "elapsed_seconds": round(elapsed, 3),
                "status": entry.get("status", {}),
            }
        time.sleep(0.5)

    raise TimeoutError(f"Prompt {prompt_id} did not finish within {timeout_s}s.")


def main() -> None:
    parser = argparse.ArgumentParser(description="AMD ROCm & ComfyUI Qwen-Image-2.1 benchmark helper.")
    parser.add_argument("--check-env", action="store_true", help="Print local PyTorch/ROCm/Triton diagnostics.")
    parser.add_argument("--comfy-url", default=None, help="ComfyUI base URL, e.g. http://127.0.0.1:8188")
    parser.add_argument("--prompt-json", type=Path, default=None, help="Optional ComfyUI API prompt JSON to time.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Benchmark timeout in seconds.")
    args = parser.parse_args()

    output: dict = {}
    if args.check_env or (args.comfy_url is None and args.prompt_json is None):
        output["environment"] = probe_rocm_environment()
    if args.comfy_url:
        try:
            output["comfy_system_stats"] = fetch_comfy_system_stats(args.comfy_url)
            if args.prompt_json is not None:
                output["benchmark"] = run_prompt_benchmark(args.comfy_url, args.prompt_json, timeout_s=args.timeout)
        except urllib.error.URLError as exc:
            output["comfy_error"] = str(exc)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
