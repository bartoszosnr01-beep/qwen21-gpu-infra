# Qwen-Image-2.1 Multi-Cloud GPU Infrastructure (`ComfyUI`)

Reproducible infrastructure, shared model manifest, and cross-platform runners for deploying **ComfyUI** with **`Comfy-Org/Qwen-Image-2.1`** (multimodal image generation and editing with Qwen3-VL 8B and Qwen3.5-9B Prompt Enhancer) across ROCm bare-metal and CUDA cloud runtimes.

## Repository Architecture

All environments share a single canonical definition in [`GPU/Wagi/qwen_image_2_1`](GPU/Wagi/qwen_image_2_1) (`models.json`, `common.py`, and `workflows/`) to avoid duplicated weight lists, mismatched loader nodes, or redundant local installations:

| Directory | Target Environment | Weight Profile | Notes |
| :--- | :--- | :--- | :--- |
| [`GPU/Wagi/qwen_image_2_1`](GPU/Wagi/qwen_image_2_1) | Shared Core | `full-int8`, `lite-int8`, `turbo-lite-int8`, `w4a8`, `full-bf16` | Canonical HF manifest, atomic downloader, and live-preview workflow builder + profile adapter |
| [`GPU/AMD Developer Cloud`](GPU/AMD%20Developer%20Cloud) | **AMD Instinct™ MI300X (ROCm)** | `full-bf16` (`~32.4 GB`) | Native unquantized BF16 setup preserving ROCm PyTorch wheels + [`benchmark_rocm_bf16.py`](GPU/AMD%20Developer%20Cloud/benchmark_rocm_bf16.py) diagnostic/benchmark helper |
| [`GPU/Modal`](GPU/Modal) | **Modal (`L4` / `A10`, 24 GB)** | `full-int8` (`~34.0 GB`) | Serverless `@modal.web_server`, persistent `1 TiB` Volume, `24 GiB` RAM, `180 s` scaledown, WS + `/userdata` proxy fixes |
| [`GPU/Kaggle`](GPU/Kaggle) | **Kaggle (`2× Tesla T4`, 16 GB)** | `lite-int8` (`~17.3 GB`) | Installs into `/tmp/ComfyUI` to stay under the `/kaggle/working` 20 GB cap, `--cache-none`, `cloudflared` tunnel |
| [`GPU/ModelScope`](GPU/ModelScope) | **ModelScope PAI-DSW (`A10`, 24 GB)** | `full-int8` (`~34.0 GB`) | Uses `/mnt/workspace` persistent disk and `HF_ENDPOINT=https://hf-mirror.com` |
| [`GPU/Beam.cloud`](GPU/Beam.cloud) | **Beam.cloud (`beam.Service`)** | `full-int8` / `lite-int8` | Scale-to-zero `beam.Service` template *(note: Developer plan requires `$20/mo` or prepaid credits as of Nov 2026)* |

---

## Key Engineering Highlights

* **ROCm BF16 vs. INT8 ConvRot Evaluation (`MI300X`):** Strips `torch*` entries from ComfyUI `requirements.txt` during install so `pip` does not overwrite preinstalled ROCm/HIP PyTorch builds with CUDA wheels, and adapts workflow loader nodes cleanly to `full-bf16` (`qwen_image_2.1_bf16.safetensors` + `qwen3vl_8b_bf16.safetensors`) to benchmark native BF16 inference alongside `comfy-kitchen` / Triton diagnostics.
* **Shared Workflow Live-Preview Builder:** Automatically splits `KSampler` + `VAEDecode` onto the root ComfyUI canvas (`taeqi2_1_decoder.pth` TAESD live previews) while keeping the image-edit subgraph compact.
* **Reverse-Proxy & Memory Hardening:** Includes `sitecustomize.py` runtime patches disabling WebSocket `permessage-deflate` mismatches and normalizing URL-encoded `%2F` paths on `/api/userdata/*`.

---

## Quick Usage (Windows PowerShell)

```powershell
# Modal deployment (L4/A10, 24 GiB RAM, 180s idle timeout)
.\.venv\Scripts\modal.exe deploy .\GPU\Modal\comfy_qwen21.py

# Shared CLI downloader / workflow profile adapter
.\.venv\Scripts\python.exe .\GPU\Wagi\qwen_image_2_1\common.py --profile full-bf16 --model-dir .\models --workflow-dir .\workflows
```

See platform-specific instructions inside:
* [AMD Developer Cloud (MI300X ROCm)](GPU/AMD%20Developer%20Cloud/README.md)
* [Kaggle (T4 × 2)](GPU/Kaggle/README.md)
* [ModelScope PAI-DSW (A10)](GPU/ModelScope/README.md)
* [Beam.cloud](GPU/Beam.cloud/README.md)

## License

MIT — see [LICENSE](LICENSE).
