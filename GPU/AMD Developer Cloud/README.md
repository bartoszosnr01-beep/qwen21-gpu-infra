# AMD Developer Cloud — MI300X ROCm Evaluation (`full-bf16`)

Reproducible setup and diagnostics for running **ComfyUI + `Comfy-Org/Qwen-Image-2.1`** on **AMD Instinct™ MI300X (192 GB HBM3)** using the **ROCm** PyTorch stack.

## Why MI300X + `full-bf16`
The full Qwen-Image-2.1 pipeline combines a 7B DiT backbone (`14.2 GB` in BF16), an 8B Qwen3-VL vision-language encoder (`17.53 GB` in BF16), and a BF16 VAE (`0.68 GB`) — **~32.4 GB of raw weights** before 2K activation buffers and prompt-enhancer stages. Consumer 16–24 GB GPUs must quantize (`int8_convrot` / `w4a8`) or offload to system RAM. On MI300X (`192 GB` HBM3), we can evaluate:
1. **Unquantized `full-bf16`** inference quality and step latency without CPU offloading,
2. **`comfy-kitchen` / Triton INT8 ConvRot** compatibility and throughput on `gfx942` / ROCm vs native BF16.

> **Billing Safety Reminder:** Stopping/powering off an instance does **not** release provisioned compute resources. Always **Delete / Terminate** the instance after exporting results.

---

## Setup & Launch (from Windows PowerShell)

1. Copy the shared manifest/workflow directory and setup/benchmark scripts to the MI300X VM:
   ```powershell
   scp -r ".\GPU\Wagi\qwen_image_2_1" <user>@<VM_IP>:~/qwen21_shared
   scp ".\GPU\AMD Developer Cloud\setup_amd_bf16.sh" ".\GPU\AMD Developer Cloud\benchmark_rocm_bf16.py" <user>@<VM_IP>:~/
   ```
2. Verify ROCm/Triton environment and launch ComfyUI bound to localhost:
   ```powershell
   ssh <user>@<VM_IP> "python3 ~/benchmark_rocm_bf16.py --check-env && sed -i 's/\r$//' ~/setup_amd_bf16.sh && bash ~/setup_amd_bf16.sh"
   ```
   *(Override `ROCM_INDEX_URL=https://download.pytorch.org/whl/rocm7.0` if needed when preinstalled ROCm torch is absent.)*
3. Forward port `8188` over an SSH tunnel in a second PowerShell window:
   ```powershell
   ssh -N -L 8188:127.0.0.1:8188 <user>@<VM_IP>
   ```
   Open **`http://127.0.0.1:8188`** in your browser (or via Bartosz OS `AMD SSH (8188)` preset).
4. Optional API latency check while ComfyUI is running:
   ```bash
   python3 ~/benchmark_rocm_bf16.py --comfy-url http://127.0.0.1:8188 --prompt-json <api_prompt.json>
   ```
5. Export results and **Delete / Terminate** the instance in AMD Developer Cloud when done.
