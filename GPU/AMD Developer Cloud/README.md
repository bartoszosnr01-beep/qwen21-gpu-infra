# AMD Developer Cloud (MI300X 192 GB VRAM, ROCm + BF16)

Środowisko pod darmowe **100 USD kredytów (ważne 30 dni)** z programu **AMD AI Developer Program**.

## Dlaczego profil `full-bf16`?
Na MI300X mamy **192 GB VRAM HBM3**, więc zamiast walczyć z kernelami `int8_convrot` (`comfy-kitchen` / Triton pod ROCm), pobieramy z `Comfy-Org/Qwen-Image-2.1` **pełne wagi BF16 (~32,4 GB)** i automatycznie podmieniamy nazwy plików w workflowach przez [common.py](file:///g:/Projekty/Infrastruktura/GPU/Wagi/qwen_image_2_1/common.py).

> **KRYTYCZNA ZASADA BEZPIECZEŃSTWA (ROZLICZENIA):**
> Zatrzymanie VM (`Stop / Power Off`) **NIE przerywa naliczania kredytów**. Po każdej sesji musisz całkowicie **usunąć/zniszczyć instancję (`Delete / Terminate`)**.

---

## Kroki uruchomienia (Windows PowerShell)

1. **Złóż wniosek do AMD AI Developer Program** (profil GitHub/LinkedIn + opis projektu Qwen-Image-2.1; weryfikacja trwa 2–3 dni robocze) i utwórz instancję **MI300X** z kluczem SSH.
2. **Wyślij na VM wspólny katalog wag/workflowów i skrypt instalacyjny:**
   ```powershell
   scp -r ".\GPU\Wagi\qwen_image_2_1" <user>@<VM_IP>:~/qwen21_shared
   scp ".\GPU\AMD Developer Cloud\setup_amd_bf16.sh" <user>@<VM_IP>:~/setup_amd_bf16.sh
   ```
3. **Zaloguj się przez SSH i uruchom setup + ComfyUI:**
   ```powershell
   ssh <user>@<VM_IP> "sed -i 's/\r$//' ~/setup_amd_bf16.sh && bash ~/setup_amd_bf16.sh"
   ```
   *(Jeśli na VM jest nowsza wersja ROCm i potrzebujesz innych kół PyTorch, podaj np. `ROCM_INDEX_URL=https://download.pytorch.org/whl/rocm7.0 bash ~/setup_amd_bf16.sh`.)*
4. **W drugim oknie PowerShell otwórz bezpieczny tunel SSH:**
   ```powershell
   ssh -N -L 8188:127.0.0.1:8188 <user>@<VM_IP>
   ```
   Otwórz w przeglądarce **`http://127.0.0.1:8188`**.
5. **Po zakończeniu pracy:** pobierz wygenerowane obrazy/workflowy i **usuń (`Delete`) instancję** w panelu AMD Developer Cloud.
