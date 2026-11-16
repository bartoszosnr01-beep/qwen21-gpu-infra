# Beam.cloud (ComfyUI + Qwen-Image-2.1, `beam.Service` scale-to-zero)

> **UWAGA — BRAK DARMOWEGO LIMITU (zweryfikowano w panelu 16.11.2026):**
> Nowe konta Developer na `beam.cloud` startują obecnie z saldem **`$0.00`** i wymagają planu **`$20/mies.`** lub dokupienia kredytów przed uruchomieniem zadań (*„Add credits to run workloads”*). Ten katalog pozostaje w repozytorium wyłącznie jako gotowa rezerwa na wypadek płatnego użycia prepaid.

Wdrożenie wykorzystujące natywny `beam.Service` (`min_replicas=0`, `max_replicas=1`, `keep_warm_seconds=180`), dzięki czemu kontener usypia się automatycznie przy bezczynności (bez ręcznego zamykania jak w `Sandbox`).

---

## Bramka przed wdrożeniem (sprawdź w panelu Beam!)

1. Zarejestruj konto na `beam.cloud` i **najpierw wejdź w `Settings → Credits`**:
   * upewnij się, czy Twoje konto ma darmowe miesięczne kredyty bez obowiązkowej subskrypcji,
   * korzystaj z trybu **prepaid bez auto-doładowania** (obciążenia zatrzymują się automatycznie przy `0 USD`).
2. Zaloguj się w CLI z katalogu głównego projektu (`G:\Projekty\Infrastruktura`):
   ```powershell
   .\.venv\Scripts\beam.exe login
   ```

---

## Wdrożenie i pobranie wag (zawsze uruchamiaj z rootu repozytorium!)

1. **Pobierz wagi na stały `Volume` używając samego CPU (nie marnuj kredytów GPU na download):**
   ```powershell
   .\.venv\Scripts\python.exe .\GPU\Beam.cloud\comfy_qwen21_beam.py --download-models
   ```
   *(Opcjonalnie lżejszy profil pod ewentualne przyszłe naliczanie storage `0,0001 USD/GB-h`: `$env:QWEN21_MODEL_PROFILE="lite-int8"`).*

2. **Wdróż web service ComfyUI (`RTX4090` / `L4` / `A10G`, 24 GiB RAM, 180 s idle):**
   ```powershell
   .\.venv\Scripts\beam.exe deploy .\GPU\Beam.cloud\comfy_qwen21_beam.py:ui_service
   ```
   CLI wypisze publiczny URL do Twojego ComfyUI.

### Zmienne opcjonalne przed `deploy`
* `$env:QWEN21_BEAM_GPUS="RTX4090"` — wymusza wyłącznie RTX 4090 (domyślnie `RTX4090,L4,A10G`)
* `$env:QWEN21_BEAM_KEEP_WARM="120"` — krótsze usypianie (2 min)
* `$env:QWEN21_MODEL_PROFILE="lite-int8"` — tylko Base + VLM + VAE (~17,3 GB zamiast ~34 GB)
