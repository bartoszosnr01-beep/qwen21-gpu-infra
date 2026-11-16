# Infrastruktura GPU — Qwen-Image-2.1 (ComfyUI)

Jedno wspólne repozytorium definicji modeli, workflowów oraz wdrożeń pod darmowe platformy GPU.

## Struktura

| Katalog | Przeznaczenie | Status / Darmowy limit |
| :--- | :--- | :--- |
| [GPU/Wagi/qwen_image_2_1](GPU/Wagi/qwen_image_2_1) | Wspólny manifest modeli (`models.json`), downloader (`common.py`) i szablony `workflows/` | Jedno źródło prawdy dla wszystkich platform |
| [GPU/Modal](GPU/Modal) | Deployment serwerowy `comfy_qwen21.py` (`L4`/`A10`, `24 GiB RAM`, `180 s` idle) | **Wdrożone** (`30 USD/mies.` Starter + 1 TiB Volume) |
| [GPU/Beam.cloud](GPU/Beam.cloud) | `comfy_qwen21_beam.py` (`beam.Service` scale-to-zero, `RTX4090`/`L4`/`A10G`, `24Gi`) | **Brak darmowego limitu** (sprawdzone 16.11.2026: saldo `$0.00`, Developer od `20 USD/mies.`) |
| [GPU/AMD Developer Cloud](GPU/AMD%20Developer%20Cloud) | `setup_amd_bf16.sh` pod MI300X ROCm (`192 GB VRAM`, natywne wagi `full-bf16`) | **Gotowe** (`100 USD` / 30 dni po akceptacji; **zawsze `Delete` zamiast `Stop`**) |
| [GPU/Kaggle](GPU/Kaggle) | `run_kaggle_comfy.py` pod Notebook `T4 x2` (`lite-int8`, `/tmp/ComfyUI`, `cloudflared`) | **Gotowe** (`~30h/tydzień`) |
| [GPU/ModelScope](GPU/ModelScope) | `setup_modelscope.sh` pod `modelscope.cn` PAI-DSW (`A10 24 GB`, `HF_ENDPOINT=hf-mirror.com`) | **Gotowe** (`36h` jednorazowo po KYC Aliyun) |

---

## Szybkie komendy (Windows PowerShell z rootu repozytorium)

* **Modal — wdrożenie zmian:**
  ```powershell
  .\.venv\Scripts\modal.exe deploy .\GPU\Modal\comfy_qwen21.py
  ```
* **Beam.cloud — logowanie, pobranie wag na CPU i deploy:**
  ```powershell
  .\.venv\Scripts\beam.exe login
  .\.venv\Scripts\python.exe .\GPU\Beam.cloud\comfy_qwen21_beam.py --download-models
  .\.venv\Scripts\beam.exe deploy .\GPU\Beam.cloud\comfy_qwen21_beam.py:ui_service
  ```
* **Lokalne pobranie wag przez wspólne CLI (opcjonalnie):**
  ```powershell
  .\.venv\Scripts\python.exe .\GPU\Wagi\qwen_image_2_1\common.py --profile lite-int8
  ```

---

## Przypomnienia bezpieczeństwa (zero opłat)
1. **Modal:** w `Settings → Usage & Billing` ustaw twardy **Spend Limit = 30 USD**.
2. **Beam.cloud:** sprawdź `Settings → Credits` i wyłącz auto-doładowanie (prepaid zatrzymuje się przy `0 USD`).
3. **AMD Developer Cloud:** zatrzymana maszyna nadal nalicza opłaty — po zakończeniu sesji zawsze wybierz **Delete / Terminate**.
4. **ModelScope:** ręcznie klikaj **Stop** po sesji (instrukcje z `hf-mirror.com` w [GPU/ModelScope/README.md](GPU/ModelScope/README.md)).
