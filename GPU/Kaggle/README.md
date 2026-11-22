# Kaggle (`GPU 2× Tesla T4` & `TPU v5e-8`) — Bartosz OS & ComfyUI

Katalog łączy przeniesiony projekt **`Bartosz OS → Kaggle (qwen 3.5)`** (`qwen_kaggle/`, `notebooks/`, `Generacje/`) oraz samodzielny skrypt uruchamiający pełny interfejs ComfyUI (`run_kaggle_comfy.py`).

---

## 1. Brama i notebooki dla `Bartosz OS → Kaggle (qwen 3.5)`

* `qwen_kaggle/constants.py` — repozytorium `Comfy-Org/Qwen-Image-2.1`, rozmiary wag, `TURBO_MANUAL_SIGMAS`, bezpieczne flagi pod Turing `sm_75`.
* `qwen_kaggle/workflows.py` — deterministyczne buildery grafów API ComfyUI (`T2I`, `Image Edit`, `Turbo`, `Base`, osobny Stage-1 `TextGenerateV2` dla `qwen3.5_9b`).
* `qwen_kaggle/bootstrap.py` — diagnostyka maszyny, izolowany `/tmp/venv` przez `uv`, klon `ComfyUI` do `/tmp/ComfyUI`, sekwencyjne kopiowanie z `/kaggle/input` lub fallback przez `hf_xet`.
* `qwen_kaggle/smoke_test.py` — **Faza 0** (weryfikacja `sm_75`, braku `NaN`, RAM/VRAM i pracy GPU 0 + GPU 1).
* `qwen_kaggle/prep_dataset.py` — **Etap 2** (skrypt CPU pakujący wagi do prywatnego Kaggle Datasetu).
* `qwen_kaggle/gateway.py` — **Etap 3 / 4** (serwer `:8000` obsługujący Dual-T4 GPU i 8-chipowe TPU v5e-8, Stage-1 `9B PE` + `/free` + Stage-2, podgląd `TAESD` przez SSE, tunel `cloudflared`).
* `qwen_kaggle/tpu_runtime.py` — **Etap 4** (mostek PyTorch/XLA SPMD 128 GiB HBM na 8 chipów TPU v5e, statyczne koszyki promptów i dekwantyzacja `int8_convrot` → `BF16`).
* `build_notebooks.py` — generator samowystarczalnych notebooków w [`notebooks/`](notebooks).
* `Generacje/` — zapisane obrazy `.png` i metadane `.meta.json` z **Bartosz OS**.

### Szybkie uruchomienie notebooków bramy
1. **GPU `T4 x2` (~30h/tyg.):** wgraj `notebooks/03_server_t4x2.ipynb`, podepnij prywatny Dataset `qwen-image-21-int8-convrot` (jeśli utworzony), włącz **Accelerator: `GPU T4 x2`** i **Internet: `On`**, uruchom **Run All** i wklej `BARTOSZ_OS_JSON` w **Bartosz OS → Kaggle (qwen 3.5)** (albo użyj launchera z poziomu Bartosz OS, który sam czyta `notebooks/`).
2. **TPU `v5e-8` (128 GiB HBM, ~20h/tyg.):** analogicznie `notebooks/04_server_tpu_v5e8.ipynb` z **Accelerator: `TPU v5e-8`**.
3. **Wcześniejsze etapy:** `notebooks/01_smoke_test_t4x2.ipynb` (Faza 0) oraz `notebooks/02_prep_dataset_cpu.ipynb` (Etap 2, CPU).

---

## 2. Samodzielny runner pełnego ComfyUI UI (`run_kaggle_comfy.py`)

Skrypt [`run_kaggle_comfy.py`](run_kaggle_comfy.py) stawia pełny interfejs ComfyUI ze wspólnych definicji w [`GPU/Wagi/qwen_image_2_1`](../Wagi/qwen_image_2_1) w **`/tmp/ComfyUI`** (poza limitem `20 GB` w `/kaggle/working`), domyślnie na profilu **`lite-int8`** (~`17.3 GB`) z flagą `--cache-none`.

1. Spakuj do prywatnego Kaggle Datasetu (`qwen21-shared`, poniżej `200 KB`) folder `GPU\Wagi\qwen_image_2_1\` (bez lokalnych `weights\`) oraz `GPU\Kaggle\run_kaggle_comfy.py`.
2. W Notebooku (`GPU T4 x2`, `Internet: ON`) uruchom:
   ```python
   !python /kaggle/input/qwen21-shared/run_kaggle_comfy.py
   ```
3. Skopiuj wypisany adres `https://xxxx.trycloudflare.com/` do przeglądarki lub **Bartosz OS → QWEN 2.1 → Edytor adresu**.

---

## 3. Pakiet `MiniMax-H3-Image` (`minimax_kaggle/`, `notebooks_minimax/`, bez PE)

Równoległy, niezależny pakiet pod **MiniMax-H3-Image** (33,1B DiT + `Qwen3-VL-32B` L0-49 + 24-ch VAE FP16, **6 kroków, `CFG 1.0`, bez wbudowanego Prompt Enhancera** — prompty pisane zewnętrznie przez Codexa; przy edycji używaj znaczników `<Picture 1>`, `<Picture 2>`, choć brama automatycznie konwertuje też `<image1>` → `<Picture 1>`):

* `minimax_kaggle/constants.py` — katalog wag bez PE (żadne repo nie jest gated — `HF_TOKEN` zbędny):
  * **TPU `v5e-8`:** `tpu_int8` (`~45.91 GiB`: `dit_int8` `19.54 GiB` + `clip_int8` `25.28 GiB` + `vae` `1.09 GiB`) oraz `tpu_bf16` (`~63.84 GiB`).
  * **GPU `T4 x2` fallback (GGUF):** `t4_gguf` (`~22.09 GiB`: `Q4_K_M` DiT + `UD-IQ3_XXS` Encoder + `mmproj` + `vae`) oraz `t4_gguf_lite` (`~19.61 GiB`).
* `minimax_kaggle/workflows.py` — buildery API dla `MiniMaxH3ImageConditioning` + `ConditioningZeroOut` + 6-krokowego `KSampler` w 4 trybach (**`t2i`**, **`edit`**, **`inpaint`** z maską + dodatkowymi referencjami i pixel-lockiem `ImageCompositeMasked`, **`outpaint`** z `ImagePadForOutpaint` i featheringiem `80 px`), dla obu loaderów (`UNETLoader`/`CLIPLoader` oraz `UnetLoaderGGUF`/`CLIPLoaderGGUF` z `Nif00/ComfyUI-GGUF`).
* `minimax_kaggle/tpu_runtime.py` — mostek PyTorch/XLA SPMD 128 GiB HBM na 8 chipów TPU v5e dla `MiniMaxH3ImageModel` (pure-PyTorch `rms_rope_split_half`, statyczne koszyki promptów `256/512/1024/2048` z maskowaniem paddingu, dekwantyzacja `int8_convrot` → `BF16` z zachowaniem wysp `FP32` i replikacją 6-krokowej głowy PDD).
* `minimax_kaggle/gateway.py` — jednoetapowa brama `:8000` (bez Stage-1 i cyklu `/free`) w 100% zgodna z kontraktem Bartosz OS (`BARTOSZ_OS_JSON`, SSE, `/v1/idle/extend`, `/v1/shutdown`, `cloudflared`; limit rozdzielczości do **~48 MP** na TPU pod wieloosobowe sceny i `4,2 MP` na T4).
* `build_minimax_notebooks.py` — generator 4 samowystarczalnych notebooków w [`notebooks_minimax/`](notebooks_minimax):
  1. `01_minimax_smoke_test_t4x2.ipynb` — Faza 0 (GGUF na `GPU T4 x2`).
  2. `02_minimax_prep_dataset_cpu.ipynb` — Etap 2 (zapis wag `tpu_int8` / `tpu_bf16` / `t4_gguf` do prywatnego Kaggle Datasetu na **CPU**).
  3. `03_minimax_server_t4x2.ipynb` — Etap 3 (serwer `GPU T4 x2` GGUF, domyślnie **2 równoległe workery GPU 0 + GPU 1** z DynamicVRAM + RAM pressure cache; zweryfikowane na żywym T4 x2: `~112 s` warm T2I, `113,5 s` za 2× `1024×1024` równolegle, `peak_ram_used_gb: 5,92 GiB`).
  4. `04_minimax_server_tpu_v5e8.ipynb` — Etap 4 (docelowy serwer `TPU v5e-8` 128 GiB HBM, domyślnie 2K/4K, do **~48 MP**).
