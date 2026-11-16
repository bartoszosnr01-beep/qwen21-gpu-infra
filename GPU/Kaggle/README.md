# Kaggle (ComfyUI + Qwen-Image-2.1, T4 × 2 / ~30h/tydzień)

Skrypt [run_kaggle_comfy.py](file:///g:/Projekty/Infrastruktura/GPU/Kaggle/run_kaggle_comfy.py) stawia pełny interfejs ComfyUI z naszym wspólnym `GPU\Wagi\qwen_image_2_1` na darmowej instancji **Kaggle Notebook** (GPU `T4 x2` lub `P100` — zalecane `T4 x2`, Internet `ON`).

> *(Uwaga: jest to osobny skrypt pod pełne **ComfyUI** ze wspólnych wag w `Infrastruktura\GPU\`, niezależny od istniejącego już na dysku `G:\Projekty\Kaggle`.)*

---

## Dlaczego taki układ katalogów na Kaggle?
* `/kaggle/working` na Kaggle ma twardy limit ok. **20 GB** — włożenie tam ComfyUI i wag kończy się błędem `No space left on device`.
* Dlatego skrypt instaluje ComfyUI i wagi do **`/tmp/ComfyUI`** (~50–70 GB wolnego dysku sesyjnego), a jedynie lekkie **workflowy** (`/kaggle/working/comfy_user`) oraz **wygenerowane obrazki** (`/kaggle/working/outputs`) zapisuje trwale w `/kaggle/working`.
* Domyślny profil to **`lite-int8`** (~17,3 GB) + flaga **`--cache-none`**, co mieści się bezpiecznie na **16 GB VRAM karty T4** i ~29 GB RAM. *(Możesz zmienić na `--profile w4a8` lub `--profile full-int8`.)*

---

## Szybkie uruchomienie w Notebooku Kaggle

1. Spakuj do prywatnego Kaggle Datasetu (nazwa np. `qwen21-shared`, poniżej **200 KB**) zawartość:
   * folder `GPU\Wagi\qwen_image_2_1\` (bez lokalnych `weights\`),
   * plik `GPU\Kaggle\run_kaggle_comfy.py`.
2. W nowym Notebooku na Kaggle włącz **Accelerator: GPU T4 x2**, **Internet: ON**, podepnij dataset `qwen21-shared` i odpal w jednej komórce:
   ```python
   !python /kaggle/input/qwen21-shared/run_kaggle_comfy.py
   ```
   *(Jeśli wrzuciłeś podfolder `qwen_image_2_1` obok skryptu, skrypt wykryje go automatycznie. Możesz też jawnie podać `--shared-dir` i `--profile w4a8`.)*
3. Po ~2–3 min pobierania zobaczysz w logu linijkę:
   ```text
   [READY] ComfyUI public URL (paste into Bartosz OS): https://xxxx.trycloudflare.com/
   ```
   Możesz otworzyć ten link bezpośrednio w przeglądarce lub wkleić w **Bartosz OS → QWEN 2.1 → Edytor adresu**.
