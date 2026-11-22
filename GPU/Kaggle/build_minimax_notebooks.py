"""Generate self-contained Kaggle Jupyter notebooks embedding `minimax_kaggle` modules."""

from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PKG_DIR = ROOT_DIR / "minimax_kaggle"
OUT_DIR = ROOT_DIR / "notebooks_minimax"

EMBEDDED_MODULES = (
    "__init__.py",
    "constants.py",
    "workflows.py",
    "bootstrap.py",
    "smoke_test.py",
    "prep_dataset.py",
    "tpu_runtime.py",
    "gateway.py",
)


def _md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.rstrip("\n").splitlines()],
    }


def _code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.rstrip("\n").splitlines()],
    }


def _build_bootstrap_pkg_cell() -> str:
    payload: dict[str, str] = {}
    for mod in EMBEDDED_MODULES:
        raw = (PKG_DIR / mod).read_bytes()
        payload[mod] = base64.b64encode(raw).decode("ascii")

    encoded = json.dumps(payload, indent=2)
    return (
        "import base64, os, pathlib, sys\n\n"
        "PKG_ROOT = pathlib.Path(os.environ.get('MINIMAX_PKG_ROOT', '/tmp/minimax_pkg'))\n"
        "PKG_DIR = PKG_ROOT / 'minimax_kaggle'\n"
        "PKG_DIR.mkdir(parents=True, exist_ok=True)\n\n"
        f"_MODULES_B64 = {encoded}\n\n"
        "for name, b64 in _MODULES_B64.items():\n"
        "    (PKG_DIR / name).write_bytes(base64.b64decode(b64))\n\n"
        "for mod_key in [k for k in list(sys.modules) if k == 'minimax_kaggle' or k.startswith('minimax_kaggle.')]:\n"
        "    sys.modules.pop(mod_key, None)\n\n"
        "if str(PKG_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(PKG_ROOT))\n\n"
        "print(f'Unpacked {len(_MODULES_B64)} minimax_kaggle modules into {PKG_DIR}')\n"
    )


def _make_notebook(cells: list[dict], kaggle_meta: dict | None = None) -> dict:
    metadata: dict = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
        },
    }
    if kaggle_meta:
        metadata["kaggle"] = kaggle_meta
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": metadata,
        "cells": cells,
    }


def build_all(out_dir: Path = OUT_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pkg_cell = _build_bootstrap_pkg_cell()

    nb1 = _make_notebook(
        [
            _md_cell(
                "# Faza 0: Smoke-Test MiniMax-H3-Image GGUF na Kaggle (`GPU T4 x2`, `Internet: ON`, bez PE)\n\n"
                "Ten notebook weryfikuje awaryjny tor **GGUF** (gdy TPU v5e-8 stoi w kolejce):\n"
                "1. Widocznosc kart T4 (`sm_75`), wolny dysk w `/tmp` oraz budzet ~29 GiB RAM.\n"
                "2. Instalacje ComfyUI (`>=0.45`) + `Nif00/ComfyUI-GGUF` (z obsluga przycietego `Qwen3-VL-32B` + `mmproj`) w `/tmp/ComfyUI`.\n"
                "3. Generacje 6-krokowe (`CFG 1.0`, `euler` + `simple`, bez Prompt Enhancera): `1024x1024`, `1344x768`, oraz **Image Edit (`<Picture 1>`)**.\n"
                "4. Pomiar szczytowego RAM/VRAM i zapis `minimax_smoke_report.json` + PNG do `/kaggle/working/minimax_smoke_results/`."
            ),
            _code_cell(pkg_cell),
            _code_cell(
                "import argparse, json\n"
                "from pathlib import Path\n"
                "from minimax_kaggle.constants import DEFAULT_COMFY_DIR, DEFAULT_INPUT_DIR, DEFAULT_VENV_DIR, DEFAULT_WORKING_DIR\n"
                "from minimax_kaggle.smoke_test import run_smoke_suite\n\n"
                "# --- Opcje testu T4 GGUF ---\n"
                "PRESET = 't4_gguf'           # 't4_gguf' (~22.1 GiB, Q4 DiT + IQ3 Clip) lub 't4_gguf_lite' (~19.6 GiB, Q3 DiT)\n"
                "TEST_2K = False              # True: sprawdzi tez 2048x2048 na T4 (uwaga na czas/VRAM)\n"
                "TEST_DUAL_CONCURRENT = True  # True: przetestuje jednoczesnie 2 workery (GPU 0 + GPU 1) z cache modeli\n"
                "TEST_GPU1_ISOLATED = True    # fallback: izolowany test GPU 1, gdy TEST_DUAL_CONCURRENT=False\n"
                "TEST_XFORMERS = False\n"
                "FORCE_HF = False             # True: ignoruj /kaggle/input i pobierz prosto z HuggingFace\n\n"
                "args = argparse.Namespace(\n"
                "    preset=PRESET,\n"
                "    comfy_dir=DEFAULT_COMFY_DIR,\n"
                "    venv_dir=DEFAULT_VENV_DIR,\n"
                "    input_dir=DEFAULT_INPUT_DIR,\n"
                "    out_dir=DEFAULT_WORKING_DIR / 'minimax_smoke_results',\n"
                "    force_hf=FORCE_HF,\n"
                "    xformers=TEST_XFORMERS,\n"
                "    pytorch_sdpa=False,\n"
                "    test_2k=TEST_2K,\n"
                "    test_dual_concurrent=TEST_DUAL_CONCURRENT,\n"
                "    test_gpu1_isolated=TEST_GPU1_ISOLATED,\n"
                ")\n"
                "report = run_smoke_suite(args)\n"
                "print(json.dumps({\n"
                "    'preset': report['preset'],\n"
                "    'peak_ram_used_gb': report['peak_ram_used_gb'],\n"
                "    'peak_cgroup_ram_gb': report['peak_cgroup_ram_gb'],\n"
                "    'peak_gpu_vram_mb': report['peak_gpu_vram_mb'],\n"
                "    'dual_concurrent_wall_s': report['dual_concurrent_wall_s'],\n"
                "    'results': report['results'],\n"
                "}, indent=2))\n"
            ),
            _code_cell(
                "from pathlib import Path\n"
                "from IPython.display import Image, display\n\n"
                "res_dir = Path('/kaggle/working/minimax_smoke_results')\n"
                "for png in sorted(res_dir.glob('*.png')):\n"
                "    if png.name == 'smoke_edit_input.png':\n"
                "        continue\n"
                "    print(png.name)\n"
                "    display(Image(filename=str(png), width=384))\n"
            ),
        ],
        kaggle_meta={
            "accelerator": "nvidiaTeslaT4",
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
            "isGpuEnabled": True,
        },
    )

    nb2 = _make_notebook(
        [
            _md_cell(
                "# Etap 2: Zapis wag MiniMax-H3-Image do prywatnego Kaggle Dataset (`CPU`, `Internet: ON`, bez PE)\n\n"
                "**Wazne:** Uruchamiaj na **CPU** (nie zuzywa limitow GPU ani TPU!). Zadne z repozytoriow nie jest gated (`HF_TOKEN` nie jest wymagany).\n\n"
                "Dodaj w Kaggle Secrets (`Add-ons -> Secrets`):\n"
                "- **Nowy styl:** `KAGGLE_API_TOKEN` + `KAGGLE_USERNAME` (lub wpisz nick w `KAGGLE_OWNER` ponizej)\n"
                "- **Lub Legacy:** `KAGGLE_USERNAME` + `KAGGLE_KEY`\n\n"
                "Dostepne presety (wszystkie **bez Prompt Enhancera**, prompty z Codexa):\n"
                "- `'tpu_int8'` (~`45.91 GiB`) -> **zalecany pod TPU v5e-8** (`INT8 ConvRot` DiT + `Qwen3-VL-32B INT8` + VAE)\n"
                "- `'tpu_bf16'` (~`63.84 GiB`) -> natywny `pruned_bf16` DiT + `Qwen3-VL-32B INT8` + VAE\n"
                "- `'t4_gguf'` (~`22.09 GiB`) -> **zalecany pod T4 x2 fallback** (`Q4_K_M` DiT + `UD-IQ3_XXS` Encoder + `mmproj` + VAE)\n"
                "- `'t4_gguf_lite'` (~`19.61 GiB`) -> najlzejszy wariant na T4 (`Q3_K_M` DiT)"
            ),
            _code_cell(pkg_cell),
            _code_cell(
                "from pathlib import Path\n"
                "from minimax_kaggle.prep_dataset import build_and_upload_dataset, build_and_upload_runtime_cache\n\n"
                "KAGGLE_OWNER = ''          # Opcjonalnie: Twoj nick z kaggle.com/<nick>, jesli w Secrets jest tylko KAGGLE_API_TOKEN\n"
                "PRESET = 'tpu_int8'        # 'tpu_int8' (~45.9 GiB), 'tpu_bf16' (~63.8 GiB), 't4_gguf' (~22.1 GiB), 't4_gguf_lite' (~19.6 GiB)\n"
                "DATASET_SLUG = ''          # Puste = domyslny slug dla wybranego PRESET (np. minimax-h3-image-tpu-int8 / minimax-h3-image-t4-gguf)\n"
                "DATASET_TITLE = ''         # Puste = domyslny tytul\n"
                "BUILD_WEIGHTS = True\n"
                "BUILD_RUNTIME_CACHE = False  # True: dodatkowo spakuje offline wheels + ComfyUI + ComfyUI-GGUF + cloudflared pod T4\n"
                "UPDATE_EXISTING = False    # True, jesli Dataset juz istnieje i wysylasz nowa wersje\n\n"
                "if BUILD_WEIGHTS:\n"
                "    dataset_id = build_and_upload_dataset(\n"
                "        preset=PRESET,\n"
                "        dataset_slug=DATASET_SLUG,\n"
                "        title=DATASET_TITLE,\n"
                "        owner=KAGGLE_OWNER,\n"
                "        staging_dir=Path(f'/tmp/minimax_dataset_staging_{PRESET}'),\n"
                "        update_existing=UPDATE_EXISTING,\n"
                "    )\n"
                "    print('Gotowy prywatny dataset wag:', dataset_id)\n\n"
                "if BUILD_RUNTIME_CACHE:\n"
                "    cache_id = build_and_upload_runtime_cache(\n"
                "        owner=KAGGLE_OWNER,\n"
                "        staging_dir=Path('/tmp/minimax_runtime_staging'),\n"
                "        update_existing=UPDATE_EXISTING,\n"
                "    )\n"
                "    print('Gotowy prywatny cache srodowiska:', cache_id)\n"
            ),
        ]
    )

    nb3 = _make_notebook(
        [
            _md_cell(
                "# Etap 3: Serwer MiniMax-H3-Image (GGUF) na Kaggle (`GPU T4 x2` fallback, `Internet: ON`, bez PE)\n\n"
                "Serwer MiniMax-H3-Image na Kaggle GPU T4 x2 (pula ~30h GPU/tyg.):\n"
                "1. ComfyUI z `--lowvram` + DynamicVRAM (`comfy-aimdo`) + `Nif00/ComfyUI-GGUF` i domyslnym cache modeli (bez przeladowywania GGUF miedzy promptami; 1 lub 2 workery GPU 0 / GPU 1).\n"
                "2. Brak Stage-1 Prompt Enhancera i brak cyklu `/free` — prompty od Codexa leca bezposrednio do `MiniMaxH3ImageConditioning` (6 krokow, `CFG 1.0`, obsluga `<Picture 1>`..`<Picture 16>`).\n"
                "3. Pelne API bramy Bartosz OS (`:8000`, SSE, `/v1/idle/extend`, `/v1/shutdown`, tunel `cloudflared` + `BARTOSZ_OS_JSON`).\n\n"
                "**Wskazowka:** Podepnij prywatny dataset `minimax-h3-image-t4-gguf` (z Etapu 2), aby przyspieszyc start."
            ),
            _code_cell(pkg_cell),
            _code_cell(
                "from minimax_kaggle.constants import DEFAULT_COMFY_DIR, DEFAULT_INPUT_DIR, DEFAULT_VENV_DIR\n"
                "from minimax_kaggle.gateway import prepare_and_launch_gateway_subprocess\n\n"
                "# --- Konfiguracja serwera Kaggle T4 x2 (GGUF) ---\n"
                "MODEL_PRESET = 't4_gguf'   # 't4_gguf' (~22.1 GiB) lub 't4_gguf_lite' (~19.6 GiB)\n"
                "NUM_WORKERS = 2            # 2 workery rownolegle GPU 0 + GPU 1 (zweryfikowane live: 113.5s za 2 obrazy, peak RAM 5.92 GiB)\n"
                "PORT = 8000\n"
                "USE_TUNNEL = True\n"
                "IDLE_TIMEOUT_S = 600       # auto-stop po 10 min bez zadan (0 = wylaczone)\n"
                "PROMPT_TIMEOUT_S = 2400    # limit jednego zlecenia w sek.\n"
                "FORCE_HF = False           # True ignoruje /kaggle/input\n"
                "BEARER_TOKEN = ''          # Opcjonalny token (lub sekret MINIMAX_GATEWAY_TOKEN / QWEN_GATEWAY_TOKEN)\n\n"
                "prepare_and_launch_gateway_subprocess(\n"
                "    comfy_dir=DEFAULT_COMFY_DIR,\n"
                "    venv_dir=DEFAULT_VENV_DIR,\n"
                "    input_dir=DEFAULT_INPUT_DIR,\n"
                "    preset=MODEL_PRESET,\n"
                "    port=PORT,\n"
                "    num_workers=NUM_WORKERS,\n"
                "    force_hf=FORCE_HF,\n"
                "    bearer_token=BEARER_TOKEN,\n"
                "    use_tunnel=USE_TUNNEL,\n"
                "    accelerator='gpu',\n"
                "    idle_timeout_s=IDLE_TIMEOUT_S,\n"
                "    prompt_timeout_s=PROMPT_TIMEOUT_S,\n"
                ")\n"
            ),
        ],
        kaggle_meta={
            "accelerator": "nvidiaTeslaT4",
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
            "isGpuEnabled": True,
        },
    )

    nb4 = _make_notebook(
        [
            _md_cell(
                "# Etap 4: Serwer MiniMax-H3-Image na Kaggle (`TPU v5e-8`, 128 GiB HBM, `Internet: ON`, bez PE)\n\n"
                "Docelowy serwer pelnej jakosci na **TPU VM v5e-8 (8x16 GiB = 128 GiB HBM + ~330 GiB RAM, osobna pula ~20h/tyg.)**:\n"
                "1. Automatyczna dekwantyzacja `minimax_h3_image_pruned_int8_convrot` (`19.54 GiB`) do natywnego `BF16` (lub bezposrednio `tpu_bf16`) z zachowaniem wysp `FP32` (`RMSNorm`, `video_patch_proj`, 6-krokowy PDD `final_layer.video_out`) i podzialem 1D SPMD na wszystkie 8 chipow TPU (`~5 GiB` wag/chip).\n"
                "2. Statyczne koszyki dlugosci promptu (`256/512/1024/2048`) + maskowanie paddingu w `TokenRefiner` i `DiTBlock`, eliminujace rekompilacje XLA miedzy promptami od Codexa, oraz persistent XLA Cache (`/tmp/minimax_xla_cache`).\n"
                "3. Natywny `Qwen3-VL-32B` (`25.28 GiB` INT8) bez Prompt Enhancera — tryby **T2I, Multi-Reference Edit (`<Picture 1>`..`<Picture 16>`), Inpaint z maska + referencjami oraz Outpaint** w 6 krokach (domyslnie 2K/4K, limit TPU podniesiony do **~48 MP** pod wieloosobowe sceny grupowe)!\n"
                "4. Identyczne API bramy Bartosz OS (`BARTOSZ_OS_JSON`, SSE, `/v1/idle/extend`, `/v1/shutdown`).\n\n"
                "**Wskazowka:** Podepnij prywatny dataset `minimax-h3-image-tpu-int8` z Etapu 2, aby pominac pobieranie ~46 GiB z HF."
            ),
            _code_cell(pkg_cell),
            _code_cell(
                "from minimax_kaggle.constants import DEFAULT_COMFY_DIR, DEFAULT_INPUT_DIR, DEFAULT_VENV_DIR\n"
                "from minimax_kaggle.gateway import prepare_and_launch_gateway_subprocess\n\n"
                "# --- Konfiguracja serwera Kaggle TPU v5e-8 (128 GiB HBM) ---\n"
                "MODEL_PRESET = 'tpu_int8'  # 'tpu_int8' (~45.9 GiB -> dekwantyzacja BF16 na TPU) lub 'tpu_bf16' (~63.8 GiB)\n"
                "PORT = 8000\n"
                "USE_TUNNEL = True\n"
                "IDLE_TIMEOUT_S = 600       # auto-stop po 10 min bez zadan (0 = wylaczone)\n"
                "PROMPT_TIMEOUT_S = 2400    # pierwsza kompilacja XLA 33B DiT wymaga zapasu czasu\n"
                "FORCE_HF = False\n"
                "BEARER_TOKEN = ''\n\n"
                "prepare_and_launch_gateway_subprocess(\n"
                "    comfy_dir=DEFAULT_COMFY_DIR,\n"
                "    venv_dir=DEFAULT_VENV_DIR,\n"
                "    input_dir=DEFAULT_INPUT_DIR,\n"
                "    preset=MODEL_PRESET,\n"
                "    port=PORT,\n"
                "    num_workers=1,\n"
                "    force_hf=FORCE_HF,\n"
                "    bearer_token=BEARER_TOKEN,\n"
                "    use_tunnel=USE_TUNNEL,\n"
                "    accelerator='tpu',\n"
                "    idle_timeout_s=IDLE_TIMEOUT_S,\n"
                "    prompt_timeout_s=PROMPT_TIMEOUT_S,\n"
                ")\n"
            ),
            _md_cell(
                "### Opcjonalnie: jednorazowy Smoke-Test TPU v5e-8 (cold compile + warm cached prompt)\n"
                "Jesli chcesz najpierw przetestowac sam silnik XLA SPMD bez uruchamiania bramy, odkomentuj ponizsza komorke i uruchom zamiast komorki powyzej."
            ),
            _code_cell(
                "# from minimax_kaggle.tpu_runtime import run_tpu_smoke_suite\n"
                "# report = run_tpu_smoke_suite(preset='tpu_int8', width=2048, height=2048)\n"
                "# print(report['results'])\n"
            ),
        ],
        kaggle_meta={
            "accelerator": "tpuV5e8",
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
            "isGpuEnabled": False,
        },
    )

    p1 = out_dir / "01_minimax_smoke_test_t4x2.ipynb"
    p2 = out_dir / "02_minimax_prep_dataset_cpu.ipynb"
    p3 = out_dir / "03_minimax_server_t4x2.ipynb"
    p4 = out_dir / "04_minimax_server_tpu_v5e8.ipynb"
    p1.write_text(json.dumps(nb1, indent=2), encoding="utf-8")
    p2.write_text(json.dumps(nb2, indent=2), encoding="utf-8")
    p3.write_text(json.dumps(nb3, indent=2), encoding="utf-8")
    p4.write_text(json.dumps(nb4, indent=2), encoding="utf-8")
    print(f"[BUILD] Generated {p1}, {p2}, {p3}, and {p4}")
    return [p1, p2, p3, p4]


if __name__ == "__main__":
    build_all()
