"""Generate self-contained Kaggle Jupyter notebooks embedding `qwen_kaggle` modules."""

from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PKG_DIR = ROOT_DIR / "qwen_kaggle"
OUT_DIR = ROOT_DIR / "notebooks"

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
        "PKG_ROOT = pathlib.Path(os.environ.get('QWEN_PKG_ROOT', '/tmp/qwen_pkg'))\n"
        "PKG_DIR = PKG_ROOT / 'qwen_kaggle'\n"
        "PKG_DIR.mkdir(parents=True, exist_ok=True)\n\n"
        f"_MODULES_B64 = {encoded}\n\n"
        "for name, b64 in _MODULES_B64.items():\n"
        "    (PKG_DIR / name).write_bytes(base64.b64decode(b64))\n\n"
        "for mod_key in [k for k in list(sys.modules) if k == 'qwen_kaggle' or k.startswith('qwen_kaggle.')]:\n"
        "    sys.modules.pop(mod_key, None)\n\n"
        "if str(PKG_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(PKG_ROOT))\n\n"
        "print(f'Unpacked {len(_MODULES_B64)} modules into {PKG_DIR}')\n"
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
                "# Faza 0: Smoke-Test Qwen-Image 2.1 na Kaggle (`GPU T4 x2`, `Internet: ON`)\n\n"
                "Ten notebook weryfikuje:\n"
                "1. Widocznosc obu kart T4 (`sm_75`), wolny dysk w `/tmp` oraz 29 GB RAM.\n"
                "2. Instalacje ComfyUI w `/tmp/ComfyUI` + izolowany `/tmp/venv` przez `uv`.\n"
                "3. Generacje `Turbo INT8` (`1024x1024` krotki i dlugi prompt, `1344x768`, oraz `Image Edit`) bez `--force-fp16`.\n"
                "4. Izolowany test drugiej karty (`CUDA_VISIBLE_DEVICES=1`) bez `ComfyUI-MultiGPU`.\n"
                "5. Zapis raportu `smoke_report.json` i obrazkow PNG do `/kaggle/working/smoke_test_results/`."
            ),
            _code_cell(pkg_cell),
            _code_cell(
                "import argparse, json\n"
                "from pathlib import Path\n"
                "from qwen_kaggle.constants import DEFAULT_COMFY_DIR, DEFAULT_INPUT_DIR, DEFAULT_VENV_DIR, DEFAULT_WORKING_DIR\n"
                "from qwen_kaggle.smoke_test import run_smoke_suite\n\n"
                "# --- Opcje testu ---\n"
                "TEST_BASE = False       # True: pobierze i przetestuje tez Base INT8 (25 krokow)\n"
                "TEST_XFORMERS = False   # True: doinstaluje xformers do porownania z PyTorch SDPA\n"
                "TEST_FP16_VAE = False   # True: wymusi --fp16-vae (domyslnie testujemy bezpieczne VAE)\n"
                "TEST_DUAL_GPU = True    # True: sprawdzi rowniez osobny proces na GPU 1\n"
                "FORCE_HF = False        # True: ignoruj /kaggle/input i pobierz z HuggingFace\n\n"
                "args = argparse.Namespace(\n"
                "    comfy_dir=DEFAULT_COMFY_DIR,\n"
                "    venv_dir=DEFAULT_VENV_DIR,\n"
                "    input_dir=DEFAULT_INPUT_DIR,\n"
                "    out_dir=DEFAULT_WORKING_DIR / 'smoke_test_results',\n"
                "    force_hf=FORCE_HF,\n"
                "    xformers=TEST_XFORMERS,\n"
                "    fp16_vae=TEST_FP16_VAE,\n"
                "    pytorch_sdpa=False,\n"
                "    test_base=TEST_BASE,\n"
                "    test_dual_gpu=TEST_DUAL_GPU,\n"
                ")\n"
                "report = run_smoke_suite(args)\n"
                "print(json.dumps({\n"
                "    'peak_ram_used_gb': report['peak_ram_used_gb'],\n"
                "    'peak_gpu_vram_mb': report['peak_gpu_vram_mb'],\n"
                "    'results': report['results'],\n"
                "}, indent=2))\n"
            ),
            _code_cell(
                "from pathlib import Path\n"
                "from IPython.display import Image, display\n\n"
                "res_dir = Path('/kaggle/working/smoke_test_results')\n"
                "for png in sorted(res_dir.glob('*.png')):\n"
                "    if png.name == 'smoke_edit_input.png':\n"
                "        continue\n"
                "    print(png.name)\n"
                "    display(Image(filename=str(png), width=384))\n"
            ),
        ]
    )

    nb2 = _make_notebook(
        [
            _md_cell(
                "# Etap 2: Jednorazowy zapis wag Qwen-Image 2.1 do prywatnego Kaggle Dataset (`CPU`, `Internet: ON`)\n\n"
                "**Wazne:** Ten notebook uruchamiaj na **CPU** (nie zuzywa tygodniowego limitu 30h GPU!).\n\n"
                "Przed uruchomieniem dodaj w Kaggle Secrets (`Add-ons -> Secrets`):\n"
                "- **Nowy styl (zalecany):** `KAGGLE_API_TOKEN` + `KAGGLE_USERNAME` (lub wpisz swoj nick w `KAGGLE_OWNER` ponizej)\n"
                "- **Lub styl Legacy (`kaggle.json`):** `KAGGLE_USERNAME` + `KAGGLE_KEY`\n\n"
                "Wagi zostana pobrane przez `hf_xet` do `/tmp` w ukladzie plaskim i wyslane jako **prywatny Kaggle Dataset**."
            ),
            _code_cell(pkg_cell),
            _code_cell(
                "from pathlib import Path\n"
                "from qwen_kaggle.prep_dataset import build_and_upload_dataset, build_and_upload_runtime_cache\n\n"
                "KAGGLE_OWNER = ''  # Opcjonalnie: wpisz swoj nick z profilu kaggle.com/<nick>, jesli w Secrets masz tylko KAGGLE_API_TOKEN\n"
                "DATASET_SLUG = 'qwen-image-21-int8-convrot'\n"
                "DATASET_TITLE = 'Qwen-Image 2.1 INT8 ConvRot Weights'\n"
                "RUNTIME_CACHE_SLUG = 'qwen-t4-runtime-cache'\n"
                "PRESET = 'full_i2i'  # 'core' (~22.9 GiB), 'full_i2i' (~31.7 GiB), 'all' (~40.5 GiB)\n"
                "BUILD_WEIGHTS = True\n"
                "BUILD_RUNTIME_CACHE = True\n"
                "UPDATE_EXISTING = False  # True, jesli Dataset juz istnieje i chcesz wyslac nowa wersje\n\n"
                "if BUILD_WEIGHTS:\n"
                "    dataset_id = build_and_upload_dataset(\n"
                "        dataset_slug=DATASET_SLUG,\n"
                "        title=DATASET_TITLE,\n"
                "        preset=PRESET,\n"
                "        owner=KAGGLE_OWNER,\n"
                "        staging_dir=Path('/tmp/qwen21_dataset_staging'),\n"
                "        update_existing=UPDATE_EXISTING,\n"
                "    )\n"
                "    print('Gotowy prywatny dataset wag:', dataset_id)\n\n"
                "if BUILD_RUNTIME_CACHE:\n"
                "    cache_id = build_and_upload_runtime_cache(\n"
                "        dataset_slug=RUNTIME_CACHE_SLUG,\n"
                "        owner=KAGGLE_OWNER,\n"
                "        staging_dir=Path('/tmp/qwen21_runtime_staging'),\n"
                "        update_existing=UPDATE_EXISTING,\n"
                "    )\n"
                "    print('Gotowy prywatny cache srodowiska:', cache_id)\n"
            ),
        ]
    )

    nb3 = _make_notebook(
        [
            _md_cell(
                "# Etap 3: Serwer Qwen-Image 2.1 + Qwen 3.5 9B na Kaggle (`GPU T4 x2`, `Internet: ON`)\n\n"
                "Ten notebook uruchamia:\n"
                "1. Dwie odizolowane instancje ComfyUI (`GPU 0 :8188` oraz `GPU 1 :8189`) bez `ComfyUI-MultiGPU`.\n"
                "2. Asynchroniczny `qwen_gateway` (`:8000`) z kolejka dla obu kart, Stage-1 `9B Prompt Enhancer` + `POST /free`, podgladem `TAESD` (SSE) oraz tokenem Bearer.\n"
                "3. Tunel `cloudflared` i blok **`BARTOSZ_OS_JSON`**, ktory wklejasz jednym kliknieciem w Bartosz OS (`Inteligencja i AI -> Kaggle (qwen 3.5)`).\n\n"
                "**Wskazowka:** Jesli zrobiles Etap 2, podepnij po prawej swoj prywatny dataset (`Add Input -> qwen-image-21-int8-convrot`), a wagi skopiuja sie lokalnie do `/tmp`."
            ),
            _code_cell(pkg_cell),
            _code_cell(
                "from qwen_kaggle.constants import DEFAULT_COMFY_DIR, DEFAULT_INPUT_DIR, DEFAULT_VENV_DIR\n"
                "from qwen_kaggle.gateway import prepare_and_launch_gateway_subprocess\n\n"
                "# --- Konfiguracja serwera Kaggle 2x T4 ---\n"
                "MODEL_PRESET = 'full_i2i'  # 'smoke' (~16.1 GiB), 'core' (~22.9 GiB), 'full_i2i' (~31.7 GiB), 'all' (~40.5 GiB)\n"
                "NUM_WORKERS = 2            # 2 = rownolegle GPU 0 + GPU 1\n"
                "PORT = 8000\n"
                "USE_TUNNEL = True\n"
                "IDLE_TIMEOUT_S = 600       # auto-stop po 10 min bez zadan (0 = wylaczone); reczny stop: przycisk w Bartosz OS (POST /v1/shutdown)\n"
                "PROMPT_TIMEOUT_S = 1800    # limit wykonania jednego zlecenia w sek. (30 min; 0 = wylaczone)\n"
                "FORCE_HF = False           # True ignoruje /kaggle/input\n"
                "BEARER_TOKEN = ''          # Opcjonalnie wlasny token (lub sekret QWEN_GATEWAY_TOKEN; puste = wygeneruje bezpieczny token)\n\n"
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
                "# Etap 4: Serwer Qwen-Image 2.1 + Qwen 3.5 9B na Kaggle (`TPU v5e-8`, 128 GiB HBM, `Internet: ON`)\n\n"
                "Ten notebook uruchamia silnik **PyTorch/XLA SPMD (8 chipow TPU v5e x 16 GiB = 128 GiB HBM, ~1.57 PFLOPS BF16)** korzystajacy z **osobnej puli 20h TPU tygodniowo**:\n"
                "1. Automatyczny podzial wag `QwenImage21Transformer2DModel` w natywnym `bfloat16` na wszystkie 8 chipow TPU (`xs.mark_sharding`) + statyczne koszyki dlugosci promptu (`256/512/1024`), eliminujace rekompilacje XLA miedzy promptami.\n"
                "2. Persistent XLA Cache (`/tmp/qwen_xla_cache`) oraz pojedynczy krok kompilacji `xm.mark_step()` na krok dyfuzji (brak rekompilacji przy zmianie liczby krokow `8` vs `25`).\n"
                "3. Pelna kompatybilnosc z istniejacym prywatnym datasetem `qwen-image-21-int8-convrot` (`MODEL_PRESET = 'full_i2i'`, automatyczna dekwantyzacja do `BF16` przy ladowaniu) lub natywnymi wagami `BF16` (`MODEL_PRESET = 'tpu_bf16'`).\n"
                "4. Identyczne API oraz blok **`BARTOSZ_OS_JSON`** do wklejenia w **Bartosz OS (`Inteligencja i AI -> Kaggle (qwen 3.5)`)**."
            ),
            _code_cell(pkg_cell),
            _code_cell(
                "from qwen_kaggle.constants import DEFAULT_COMFY_DIR, DEFAULT_INPUT_DIR, DEFAULT_VENV_DIR\n"
                "from qwen_kaggle.gateway import prepare_and_launch_gateway_subprocess\n\n"
                "# --- Konfiguracja serwera Kaggle TPU v5e-8 (128 GiB HBM) ---\n"
                "# 'full_i2i' (~31.7 GiB z Twojego datasetu Etapu 2 -> dekwantyzowane w locie do BF16)\n"
                "# lub 'tpu_bf16' (~52.3 GiB natywnych wag BF16 z HuggingFace)\n"
                "MODEL_PRESET = 'full_i2i'\n"
                "PORT = 8000\n"
                "USE_TUNNEL = True\n"
                "IDLE_TIMEOUT_S = 600       # auto-stop po 10 min bez zadan (0 = wylaczone); reczny stop: przycisk w Bartosz OS (POST /v1/shutdown)\n"
                "PROMPT_TIMEOUT_S = 1800    # limit wykonania jednego zlecenia w sek. (30 min; 0 = wylaczone)\n"
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
                "    idle_timeout_s=IDLE_TIMEOUT_S,\n"
                "    prompt_timeout_s=PROMPT_TIMEOUT_S,\n"
                "    accelerator='tpu',\n"
                ")\n"
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

    p1 = out_dir / "01_smoke_test_t4x2.ipynb"
    p2 = out_dir / "02_prep_dataset_cpu.ipynb"
    p3 = out_dir / "03_server_t4x2.ipynb"
    p4 = out_dir / "04_server_tpu_v5e8.ipynb"
    p1.write_text(json.dumps(nb1, indent=2), encoding="utf-8")
    p2.write_text(json.dumps(nb2, indent=2), encoding="utf-8")
    p3.write_text(json.dumps(nb3, indent=2), encoding="utf-8")
    p4.write_text(json.dumps(nb4, indent=2), encoding="utf-8")
    print(f"[BUILD] Generated {p1}, {p2}, {p3}, and {p4}")
    return [p1, p2, p3, p4]


if __name__ == "__main__":
    build_all()
