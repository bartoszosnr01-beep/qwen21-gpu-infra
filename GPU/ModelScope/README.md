# ModelScope (`modelscope.cn` / Aliyun PAI-DSW — 36h A10 24 GB VRAM)

Konfiguracja pod darmowe środowisko Notebook na **`modelscope.cn`** (Alibaba Cloud PAI-DSW):
* **NVIDIA A10 (24 GB VRAM)**, 8 vCPU, **32 GB RAM**, **100 GB trwałego dysku** (`/mnt/workspace`).
* Dzięki 24 GB VRAM i 32 GB RAM domyślnie uruchamia pełny profil **`full-int8` (~34 GB)**.

> **Bramka wejściowa:** przed startem wymaga powiązania konta `modelscope.cn` z kontem Alibaba Cloud i przejścia weryfikacji tożsamości (dla kont spoza Chin paszportem — ręczny przegląd trwa ok. 24–36h). Po każdej sesji ręcznie kliknij **Stop** na instancji, żeby nie spalać puli 36 godzin.

---

## Uruchomienie w PAI-DSW

1. Wgraj przez panel plików JupyterLab (lub spakuj w archiwum `.zip`/`.tar.gz`) do `/mnt/workspace/`:
   * `GPU\Wagi\qwen_image_2_1\` → `/mnt/workspace/qwen21_shared` (bez lokalnych `weights\`),
   * `GPU\ModelScope\setup_modelscope.sh` → `/mnt/workspace/setup_modelscope.sh`.
2. W terminalu PAI-DSW uruchom:
   ```bash
   sed -i 's/\r$//' /mnt/workspace/setup_modelscope.sh
   bash /mnt/workspace/setup_modelscope.sh
   ```
3. Skrypt automatycznie używa `HF_ENDPOINT=https://hf-mirror.com` (żeby nie zacinało się za chińskim firewallem), buduje `/mnt/workspace/ComfyUI` i uruchamia ComfyUI na porcie `8188` (przez `cloudflared`, a jeśli jest zablokowany — przez wbudowane *Port Forwarding / Custom App* na porcie `8188` z ustawieniem `START_TUNNEL=0`).
4. Wszystkie pliki w `/mnt/workspace` przetrwają zatrzymanie (`Stop`) kontenera, ale własne workflowy z `ComfyUI/user` warto też trzymać lokalnie na wypadek czyszczenia kont nieaktywnych przez wiele dni.
