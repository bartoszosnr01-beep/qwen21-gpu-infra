"""Unit, mathematical-parity, and contract tests for minimax_kaggle workflows, TPU SPMD bridge, and notebooks."""

from __future__ import annotations

import ast
import base64
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from build_minimax_notebooks import build_all
from minimax_kaggle.bootstrap import (
    _extract_tar_safely,
    ensure_gguf_custom_node,
    ensure_physical_tpu_or_raise,
    find_input_asset,
    sequential_copy_file,
    setup_comfy_and_venv,
)
from minimax_kaggle.constants import (
    GGUF_NODE_DIRNAME,
    GGUF_PRESETS,
    MINIMAX_DEFAULT_CFG,
    MINIMAX_DEFAULT_STEPS,
    MODELS,
    PRESETS,
    RUNTIME_COMFY_ARCHIVE,
    RUNTIME_GGUF_ARCHIVE,
    RUNTIME_META_FILENAME,
    RUNTIME_WHEELS_ARCHIVE,
)
from minimax_kaggle.prep_dataset import DEFAULT_DATASET_SLUGS, _resolve_kaggle_auth
from minimax_kaggle.smoke_test import _http_json
from minimax_kaggle.workflows import (
    GenerationRequest,
    build_minimax_h3_api_prompt,
    normalize_minimax_reference_tags,
    prepare_single_branch_prompt,
    round_to_canvas_multiple,
)


class TestConstantsAndWorkflows(unittest.TestCase):
    def test_presets_reference_valid_models_and_exclude_prompt_enhancer(self) -> None:
        from minimax_kaggle.constants import T4_SAFE_COMFY_FLAGS
        from minimax_kaggle.prep_dataset import DEFAULT_RUNTIME_CACHE_TITLE

        self.assertIn("--lowvram", T4_SAFE_COMFY_FLAGS)
        self.assertNotIn("--cache-none", T4_SAFE_COMFY_FLAGS)
        self.assertTrue(6 <= len(DEFAULT_RUNTIME_CACHE_TITLE) <= 50)
        for preset_name, keys in PRESETS.items():
            self.assertGreater(len(keys), 0, preset_name)
            self.assertIn(preset_name, DEFAULT_DATASET_SLUGS)
            _slug, title = DEFAULT_DATASET_SLUGS[preset_name]
            self.assertTrue(6 <= len(title) <= 50, f"{preset_name} title too long ({len(title)}): {title!r}")
            for key in keys:
                self.assertIn(key, MODELS)
                self.assertNotIn("pe_", key)
                self.assertNotIn("_pe_", MODELS[key].filename)
            if preset_name in GGUF_PRESETS:
                self.assertIn("clip_gguf_mmproj", keys)

    def test_canvas_rounding_and_reference_tag_normalization(self) -> None:
        self.assertEqual(round_to_canvas_multiple(0), 0)
        self.assertEqual(round_to_canvas_multiple(1000), 992)
        self.assertEqual(round_to_canvas_multiple(1024), 1024)
        self.assertEqual(round_to_canvas_multiple(4096), 4096)

        self.assertEqual(
            normalize_minimax_reference_tags("Replace car in <image1> using style of <image2>"),
            "Replace car in <Picture 1> using style of <Picture 2>",
        )
        self.assertEqual(
            normalize_minimax_reference_tags("Edit <Picture 1> cleanly"),
            "Edit <Picture 1> cleanly",
        )

    def test_safetensors_tpu_t2i_graph(self) -> None:
        graph = build_minimax_h3_api_prompt(
            GenerationRequest(
                prompt="Editorial studio photograph of a matte ceramic vase on travertine stone.",
                mode="t2i",
                profile="quality",
                width=4096,
                height=4096,
                seed=1234,
                dit_key="dit_int8",
                clip_key="clip_int8",
            )
        )
        self.assertEqual(graph["1"]["class_type"], "UNETLoader")
        self.assertEqual(graph["1"]["inputs"]["unet_name"], MODELS["dit_int8"].filename)
        self.assertEqual(graph["3"]["class_type"], "CLIPLoader")
        self.assertEqual(graph["3"]["inputs"]["clip_name"], MODELS["clip_int8"].filename)
        self.assertEqual(graph["3"]["inputs"]["type"], "minimax")
        self.assertEqual(graph["4"]["class_type"], "VAELoader")
        self.assertEqual(graph["4"]["inputs"]["vae_name"], MODELS["vae"].filename)
        self.assertEqual(graph["5"]["class_type"], "MiniMaxH3ImageConditioning")
        self.assertEqual(graph["5"]["inputs"]["width"], 4096)
        self.assertEqual(graph["5"]["inputs"]["height"], 4096)
        self.assertEqual(graph["6"]["class_type"], "ConditioningZeroOut")
        self.assertEqual(graph["6"]["inputs"]["conditioning"], ["5", 0])
        self.assertEqual(graph["11"]["class_type"], "KSampler")
        self.assertEqual(graph["11"]["inputs"]["steps"], MINIMAX_DEFAULT_STEPS)
        self.assertEqual(graph["11"]["inputs"]["cfg"], MINIMAX_DEFAULT_CFG)
        self.assertEqual(graph["11"]["inputs"]["sampler_name"], "euler")
        self.assertEqual(graph["11"]["inputs"]["scheduler"], "simple")
        self.assertEqual(graph["11"]["inputs"]["latent_image"], ["5", 1])
        self.assertEqual(graph["12"]["class_type"], "VAEDecode")
        self.assertEqual(graph["13"]["class_type"], "SaveImage")

    def test_gguf_t4_edit_graph_normalizes_tags_and_supports_auto_canvas(self) -> None:
        graph = build_minimax_h3_api_prompt(
            GenerationRequest(
                prompt="Place the person from <image1> beside the red Vespa in <image2>.",
                mode="edit",
                profile="fast",
                width=0,
                height=0,
                image_names=("person.png", "vespa.png"),
                use_image_latent=True,
                dit_key="dit_gguf_q4",
                clip_key="clip_gguf_iq3",
            )
        )
        self.assertEqual(graph["1"]["class_type"], "UnetLoaderGGUF")
        self.assertEqual(graph["1"]["inputs"]["unet_name"], MODELS["dit_gguf_q4"].filename)
        self.assertEqual(graph["3"]["class_type"], "CLIPLoaderGGUF")
        self.assertEqual(graph["3"]["inputs"]["clip_name"], MODELS["clip_gguf_iq3"].filename)
        self.assertEqual(graph["3"]["inputs"]["type"], "minimax")
        self.assertEqual(graph["20"]["class_type"], "LoadImage")
        self.assertEqual(graph["20"]["inputs"]["image"], "person.png")
        self.assertEqual(graph["21"]["inputs"]["image"], "vespa.png")
        self.assertEqual(graph["5"]["inputs"]["images.image_1"], ["20", 0])
        self.assertEqual(graph["5"]["inputs"]["images.image_2"], ["21", 0])
        self.assertEqual(graph["5"]["inputs"]["width"], 0)
        self.assertEqual(graph["5"]["inputs"]["height"], 0)
        self.assertIn("<Picture 1>", graph["5"]["inputs"]["prompt"])
        self.assertIn("<Picture 2>", graph["5"]["inputs"]["prompt"])
        self.assertNotIn("<image1>", graph["5"]["inputs"]["prompt"])

    def test_validation_rejects_missing_edit_images_and_zero_t2i_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            build_minimax_h3_api_prompt(GenerationRequest(prompt="Edit", mode="edit"))
        with self.assertRaises(ValueError):
            build_minimax_h3_api_prompt(GenerationRequest(prompt="Test", mode="t2i", width=0, height=0))
        with self.assertRaises(ValueError):
            build_minimax_h3_api_prompt(
                GenerationRequest(prompt="Inpaint", mode="inpaint", image_names=("base.png",))
            )
        with self.assertRaises(ValueError):
            build_minimax_h3_api_prompt(
                GenerationRequest(prompt="Outpaint", mode="outpaint", image_names=("base.png",))
            )

    def test_inpaint_with_mask_and_extra_reference_images_locks_unmasked_pixels(self) -> None:
        graph = build_minimax_h3_api_prompt(
            GenerationRequest(
                prompt="Put the pink vest from <image2> and water bottle from <image3> onto the person in <image1>.",
                mode="inpaint",
                width=0,
                height=0,
                image_names=("woman.png", "pink_vest.png", "bottle.png"),
                mask_image_name="torso_mask.png",
                mask_channel="red",
                mask_expand=16,
                mask_blur=12,
                dit_key="dit_int8",
                clip_key="clip_int8",
            )
        )
        self.assertEqual(graph["20"]["inputs"]["image"], "woman.png")
        self.assertEqual(graph["21"]["inputs"]["image"], "pink_vest.png")
        self.assertEqual(graph["22"]["inputs"]["image"], "bottle.png")
        self.assertEqual(graph["40"]["class_type"], "LoadImageMask")
        self.assertEqual(graph["40"]["inputs"]["image"], "torso_mask.png")
        self.assertEqual(graph["41"]["class_type"], "GrowMask")
        self.assertEqual(graph["41"]["inputs"]["expand"], 16)
        self.assertEqual(graph["43"]["class_type"], "ImageBlur")
        self.assertEqual(graph["43"]["inputs"]["blur_radius"], 12)
        self.assertEqual(graph["5"]["inputs"]["images.image_1"], ["20", 0])
        self.assertEqual(graph["5"]["inputs"]["images.image_2"], ["21", 0])
        self.assertEqual(graph["5"]["inputs"]["images.image_3"], ["22", 0])
        self.assertIn("<Picture 2>", graph["5"]["inputs"]["prompt"])
        self.assertIn("<Picture 3>", graph["5"]["inputs"]["prompt"])
        self.assertEqual(graph["7"]["class_type"], "VAEEncode")
        self.assertEqual(graph["8"]["class_type"], "SetLatentNoiseMask")
        self.assertEqual(graph["8"]["inputs"]["mask"], ["44", 0])
        self.assertEqual(graph["11"]["inputs"]["latent_image"], ["8", 0])
        self.assertEqual(graph["14"]["class_type"], "ImageCompositeMasked")
        self.assertEqual(graph["14"]["inputs"]["destination"], ["20", 0])
        self.assertEqual(graph["14"]["inputs"]["source"], ["12", 0])
        self.assertEqual(graph["14"]["inputs"]["mask"], ["44", 0])
        self.assertEqual(graph["13"]["inputs"]["images"], ["14", 0])

    def test_outpaint_pads_to_multiple_of_32_and_composites_with_feather(self) -> None:
        graph = build_minimax_h3_api_prompt(
            GenerationRequest(
                prompt="Extend the scenic alpine mountain background seamlessly across the left and right edges.",
                mode="outpaint",
                width=0,
                height=0,
                image_names=("portrait.png",),
                outpaint_left=500,
                outpaint_right=500,
                outpaint_feather=80,
                dit_key="dit_int8",
                clip_key="clip_int8",
            )
        )
        self.assertEqual(graph["30"]["class_type"], "ImagePadForOutpaint")
        self.assertEqual(graph["30"]["inputs"]["left"], 512)
        self.assertEqual(graph["30"]["inputs"]["right"], 512)
        self.assertEqual(graph["30"]["inputs"]["top"], 0)
        self.assertEqual(graph["30"]["inputs"]["bottom"], 0)
        self.assertEqual(graph["30"]["inputs"]["feathering"], 80)
        self.assertEqual(graph["31"]["class_type"], "GetImageSize")
        self.assertEqual(graph["5"]["inputs"]["width"], ["31", 0])
        self.assertEqual(graph["5"]["inputs"]["height"], ["31", 1])
        self.assertEqual(graph["8"]["class_type"], "SetLatentNoiseMask")
        self.assertEqual(graph["8"]["inputs"]["mask"], ["30", 1])
        self.assertEqual(graph["14"]["class_type"], "ImageCompositeMasked")
        self.assertEqual(graph["14"]["inputs"]["destination"], ["30", 0])
        self.assertEqual(graph["14"]["inputs"]["mask"], ["30", 1])
        self.assertEqual(graph["13"]["inputs"]["images"], ["14", 0])

    def test_prepare_single_branch_prompt_and_identity_blend_modes(self) -> None:
        cleaned = prepare_single_branch_prompt(
            "Portrait blending <image1> and <image2> with both parents' features. Do not copy clothing from either reference."
        )
        self.assertIn("<Picture 1>", cleaned)
        self.assertNotIn("<Picture 2>", cleaned)
        self.assertNotIn("either reference", cleaned)
        self.assertIn("<Picture 1>'s", cleaned)

        graph = build_minimax_h3_api_prompt(
            GenerationRequest(
                prompt="Blend <Picture 1> and <Picture 2>",
                mode="edit",
                width=1024,
                height=1024,
                image_names=("a.png", "b.png"),
                identity_blend_mode="velocity_blend",
                identity_blend_ratio=0.4,
            )
        )
        self.assertNotIn("5", graph)
        conds = [n for n in graph.values() if n["class_type"] == "MiniMaxH3ImageConditioning"]
        self.assertEqual(len(conds), 2)
        strengths = [n["inputs"]["strength"] for n in graph.values() if n["class_type"] == "ConditioningSetAreaStrength"]
        self.assertEqual(strengths, [0.6, 0.4])
        combines = [nid for nid, n in graph.items() if n["class_type"] == "ConditioningCombine"]
        self.assertEqual(len(combines), 1)
        self.assertEqual(graph["6"]["inputs"]["conditioning"], [combines[0], 0])
        self.assertEqual(graph["11"]["inputs"]["positive"], [combines[0], 0])
        self.assertEqual(graph["11"]["inputs"]["negative"], ["6", 0])

        step_graph = build_minimax_h3_api_prompt(
            GenerationRequest(
                prompt="Blend <Picture 1>, <Picture 2>, and <Picture 3>",
                mode="edit",
                width=1024,
                height=1024,
                image_names=("a.png", "b.png", "c.png"),
                identity_blend_mode="step_swap",
            )
        )
        ranges = [
            (n["inputs"]["start"], n["inputs"]["end"])
            for n in step_graph.values()
            if n["class_type"] == "ConditioningSetTimestepRange"
        ]
        self.assertEqual(len(ranges), 3)
        self.assertEqual(ranges[0][0], 0.0)
        self.assertEqual(ranges[0][1], ranges[1][0])
        self.assertEqual(ranges[1][1], ranges[2][0])
        self.assertEqual(ranges[2][1], 1.0)


class TestBootstrapAndNotebooks(unittest.TestCase):
    def test_find_and_sequential_copy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp = root / "input" / "minimax-dataset"
            inp.mkdir(parents=True)
            src = inp / "minimax_h3_image_pruned_int8_convrot.safetensors"
            payload = b"minimax-h3-binary-payload-" * 100
            src.write_bytes(payload)

            found = find_input_asset(src.name, root / "input")
            self.assertEqual(found, src)

            dst = root / "tmp" / "models" / src.name
            sequential_copy_file(src, dst)
            self.assertEqual(dst.read_bytes(), payload)

    def test_setup_comfy_and_gguf_uses_cached_archives_and_rejects_traversal(self) -> None:
        import minimax_kaggle.bootstrap as bs

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_dir = root / "input" / "runtime-cache"
            input_dir.mkdir(parents=True)

            # ComfyUI tar
            comfy_src = root / "src_comfy" / "ComfyUI"
            comfy_src.mkdir(parents=True)
            (comfy_src / "main.py").write_text("# comfy\n", encoding="utf-8")
            (comfy_src / "requirements.txt").write_text("numpy\n", encoding="utf-8")
            with tarfile.open(input_dir / RUNTIME_COMFY_ARCHIVE, "w") as tf:
                tf.add(comfy_src / "main.py", arcname="ComfyUI/main.py")
                tf.add(comfy_src / "requirements.txt", arcname="ComfyUI/requirements.txt")

            # ComfyUI-GGUF tar
            gguf_src = root / "src_gguf" / GGUF_NODE_DIRNAME
            gguf_src.mkdir(parents=True)
            (gguf_src / "nodes.py").write_text("# gguf nodes\n", encoding="utf-8")
            with tarfile.open(input_dir / RUNTIME_GGUF_ARCHIVE, "w") as tf:
                tf.add(gguf_src / "nodes.py", arcname=f"{GGUF_NODE_DIRNAME}/nodes.py")

            # Wheels tar + python_mm metadata
            wheel_file = root / "numpy-0.0.1-py3-none-any.whl"
            wheel_file.write_bytes(b"wheel")
            with tarfile.open(input_dir / RUNTIME_WHEELS_ARCHIVE, "w") as tf:
                tf.add(wheel_file, arcname=wheel_file.name)
            (input_dir / RUNTIME_META_FILENAME).write_text(
                json.dumps({"python_mm": f"{sys.version_info[0]}.{sys.version_info[1]}"}),
                encoding="utf-8",
            )

            # Unsafe traversal tar must fail closed
            bad_tar = root / "bad.tar"
            with tarfile.open(bad_tar, "w") as tf:
                info = tarfile.TarInfo("../evil.txt")
                info.size = 3
                tf.addfile(info, io.BytesIO(b"bad"))
            with self.assertRaises(RuntimeError):
                bs._extract_tar_safely(bad_tar, root / "extract_dst")

            comfy_dir = root / "tmp" / "ComfyUI"
            venv_dir = root / "tmp" / "venv"
            recorded_cmds: list[list[str]] = []

            def fake_run(cmd: list[str], **_kwargs: Any) -> None:
                recorded_cmds.append(list(cmd))
                if cmd[:2] == ["uv", "venv"]:
                    py = bs.venv_python_path(venv_dir)
                    py.parent.mkdir(parents=True, exist_ok=True)
                    py.write_text("", encoding="utf-8")

            with (
                mock.patch.object(bs, "ensure_uv", return_value="uv"),
                mock.patch.object(bs, "_gpu_venv_ready", return_value=False),
                mock.patch.object(bs, "_run", side_effect=fake_run),
            ):
                bs.setup_comfy_and_venv(comfy_dir, venv_dir, accelerator="gpu", input_root=root / "input")

            self.assertTrue((comfy_dir / "main.py").exists())
            self.assertTrue((comfy_dir / "custom_nodes" / GGUF_NODE_DIRNAME / "nodes.py").exists())
            self.assertFalse(any("git" in c and "clone" in c for c in recorded_cmds))
            pip_cmds = [c for c in recorded_cmds if len(c) >= 3 and c[:3] == ["uv", "pip", "install"]]
            self.assertEqual(len(pip_cmds), 1)
            self.assertIn("--no-index", pip_cmds[0])
            verify_cmds = [c for c in recorded_cmds if len(c) >= 3 and c[1] == "-c" and "[TORCH]" in c[2]]
            self.assertEqual(len(verify_cmds), 1)
            ast.parse(verify_cmds[0][2])
            self.assertNotIn("gguf.__version__", verify_cmds[0][2])
            self.assertIn("importlib.metadata.version('gguf')", verify_cmds[0][2])

    def test_notebook_generation_syntax_unpack_and_accelerators(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = build_all(root / "notebooks_minimax")
            self.assertEqual(len(paths), 4)

            expected_accels = {
                "01_minimax_smoke_test_t4x2.ipynb": "nvidiaTeslaT4",
                "02_minimax_prep_dataset_cpu.ipynb": None,
                "03_minimax_server_t4x2.ipynb": "nvidiaTeslaT4",
                "04_minimax_server_tpu_v5e8.ipynb": "tpuV5e8",
            }
            for nb_path in paths:
                nb = json.loads(nb_path.read_text(encoding="utf-8"))
                self.assertEqual(nb["nbformat"], 4)
                kmeta = nb["metadata"].get("kaggle") or {}
                self.assertEqual(kmeta.get("accelerator"), expected_accels[nb_path.name])

                code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
                self.assertGreaterEqual(len(code_cells), 2)
                for cell in code_cells:
                    ast.parse("".join(cell["source"]))

                unpack_dir = root / f"unpack_{nb_path.stem}"
                first_src = "".join(code_cells[0]["source"])
                old_env = os.environ.get("MINIMAX_PKG_ROOT")
                old_path = list(sys.path)
                old_mods = {
                    k: v
                    for k, v in sys.modules.items()
                    if k == "minimax_kaggle" or k.startswith("minimax_kaggle.")
                }
                try:
                    os.environ["MINIMAX_PKG_ROOT"] = str(unpack_dir)
                    ns: dict[str, Any] = {}
                    exec(compile(first_src, str(nb_path), "exec"), ns)
                    self.assertEqual(len(ns["_MODULES_B64"]), 8)
                    for mod_name, b64 in ns["_MODULES_B64"].items():
                        mod_file = unpack_dir / "minimax_kaggle" / mod_name
                        self.assertTrue(mod_file.exists())
                        ast.parse(base64.b64decode(b64).decode("utf-8"), filename=mod_name)
                finally:
                    sys.path[:] = old_path
                    for k in [k for k in list(sys.modules) if k == "minimax_kaggle" or k.startswith("minimax_kaggle.")]:
                        sys.modules.pop(k, None)
                    sys.modules.update(old_mods)
                    if old_env is None:
                        os.environ.pop("MINIMAX_PKG_ROOT", None)
                    else:
                        os.environ["MINIMAX_PKG_ROOT"] = old_env

    def test_inline_hf_download_scripts_are_valid_python_syntax(self) -> None:
        import minimax_kaggle.bootstrap as bs
        import minimax_kaggle.prep_dataset as pd

        captured_scripts: list[str] = []

        def fake_bootstrap_run(cmd: list[str], **_kwargs: Any) -> None:
            if len(cmd) >= 3 and cmd[1] == "-c":
                captured_scripts.append(cmd[2])

        def fake_prep_subproc(cmd: list[str], **_kwargs: Any) -> None:
            if len(cmd) >= 3 and cmd[1] == "-c":
                captured_scripts.append(cmd[2])

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fake_py = root / "python"
            with mock.patch.object(bs, "_run", side_effect=fake_bootstrap_run):
                bs._hf_download_in_venv(
                    fake_py,
                    "Comfy-Org/MiniMax-H3-Image",
                    "diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors",
                    root / "models" / "diffusion_models" / "minimax_h3_image_pruned_int8_convrot.safetensors",
                )

            fake_resp = mock.MagicMock()
            fake_resp.read.side_effect = [b"{}", b""]
            fake_ctx = mock.MagicMock()
            fake_ctx.__enter__.return_value = fake_resp
            fake_ctx.__exit__.return_value = False
            with (
                mock.patch.object(pd.subprocess, "run", side_effect=fake_prep_subproc),
                mock.patch.object(pd.urllib.request, "urlopen", return_value=fake_ctx),
                mock.patch.object(pd.shutil, "copyfileobj"),
            ):
                pd._stage_flat_assets(fake_py, root / "staging", "tpu_int8")

        self.assertGreaterEqual(len(captured_scripts), 4)
        for script in captured_scripts:
            ast.parse(script)

    def test_http_json_and_kaggle_auth_helpers(self) -> None:
        empty_resp = mock.MagicMock()
        empty_resp.read.return_value = b""
        empty_ctx = mock.MagicMock()
        empty_ctx.__enter__.return_value = empty_resp
        empty_ctx.__exit__.return_value = False
        with mock.patch("minimax_kaggle.smoke_test.urllib.request.urlopen", return_value=empty_ctx):
            self.assertEqual(_http_json("http://127.0.0.1:8188/free", payload={"unload_models": True}), {})

        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch("minimax_kaggle.prep_dataset.Path.home", return_value=Path(td)),
            mock.patch("minimax_kaggle.prep_dataset._read_kaggle_secret", return_value=""),
            mock.patch.dict(os.environ, {"KAGGLE_API_TOKEN": "KGAT_mm_token"}, clear=True),
        ):
            owner, env = _resolve_kaggle_auth(owner="bartek")
            self.assertEqual(owner, "bartek")
            self.assertEqual(env, {"KAGGLE_API_TOKEN": "KGAT_mm_token"})

    def test_ensure_physical_tpu_or_raise(self) -> None:
        import minimax_kaggle.bootstrap as bs

        fake_diag = {"memory": {"ram_total_gb": 31.35}, "cpu_count": 4}
        with mock.patch.object(bs, "probe_outer_tpu_jax", return_value="OUTER_JAX_RESULT: 1 cpu cpu"):
            with self.assertRaises(RuntimeError) as ctx:
                ensure_physical_tpu_or_raise(fake_diag)
            self.assertIn("WITHOUT physical TPU", str(ctx.exception))

        with mock.patch.object(bs, "probe_outer_tpu_jax", return_value="OUTER_JAX_RESULT: 8 tpu TPU v5 lite"):
            self.assertIn("8 tpu", ensure_physical_tpu_or_raise(fake_diag))


class TestMiniMaxTpuRuntimeBridge(unittest.TestCase):
    def test_xla_rms_rope_split_half_matches_comfy_kitchen_reference_math(self) -> None:
        import torch
        import torch.nn.functional as F
        from minimax_kaggle.tpu_runtime import xla_rms_rope_split_half

        torch.manual_seed(7)
        s, heads, head_dim, rot_dim = 19, 4, 128, 96
        x = torch.randn(1, s, heads, head_dim, dtype=torch.float32)
        w = torch.randn(head_dim, dtype=torch.float32)
        angles = torch.randn(s, rot_dim, dtype=torch.float32)
        half = rot_dim // 2
        ang = angles[:, :half]
        c, sn = torch.cos(ang), torch.sin(ang)
        freqs = torch.stack([c, -sn, sn, c], dim=-1).reshape(1, s, 1, half, 2, 2)

        # Reference matching comfy_kitchen/backends/eager/rope.py::_rms_rope1 + apply_rope_split_half1
        x_norm = F.rms_norm(x, (head_dim,), weight=w, eps=1e-5)
        xr = x_norm[..., :rot_dim]
        diag = freqs.diagonal(dim1=-2, dim2=-1).movedim(-1, -2)
        ref_rot = xr.unflatten(-1, (2, -1)) * diag
        ref_rot[..., 0, :] = ref_rot[..., 0, :] + xr[..., half:] * freqs[..., 0, 1]
        ref_rot[..., 1, :] = ref_rot[..., 1, :] + xr[..., :half] * freqs[..., 1, 0]
        ref = torch.cat([ref_rot.reshape(xr.shape), x_norm[..., rot_dim:]], dim=-1)

        actual = xla_rms_rope_split_half(x, freqs, w, epsilon=1e-5, rot_dim=rot_dim)
        self.assertEqual(actual.shape, ref.shape)
        self.assertTrue(torch.allclose(actual, ref, atol=1e-6))

    def test_static_tpu_context_padding_preserves_rope_and_attention_parity(self) -> None:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from minimax_kaggle.tpu_runtime import (
            build_static_tpu_layout_and_masks,
            choose_sharding_spec,
            choose_static_context_bucket,
            pad_minimax_context,
            shard_module_for_spmd,
        )

        self.assertEqual(choose_static_context_bucket(11), 256)
        self.assertEqual(choose_static_context_bucket(256), 256)
        self.assertEqual(choose_static_context_bucket(300), 512)

        class _FakePackedLayout:
            def __init__(self, text_len: int, latent_t: int, lat_h: int, lat_w: int, audio_t: int, refs=None) -> None:
                img_n = latent_t * (lat_h // 2) * (lat_w // 2)
                g_txt = torch.zeros(text_len, 3, dtype=torch.float64)
                g_txt[:, 0] = torch.arange(text_len, dtype=torch.float64)
                g_img = torch.zeros(img_n, 3, dtype=torch.float64)
                g_img[:, 0] = float(text_len)
                g_img[:, 1] = torch.arange(img_n, dtype=torch.float64)
                self.position_ids = torch.cat([g_txt, g_img], dim=0)
                self.seq_len = text_len + img_n
                self.segments = [(0, text_len, "text"), (text_len, self.seq_len, "video")]

        torch.manual_seed(42)
        dim = 64
        ctx_11 = torch.randn(1, 11, dim)
        ctx_23 = torch.randn(1, 23, dim)
        tags_11 = torch.zeros(11, dtype=torch.long)

        p11, pt11, r11, b11 = pad_minimax_context(ctx_11, tags_11, buckets=(32,))
        p23, _, r23, b23 = pad_minimax_context(ctx_23, None, buckets=(32,))
        self.assertEqual((r11, b11), (11, 32))
        self.assertEqual((r23, b23), (23, 32))
        self.assertEqual(p11.shape, p23.shape)
        self.assertEqual(pt11.shape, (32,))

        lay11, ref_mask11, dit_mask11 = build_static_tpu_layout_and_masks(
            r11, b11, 1, 8, 8, None, _FakePackedLayout, torch.device("cpu")
        )
        lay23, ref_mask23, dit_mask23 = build_static_tpu_layout_and_masks(
            r23, b23, 1, 8, 8, None, _FakePackedLayout, torch.device("cpu")
        )
        lay_unpad, ref_mask_u, dit_mask_u = build_static_tpu_layout_and_masks(
            11, 11, 1, 8, 8, None, _FakePackedLayout, torch.device("cpu")
        )
        self.assertIsNone(ref_mask_u)
        self.assertIsNone(dit_mask_u)
        self.assertEqual(lay11.seq_len, lay23.seq_len)
        self.assertEqual(dit_mask11.shape, dit_mask23.shape)

        # Target image RoPE position_ids after bucketed text must match unpadded positions exactly
        self.assertTrue(torch.allclose(lay11.position_ids[32:], lay_unpad.position_ids[11:]))

        # Bidirectional attention with dit_mask must yield identical outputs for real text & image tokens
        img_toks = torch.randn(1, 16, dim)
        q_proj = nn.Linear(dim, dim, bias=False)
        k_proj = nn.Linear(dim, dim, bias=False)
        v_proj = nn.Linear(dim, dim, bias=False)

        seq_pad = torch.cat([p11, img_toks], dim=1)  # [1, 48, dim]
        seq_unpad = torch.cat([ctx_11, img_toks], dim=1)  # [1, 27, dim]

        out_pad = F.scaled_dot_product_attention(
            q_proj(seq_pad).unsqueeze(1),
            k_proj(seq_pad).unsqueeze(1),
            v_proj(seq_pad).unsqueeze(1),
            attn_mask=dit_mask11,
        ).squeeze(1)
        out_unpad = F.scaled_dot_product_attention(
            q_proj(seq_unpad).unsqueeze(1),
            k_proj(seq_unpad).unsqueeze(1),
            v_proj(seq_unpad).unsqueeze(1),
            attn_mask=None,
        ).squeeze(1)
        self.assertTrue(torch.allclose(out_pad[:, :11], out_unpad[:, :11], atol=1e-5))
        self.assertTrue(torch.allclose(out_pad[:, 32:], out_unpad[:, 11:], atol=1e-5))

        # Verify FP32 island preservation & PDD head replication in SPMD sharding
        self.assertEqual(choose_sharding_spec((5376, 14336), 8), ("model", None))

        class _FakeQuant:
            def __init__(self, t: torch.Tensor) -> None:
                self._t = t

            def dequantize(self) -> torch.Tensor:
                return self._t

        class _MiniDiT(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.video_patch_proj = nn.Linear(96, 512, bias=True, dtype=torch.float32)
                self.blocks = nn.ModuleList([nn.Linear(512, 512, bias=False, dtype=torch.float32)])
                self.blocks[0]._buffers["q_buf"] = _FakeQuant(torch.ones(512, 512, dtype=torch.float32))  # type: ignore[assignment]
                self.final_layer = nn.Module()
                self.final_layer.video_out = nn.Linear(512, 576, bias=True, dtype=torch.float32)

        dit = _MiniDiT()
        sharded_shapes: list[tuple[tuple[int, ...], Any]] = []
        stats = shard_module_for_spmd(
            dit,
            xla_device=torch.device("cpu"),
            mesh="mesh8",
            num_chips=8,
            mark_sharding_fn=lambda t, m, s: sharded_shapes.append((tuple(t.shape), s)),
        )
        self.assertEqual(stats["dequantized"], 1)
        # DiT block linear converts to BF16 and shards across 8 chips
        self.assertEqual(dit.blocks[0].weight.dtype, torch.bfloat16)
        # FP32 islands stay float32 and replicated (NOT sharded)
        self.assertEqual(dit.video_patch_proj.weight.dtype, torch.float32)
        self.assertEqual(dit.final_layer.video_out.weight.dtype, torch.float32)
        self.assertNotIn(((576, 512), ("model", None)), sharded_shapes)
        self.assertIn(((512, 512), ("model", None)), sharded_shapes)


if __name__ == "__main__":
    unittest.main()
