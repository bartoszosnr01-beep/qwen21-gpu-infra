"""Unit and contract tests for qwen_kaggle workflows, bootstrap, and notebook generation."""

from __future__ import annotations

import ast
import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_notebooks import build_all
from qwen_kaggle.bootstrap import ensure_physical_tpu_or_raise, find_input_asset, sequential_copy_file
from qwen_kaggle.constants import MODELS, PRESETS, TURBO_MANUAL_SIGMAS
from qwen_kaggle.prep_dataset import _resolve_kaggle_auth
from qwen_kaggle.smoke_test import _http_json
from qwen_kaggle.workflows import (
    GenerationRequest,
    build_qwen21_api_prompt,
    build_refine_prompt_graph,
    prepare_single_branch_prompt,
)


class TestConstantsAndWorkflows(unittest.TestCase):
    def test_presets_reference_valid_models(self) -> None:
        for preset_name, keys in PRESETS.items():
            self.assertGreater(len(keys), 0, preset_name)
            for key in keys:
                self.assertIn(key, MODELS)

    def test_turbo_t2i_prompt_graph(self) -> None:
        graph = build_qwen21_api_prompt(
            GenerationRequest(
                prompt="A crisp test photo",
                mode="t2i",
                profile="fast",
                width=1344,
                height=768,
                seed=1234,
                cache_device="cpu",
                cache_dtype="int8",
            )
        )
        self.assertEqual(graph["1"]["inputs"]["unet_name"], MODELS["unet_turbo"].filename)
        self.assertEqual(graph["2"]["class_type"], "QwenImage21Cache")
        self.assertEqual(graph["2"]["inputs"]["device"], "cpu")
        self.assertEqual(graph["2"]["inputs"]["dtype"], "int8")
        self.assertEqual(graph["3"]["inputs"]["type"], "qwen_image")
        self.assertEqual(graph["5"]["class_type"], "TextEncodeQwenImage21")
        self.assertNotIn("vae", graph["5"]["inputs"])
        self.assertEqual(graph["6"]["inputs"]["width"], 1344)
        self.assertEqual(graph["6"]["inputs"]["height"], 768)
        self.assertEqual(graph["9"]["class_type"], "ManualSigmas")
        self.assertEqual(graph["9"]["inputs"]["sigmas"], TURBO_MANUAL_SIGMAS)
        self.assertEqual(graph["10"]["inputs"]["cfg"], 1.0)
        self.assertEqual(graph["11"]["class_type"], "SamplerCustomAdvanced")
        self.assertEqual(graph["11"]["inputs"]["latent_image"], ["6", 0])

    def test_base_edit_prompt_graph_uses_image_latent_and_multiple_images(self) -> None:
        graph = build_qwen21_api_prompt(
            GenerationRequest(
                prompt="Combine <image1> and <image2>",
                mode="edit",
                profile="quality",
                image_names=("img_a.png", "img_b.png"),
                use_image_latent=True,
                steps=25,
                cfg=1.0,
            )
        )
        self.assertEqual(graph["1"]["inputs"]["unet_name"], MODELS["unet_base"].filename)
        self.assertEqual(graph["20"]["class_type"], "LoadImage")
        self.assertEqual(graph["20"]["inputs"]["image"], "img_a.png")
        self.assertEqual(graph["21"]["inputs"]["image"], "img_b.png")
        self.assertEqual(graph["5"]["inputs"]["images.image_1"], ["20", 0])
        self.assertEqual(graph["5"]["inputs"]["images.image_2"], ["21", 0])
        self.assertEqual(graph["5"]["inputs"]["vae"], ["4", 0])
        self.assertEqual(graph["11"]["class_type"], "KSampler")
        self.assertEqual(graph["11"]["inputs"]["steps"], 25)
        self.assertEqual(graph["11"]["inputs"]["latent_image"], ["5", 2])

    def test_edit_requires_images(self) -> None:
        with self.assertRaises(ValueError):
            build_qwen21_api_prompt(GenerationRequest(prompt="Edit", mode="edit"))

    def test_prepare_single_branch_prompt_normalizes_multi_image_tags(self) -> None:
        cleaned = prepare_single_branch_prompt(
            "Portrait blending <image1> and <image2> with both parents' features. Do not copy clothing from either reference."
        )
        self.assertIn("<image1>", cleaned)
        self.assertNotIn("<image2>", cleaned)
        self.assertNotIn("either reference", cleaned)
        self.assertIn("<image1>'s", cleaned)

        no_tag = prepare_single_branch_prompt("Photorealistic studio portrait")
        self.assertTrue(no_tag.startswith("Using the facial features of <image1>:"))

    def test_velocity_blend_builds_isolated_branches_with_area_strength_and_combine(self) -> None:
        graph = build_qwen21_api_prompt(
            GenerationRequest(
                prompt="Portrait blending <image1> and <image2> equally",
                mode="edit",
                profile="fast",
                image_names=("face_a.png", "face_b.png"),
                use_image_latent=True,  # must still force EmptyLatentImage for fusion
                identity_blend_mode="velocity_blend",
                identity_blend_ratio=0.4,
            )
        )
        self.assertNotIn("5", graph)
        self.assertEqual(graph["6"]["class_type"], "EmptyLatentImage")
        self.assertEqual(graph["11"]["inputs"]["latent_image"], ["6", 0])
        self.assertEqual(graph["20"]["inputs"]["image"], "face_a.png")
        self.assertEqual(graph["21"]["inputs"]["image"], "face_b.png")

        encoders = [n for n in graph.values() if n["class_type"] == "TextEncodeQwenImage21"]
        self.assertEqual(len(encoders), 2)
        for enc in encoders:
            self.assertIn("images.image_1", enc["inputs"])
            self.assertNotIn("images.image_2", enc["inputs"])
            self.assertNotIn("<image2>", enc["inputs"]["prompt"])

        strengths = [n["inputs"]["strength"] for n in graph.values() if n["class_type"] == "ConditioningSetAreaStrength"]
        self.assertEqual(strengths, [0.6, 0.6, 0.4, 0.4])

        combines = [nid for nid, n in graph.items() if n["class_type"] == "ConditioningCombine"]
        self.assertEqual(len(combines), 2)
        self.assertEqual(graph["10"]["inputs"]["positive"], [combines[0], 0])
        self.assertEqual(graph["10"]["inputs"]["negative"], [combines[1], 0])
        self.assertEqual(graph["12"]["class_type"], "VAEDecode")
        self.assertEqual(graph["13"]["class_type"], "SaveImage")

    def test_step_swap_builds_seamless_timestep_ranges_for_three_images(self) -> None:
        graph = build_qwen21_api_prompt(
            GenerationRequest(
                prompt="Blend <image1>, <image2>, and <image3>",
                mode="edit",
                profile="quality",
                image_names=("a.png", "b.png", "c.png"),
                identity_blend_mode="step_swap",
                identity_blend_ratio=0.5,
            )
        )
        ranges = [
            (n["inputs"]["start"], n["inputs"]["end"])
            for n in graph.values()
            if n["class_type"] == "ConditioningSetTimestepRange"
        ]
        # 3 branches x 2 (pos + neg) = 6 timestep range nodes, seamless from 0.0 to 1.0
        self.assertEqual(len(ranges), 6)
        pos_ranges = ranges[0::2]
        self.assertEqual(pos_ranges[0][0], 0.0)
        self.assertEqual(pos_ranges[0][1], pos_ranges[1][0])
        self.assertEqual(pos_ranges[1][1], pos_ranges[2][0])
        self.assertEqual(pos_ranges[2][1], 1.0)
        self.assertEqual(graph["11"]["class_type"], "KSampler")
        self.assertEqual(graph["11"]["inputs"]["latent_image"], ["6", 0])

    def test_refine_prompt_graph_stage1(self) -> None:
        graph = build_refine_prompt_graph(
            "Make the sky sunset orange",
            mode="edit",
            image_names=("photo.png",),
            thinking=True,
        )
        self.assertEqual(graph["1"]["inputs"]["clip_name"], MODELS["pe_i2i"].filename)
        self.assertEqual(graph["2"]["class_type"], "TextGenerateV2")
        self.assertEqual(graph["2"]["inputs"]["sampling_mode"], "on")
        self.assertEqual(graph["2"]["inputs"]["sampling_mode.seed"], 424242)
        self.assertTrue(graph["2"]["inputs"]["thinking"])
        self.assertEqual(graph["2"]["inputs"]["images.image1"], ["10", 0])
        self.assertEqual(graph["3"]["class_type"], "PreviewAny")
        self.assertEqual(graph["4"]["class_type"], "PreviewAny")
        self.assertEqual(graph["4"]["inputs"]["source"], ["2", 1])


class TestBootstrapAndNotebooks(unittest.TestCase):
    def test_find_and_sequential_copy_flat_and_nested(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp = root / "input" / "my-dataset"
            inp.mkdir(parents=True)
            src = inp / "test_model.safetensors"
            payload = b"qwenv21-binary-payload-" * 100
            src.write_bytes(payload)

            found = find_input_asset("test_model.safetensors", root / "input")
            self.assertEqual(found, src)

            dst = root / "tmp" / "models" / "test_model.safetensors"
            sequential_copy_file(src, dst)
            self.assertEqual(dst.read_bytes(), payload)

    def test_notebook_generation_and_python_syntax(self) -> None:
        import os
        import sys

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = build_all(root / "notebooks")
            self.assertEqual(len(paths), 4)
            for nb_path in paths:
                nb = json.loads(nb_path.read_text(encoding="utf-8"))
                self.assertEqual(nb["nbformat"], 4)
                code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
                self.assertGreaterEqual(len(code_cells), 2)
                for cell in code_cells:
                    src = "".join(cell["source"])
                    ast.parse(src)

                unpack_dir = root / f"unpack_{nb_path.stem}"
                first_src = "".join(code_cells[0]["source"])
                old_env = os.environ.get("QWEN_PKG_ROOT")
                old_path = list(sys.path)
                try:
                    os.environ["QWEN_PKG_ROOT"] = str(unpack_dir)
                    ns: dict = {}
                    exec(compile(first_src, str(nb_path), "exec"), ns)
                    for mod_name, b64 in ns["_MODULES_B64"].items():
                        mod_file = unpack_dir / "qwen_kaggle" / mod_name
                        self.assertTrue(mod_file.exists())
                        ast.parse(base64.b64decode(b64).decode("utf-8"), filename=mod_name)
                finally:
                    sys.path[:] = old_path
                    if old_env is None:
                        os.environ.pop("QWEN_PKG_ROOT", None)
                    else:
                        os.environ["QWEN_PKG_ROOT"] = old_env

    def test_http_json_handles_empty_free_response(self) -> None:
        empty_resp = mock.MagicMock()
        empty_resp.read.return_value = b""
        empty_ctx = mock.MagicMock()
        empty_ctx.__enter__.return_value = empty_resp
        empty_ctx.__exit__.return_value = False

        with mock.patch("qwen_kaggle.smoke_test.urllib.request.urlopen", return_value=empty_ctx):
            self.assertEqual(
                _http_json("http://127.0.0.1:8188/free", payload={"unload_models": True, "free_memory": True}),
                {},
            )

        json_resp = mock.MagicMock()
        json_resp.read.return_value = b'{"prompt_id": "abc"}'
        json_ctx = mock.MagicMock()
        json_ctx.__enter__.return_value = json_resp
        json_ctx.__exit__.return_value = False

        with mock.patch("qwen_kaggle.smoke_test.urllib.request.urlopen", return_value=json_ctx):
            self.assertEqual(
                _http_json("http://127.0.0.1:8188/prompt", payload={"prompt": {}}),
                {"prompt_id": "abc"},
            )

    def test_resolve_kaggle_auth_supports_api_token_and_legacy(self) -> None:
        import os

        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch("qwen_kaggle.prep_dataset.Path.home", return_value=Path(td)),
            mock.patch("qwen_kaggle.prep_dataset._read_kaggle_secret", return_value=""),
        ):
            with mock.patch.dict(
                os.environ,
                {"KAGGLE_API_TOKEN": "KGAT_test_token"},
                clear=True,
            ):
                owner, env = _resolve_kaggle_auth(owner="bartek")
                self.assertEqual(owner, "bartek")
                self.assertEqual(env, {"KAGGLE_API_TOKEN": "KGAT_test_token"})

            with mock.patch.dict(
                os.environ,
                {"KAGGLE_USERNAME": "bartek", "KAGGLE_KEY": "legacy123"},
                clear=True,
            ):
                owner, env = _resolve_kaggle_auth()
                self.assertEqual(owner, "bartek")
                self.assertEqual(env, {"KAGGLE_USERNAME": "bartek", "KAGGLE_KEY": "legacy123"})

    def test_setup_comfy_and_venv_uses_cached_archives_offline_and_rejects_traversal(self) -> None:
        import io
        import sys
        import tarfile
        import qwen_kaggle.bootstrap as bs
        from qwen_kaggle.constants import RUNTIME_COMFY_ARCHIVE, RUNTIME_META_FILENAME, RUNTIME_WHEELS_ARCHIVE

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_dir = root / "input" / "runtime-cache"
            input_dir.mkdir(parents=True)

            # Build safe ComfyUI tar archive
            comfy_src = root / "src_comfy" / "ComfyUI"
            comfy_src.mkdir(parents=True)
            (comfy_src / "main.py").write_text("# comfy\n", encoding="utf-8")
            (comfy_src / "requirements.txt").write_text("numpy\n", encoding="utf-8")
            with tarfile.open(input_dir / RUNTIME_COMFY_ARCHIVE, "w") as tf:
                tf.add(comfy_src / "main.py", arcname="ComfyUI/main.py")
                tf.add(comfy_src / "requirements.txt", arcname="ComfyUI/requirements.txt")

            # Build safe wheels tar + matching python_mm meta
            wheel_file = root / "numpy-0.0.1-py3-none-any.whl"
            wheel_file.write_bytes(b"wheel")
            with tarfile.open(input_dir / RUNTIME_WHEELS_ARCHIVE, "w") as tf:
                tf.add(wheel_file, arcname=wheel_file.name)
            (input_dir / RUNTIME_META_FILENAME).write_text(
                json.dumps({"python_mm": f"{sys.version_info[0]}.{sys.version_info[1]}"}),
                encoding="utf-8",
            )

            # Verify unsafe path traversal tar is rejected
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
                bs.setup_comfy_and_venv(comfy_dir, venv_dir, input_root=root / "input")

            self.assertTrue((comfy_dir / "main.py").exists())
            self.assertFalse(any("git" in c and "clone" in c for c in recorded_cmds))
            pip_cmds = [c for c in recorded_cmds if len(c) >= 3 and c[:3] == ["uv", "pip", "install"]]
            self.assertEqual(len(pip_cmds), 1)
            self.assertIn("--no-index", pip_cmds[0])
            self.assertIn("--find-links", pip_cmds[0])


class TestTpuRuntimeBridge(unittest.TestCase):
    def test_bf16_workflow_overrides(self) -> None:
        graph = build_qwen21_api_prompt(
            GenerationRequest(
                prompt="TPU native BF16 test",
                mode="t2i",
                profile="fast",
                cache_device="off",
                unet_key_override="unet_turbo_bf16",
                clip_key_override="clip_vl_bf16",
            )
        )
        self.assertEqual(graph["1"]["inputs"]["unet_name"], MODELS["unet_turbo_bf16"].filename)
        self.assertEqual(graph["3"]["inputs"]["clip_name"], MODELS["clip_vl_bf16"].filename)
        self.assertEqual(graph["2"]["inputs"]["device"], "off")

    def test_static_padded_sequence_preserves_exact_image_attention_and_rope(self) -> None:
        import torch
        import torch.nn as nn
        from qwen_kaggle.tpu_runtime import (
            build_static_padded_sequence,
            choose_sharding_spec,
            choose_static_context_bucket,
            shard_module_for_spmd,
        )

        self.assertEqual(choose_static_context_bucket(11), 256)
        self.assertEqual(choose_static_context_bucket(256), 256)
        self.assertEqual(choose_static_context_bucket(300), 512)

        torch.manual_seed(42)
        dim = 64

        class _DummyDiT(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.txt_in = nn.Linear(dim, dim, bias=False)
                self.img_in = nn.Linear(dim, dim, bias=False)

            def pe_embedder(self, ids: torch.Tensor) -> torch.Tensor:
                # Deterministic positional encoding matching EmbedND output layout (B, 1, N, ...)
                return ids.unsqueeze(1)

        dit = _DummyDiT().eval()
        x = torch.randn(1, dim, 4, 4)
        ctx_11 = torch.randn(1, 11, dim)
        ctx_23 = torch.randn(1, 23, dim)

        # Build padded sequences with bucket=32 for two different prompt lengths (11 vs 23)
        h_11, pe_11, segs_11 = build_static_padded_sequence(dit, x, ctx_11, [], [], buckets=(32,))
        h_23, pe_23, segs_23 = build_static_padded_sequence(dit, x, ctx_23, [], [], buckets=(32,))

        # Shapes must be 100% identical across different prompt lengths (32 text + 16 image = 48)
        self.assertEqual(h_11.shape, (1, 48, dim))
        self.assertEqual(h_23.shape, (1, 48, dim))
        self.assertEqual(pe_11.shape, pe_23.shape)

        # Compare against unpadded execution for ctx_11 (bucket=11 -> zero padding)
        h_unpad, pe_unpad, segs_unpad = build_static_padded_sequence(dit, x, ctx_11, [], [], buckets=(11,))
        self.assertTrue(torch.allclose(pe_11[:, 32:], pe_unpad[:, 11:], atol=1e-6))

        # Verify scaled_dot_product_attention on the image segment produces identical values
        q_proj = nn.Linear(dim, dim, bias=False)
        k_proj = nn.Linear(dim, dim, bias=False)
        v_proj = nn.Linear(dim, dim, bias=False)

        def _run_seg_attn(h: torch.Tensor, segs: list) -> torch.Tensor:
            q, k, v = q_proj(h), k_proj(h), v_proj(h)
            outs = []
            for start, end, mask in segs:
                q_s = q[:, start:end]
                k_s = k[:, :end]
                v_s = v[:, :end]
                attn_out = torch.nn.functional.scaled_dot_product_attention(
                    q_s, k_s, v_s, attn_mask=mask
                )
                outs.append(attn_out)
            return torch.cat(outs, dim=1)

        out_pad = _run_seg_attn(h_11, segs_11)
        out_unpad = _run_seg_attn(h_unpad, segs_unpad)
        # Real text tokens (0..11) and target image tokens (last 16) must match unpadded execution
        self.assertTrue(torch.allclose(out_pad[:, :11], out_unpad[:, :11], atol=1e-5))
        self.assertTrue(torch.allclose(out_pad[:, 32:], out_unpad[:, 11:], atol=1e-5))

        # Verify 8-chip SPMD sharding spec and dequantization helper
        self.assertEqual(choose_sharding_spec((4096, 4096), 8), ("model", None))
        self.assertIsNone(choose_sharding_spec((64,), 8))

        class _FakeQuantParam:
            def __init__(self, t: torch.Tensor) -> None:
                self._t = t

            def dequantize(self) -> torch.Tensor:
                return self._t

        block = nn.Linear(512, 512, bias=False)
        block._buffers["q_buf"] = _FakeQuantParam(torch.ones(512, 512, dtype=torch.float32))  # type: ignore[assignment]
        marked: list[tuple[tuple[int, ...], Any]] = []
        stats = shard_module_for_spmd(
            block,
            xla_device=torch.device("cpu"),
            mesh="mesh8",
            num_chips=8,
            mark_sharding_fn=lambda t, m, s: marked.append((tuple(t.shape), s)),
        )
        self.assertEqual(stats["dequantized"], 1)
        self.assertEqual(stats["sharded"], 2)
        self.assertEqual(block.weight.dtype, torch.bfloat16)

    def test_ensure_physical_tpu_or_raise_fails_fast_on_cpu_fallback(self) -> None:
        import qwen_kaggle.bootstrap as bs

        fake_diag = {"memory": {"ram_total_gb": 31.35}, "cpu_count": 4}
        with mock.patch.object(bs, "probe_outer_tpu_jax", return_value="OUTER_JAX_RESULT: 1 cpu cpu"):
            with self.assertRaises(RuntimeError) as ctx:
                bs.ensure_physical_tpu_or_raise(fake_diag)
            self.assertIn("WITHOUT physical TPU", str(ctx.exception))

        with mock.patch.object(bs, "probe_outer_tpu_jax", return_value="OUTER_JAX_MISSING"):
            self.assertEqual(bs.ensure_physical_tpu_or_raise(fake_diag), "OUTER_JAX_MISSING")

        with mock.patch.object(
            bs,
            "probe_outer_tpu_jax",
            return_value="OUTER_JAX_RESULT: 8 tpu TPU v5 lite",
        ):
            res = bs.ensure_physical_tpu_or_raise(fake_diag)
            self.assertIn("8 tpu", res)


if __name__ == "__main__":
    unittest.main()
