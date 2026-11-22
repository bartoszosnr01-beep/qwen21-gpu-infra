"""Async contract and end-to-end mocked dual-worker tests for `qwen_kaggle.gateway`."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
import struct
import tempfile
import unittest
from typing import Any

from aiohttp import WSMsgType, web
from aiohttp.test_utils import AioHTTPTestCase, TestServer

from qwen_kaggle.constants import MODELS, TAESD_DECODER_FILENAME, TAESD_DECODER_SUBDIR
from qwen_kaggle.gateway import (
    BINARY_EVENT_PREVIEW_IMAGE,
    BINARY_EVENT_PREVIEW_IMAGE_WITH_METADATA,
    GatewayConfig,
    QwenGateway,
    create_gateway_app,
    parse_comfy_preview_frame,
    parse_png_dimensions,
)

TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
    b"\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82"
)


class _FakeComfyWorker:
    """In-memory mock ComfyUI HTTP + WebSocket worker verifying gateway requests."""

    def __init__(self, gpu_index: int, delay_s: float = 0.03) -> None:
        self.gpu_index = gpu_index
        self.delay_s = delay_s
        self.free_calls: list[dict[str, Any]] = []
        self.interrupt_calls: int = 0
        self.queue_clear_calls: int = 0
        self.uploaded_names: list[str] = []
        self.submitted_prompts: list[dict[str, Any]] = []
        self.history: dict[str, dict[str, Any]] = {}
        self.queue_running: list[Any] = []
        self.queue_pending: list[Any] = []
        self.interrupted_prompts: set[str] = set()
        self._ws_clients: dict[str, web.WebSocketResponse] = {}
        self.suppress_ws_events: bool = False
        self.stall_event: asyncio.Event | None = None
        self.track_queue: bool = False
        self.reject_prompt_if_busy: bool = False
        self.interrupt_clear_delay_s: float = 0.0
        self.continuous_ws_ticks: bool = False
        self.app = self._build_app()

    def _build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/system_stats", self._system_stats)
        app.router.add_post("/free", self._free)
        app.router.add_post("/interrupt", self._interrupt)
        app.router.add_get("/queue", self._get_queue)
        app.router.add_post("/queue", self._post_queue)
        app.router.add_post("/upload/image", self._upload)
        app.router.add_post("/prompt", self._prompt)
        app.router.add_get("/history/{prompt_id}", self._history)
        app.router.add_get("/view", self._view)
        app.router.add_get("/ws", self._ws)
        return app

    async def _system_stats(self, _req: web.Request) -> web.Response:
        return web.json_response({"system": {"gpu_index": self.gpu_index}})

    async def _free(self, req: web.Request) -> web.Response:
        self.free_calls.append(await req.json())
        # Deliberately return empty HTTP 200 body matching real ComfyUI behavior.
        return web.Response(status=200, body=b"")

    async def _interrupt(self, _req: web.Request) -> web.Response:
        self.interrupt_calls += 1
        if self.track_queue:
            self.interrupted_prompts.update(str(item[1]) for item in self.queue_running if len(item) >= 2)

            async def _drain() -> None:
                if self.interrupt_clear_delay_s > 0:
                    await asyncio.sleep(self.interrupt_clear_delay_s)
                self.queue_running.clear()
                self.queue_pending.clear()
                if self.stall_event is not None:
                    self.stall_event.set()
            asyncio.create_task(_drain())
        return web.Response(status=200, body=b"")

    async def _get_queue(self, _req: web.Request) -> web.Response:
        return web.json_response({"queue_running": list(self.queue_running), "queue_pending": list(self.queue_pending)})

    async def _post_queue(self, req: web.Request) -> web.Response:
        body = await req.json()
        if body.get("clear"):
            self.queue_clear_calls += 1
            self.queue_pending.clear()
        return web.Response(status=200, body=b"")

    async def _upload(self, req: web.Request) -> web.Response:
        reader = await req.multipart()
        fname = "uploaded.png"
        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name == "image":
                fname = part.filename or fname
                await part.read()
        self.uploaded_names.append(fname)
        return web.json_response({"name": fname, "subfolder": "", "type": "input"})

    async def _ws(self, req: web.Request) -> web.WebSocketResponse:
        client_id = req.query.get("clientId", "")
        ws = web.WebSocketResponse()
        await ws.prepare(req)
        self._ws_clients[client_id] = ws
        try:
            async for msg in ws:
                if msg.type == WSMsgType.ERROR:
                    break
        finally:
            self._ws_clients.pop(client_id, None)
        return ws

    async def _prompt(self, req: web.Request) -> web.Response:
        if self.reject_prompt_if_busy and self.queue_running:
            return web.Response(status=500, text="ComfyUI still running previous interrupted prompt")
        body = await req.json()
        prompt_id = f"p{self.gpu_index}_{len(self.submitted_prompts) + 1}"
        prompt_graph = body["prompt"]
        client_id = body.get("client_id", "")
        self.submitted_prompts.append(prompt_graph)
        if self.track_queue:
            self.queue_running.append([0, prompt_id])
        is_refine = prompt_graph.get("2", {}).get("class_type") == "TextGenerateV2"
        asyncio.create_task(self._simulate_execution(prompt_id, client_id, is_refine, prompt_graph))
        return web.json_response({"prompt_id": prompt_id})

    async def _simulate_execution(
        self,
        prompt_id: str,
        client_id: str,
        is_refine: bool,
        prompt_graph: dict[str, Any],
    ) -> None:
        ws = self._ws_clients.get(client_id)
        if self.stall_event is not None:
            while not self.stall_event.is_set():
                if prompt_id in self.interrupted_prompts:
                    return
                if self.continuous_ws_ticks and ws is not None and not ws.closed:
                    await ws.send_json({"type": "progress", "data": {"prompt_id": prompt_id, "value": 1, "max": 25}})
                await asyncio.sleep(0.04)
            if prompt_id in self.interrupted_prompts:
                return
        await asyncio.sleep(self.delay_s)
        if is_refine:
            orig_prompt = prompt_graph["2"]["inputs"]["prompt"]
            self.history[prompt_id] = {
                "status": {"status_str": "success"},
                "outputs": {
                    "3": {"text": [f"Refined: {orig_prompt}, studio lighting"]},
                    "4": {"text": ["Analyzed subject and expanded lighting details."]},
                },
            }
        else:
            if not self.suppress_ws_events and ws is not None and not ws.closed:
                await ws.send_json({"type": "executing", "data": {"prompt_id": prompt_id, "node": "11"}})
                await ws.send_json(
                    {
                        "type": "progress",
                        "data": {"prompt_id": prompt_id, "value": 1, "max": 8},
                    }
                )
                await asyncio.sleep(self.delay_s)
                await ws.send_json(
                    {
                        "type": "progress",
                        "data": {"prompt_id": prompt_id, "value": 8, "max": 8},
                    }
                )
                preview_pkt = (
                    struct.pack(">II", BINARY_EVENT_PREVIEW_IMAGE, 2) + TINY_PNG
                )
                await ws.send_bytes(preview_pkt)
                await ws.send_json({"type": "executing", "data": {"prompt_id": prompt_id, "node": "12"}})
                await asyncio.sleep(self.delay_s)
            self.history[prompt_id] = {
                "status": {"status_str": "success"},
                "outputs": {
                    "13": {
                        "images": [
                            {"filename": f"out_gpu{self.gpu_index}_{prompt_id}.png", "subfolder": "", "type": "output"}
                        ]
                    }
                },
            }
        if self.track_queue:
            self.queue_running = [item for item in self.queue_running if item[1] != prompt_id]
        if not self.suppress_ws_events and ws is not None and not ws.closed:
            await ws.send_json({"type": "executing", "data": {"prompt_id": prompt_id, "node": None}})

    async def _history(self, req: web.Request) -> web.Response:
        pid = req.match_info["prompt_id"]
        if pid not in self.history:
            return web.json_response({})
        return web.json_response({pid: self.history[pid]})

    async def _view(self, _req: web.Request) -> web.Response:
        return web.Response(body=TINY_PNG, content_type="image/png")


def _seed_fake_comfy_models(comfy_dir: Path, keys: tuple[str, ...]) -> None:
    for key in keys:
        spec = MODELS[key]
        path = comfy_dir / "models" / spec.subdir / spec.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights")
    taesd = comfy_dir / "models" / TAESD_DECODER_SUBDIR / TAESD_DECODER_FILENAME
    taesd.parent.mkdir(parents=True, exist_ok=True)
    taesd.write_bytes(b"taesd")


class TestPreviewParser(unittest.TestCase):
    def test_parses_v1_and_v4_preview_frames(self) -> None:
        v1 = struct.pack(">II", BINARY_EVENT_PREVIEW_IMAGE, 2) + TINY_PNG
        self.assertEqual(parse_comfy_preview_frame(v1), ("image/png", TINY_PNG))

        meta = json.dumps({"image_type": "image/jpeg"}).encode("utf-8")
        v4 = struct.pack(">II", BINARY_EVENT_PREVIEW_IMAGE_WITH_METADATA, len(meta)) + meta + b"JPEGDATA"
        self.assertEqual(parse_comfy_preview_frame(v4), ("image/jpeg", b"JPEGDATA"))

    def test_parses_png_dimensions(self) -> None:
        self.assertEqual(parse_png_dimensions(TINY_PNG), (1, 1))
        fake_ihdr = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + struct.pack(">II", 1344, 768) + b"\x08\x02\x00\x00\x00"
        self.assertEqual(parse_png_dimensions(fake_ihdr), (1344, 768))
        self.assertIsNone(parse_png_dimensions(b"not-a-png"))


class TestGatewayEndpoints(AioHTTPTestCase):
    async def get_application(self) -> web.Application:
        self._tmp = tempfile.TemporaryDirectory()
        self.comfy_dir = Path(self._tmp.name) / "ComfyUI"
        _seed_fake_comfy_models(self.comfy_dir, ("unet_turbo", "clip_vl", "vae", "pe_t2i", "pe_i2i"))

        self.worker0 = _FakeComfyWorker(0, delay_s=0.04)
        self.worker1 = _FakeComfyWorker(1, delay_s=0.04)
        self.w_srv0 = TestServer(self.worker0.app)
        self.w_srv1 = TestServer(self.worker1.app)
        await self.w_srv0.start_server()
        await self.w_srv1.start_server()

        url0 = str(self.w_srv0.make_url("")).rstrip("/")
        url1 = str(self.w_srv1.make_url("")).rstrip("/")
        cfg = GatewayConfig(
            comfy_dir=self.comfy_dir,
            venv_dir=Path(self._tmp.name) / "venv",
            input_dir=Path(self._tmp.name) / "input",
            log_dir=Path(self._tmp.name) / "logs",
            bearer_token="secret-kaggle-token",
            worker_urls=(url0, url1),
            ws_poll_interval_s=0.08,
        )
        self.gateway = QwenGateway(cfg)
        return create_gateway_app(self.gateway)

    async def tearDownAsync(self) -> None:
        await self.w_srv0.close()
        await self.w_srv1.close()
        await super().tearDownAsync()
        self._tmp.cleanup()

    def _auth(self) -> dict[str, str]:
        return {"Authorization": "Bearer secret-kaggle-token"}

    async def test_auth_fail_closed_and_status(self) -> None:
        resp_health = await self.client.get("/health")
        self.assertEqual(resp_health.status, 200)

        resp_unauth = await self.client.get("/v1/status")
        self.assertEqual(resp_unauth.status, 401)

        resp_bad = await self.client.get("/v1/status", headers={"Authorization": "Bearer wrong"})
        self.assertEqual(resp_bad.status, 401)

        resp_ok = await self.client.get("/v1/status", headers=self._auth())
        self.assertEqual(resp_ok.status, 200)
        data = await resp_ok.json()
        self.assertEqual(len(data["workers"]), 2)
        self.assertTrue(data["models"]["unet_turbo"]["installed"])
        self.assertFalse(data["models"]["unet_base"]["installed"])

    async def test_two_stage_refine_free_generate_and_sse(self) -> None:
        resp = await self.client.post(
            "/v1/jobs",
            headers=self._auth(),
            json={
                "action": "generate",
                "mode": "t2i",
                "profile": "fast",
                "prompt": "A red mug",
                "enhance_prompt": True,
                "thinking": True,
                "seed": -1,
            },
        )
        self.assertEqual(resp.status, 202)
        job = (await resp.json())["job"]
        job_id = job["id"]
        self.assertGreater(job["seed"], 0)

        # Follow SSE stream until completion
        sse_resp = await self.client.get(f"/v1/jobs/{job_id}/events", headers=self._auth())
        self.assertEqual(sse_resp.status, 200)
        body = (await sse_resp.read()).decode("utf-8")
        self.assertIn('"status": "completed"', body)
        self.assertIn("Refined: A red mug, studio lighting", body)

        job_resp = await self.client.get(f"/v1/jobs/{job_id}", headers=self._auth())
        done = (await job_resp.json())["job"]
        self.assertEqual(done["status"], "completed")
        self.assertTrue(done["image_available"])
        self.assertTrue(done["preview_data_url"].startswith("data:image/png;base64,"))
        self.assertEqual(done["thinking_text"], "Analyzed subject and expanded lighting details.")
        self.assertEqual(done["effective_steps"], 8)
        self.assertEqual(done["output_width"], 1)
        self.assertEqual(done["output_height"], 1)
        self.assertFalse(done["warm_worker"])
        self.assertIsNotNone(done["stage1_s"])
        self.assertIsNotNone(done["load_s"])
        self.assertIsNotNone(done["sampling_s"])
        self.assertIsNotNone(done["vae_s"])
        self.assertIsNotNone(done["sec_per_step"])

        # Verify Stage-1 called POST /free on that worker before Stage-2
        assigned = self.worker0 if done["gpu_index"] == 0 else self.worker1
        self.assertEqual(len(assigned.free_calls), 1)
        self.assertEqual(len(assigned.submitted_prompts), 2)

        img_resp = await self.client.get(f"/v1/jobs/{job_id}/image", headers=self._auth())
        self.assertEqual(img_resp.status, 200)
        self.assertEqual(await img_resp.read(), TINY_PNG)

        # Follow-up Turbo run on same GPU without Stage-1 /free should register as warm_worker=True
        warm_resp = await self.client.post(
            "/v1/jobs",
            headers=self._auth(),
            json={
                "mode": "t2i",
                "profile": "fast",
                "prompt": "Warm run check",
                "preferred_gpu": done["gpu_index"],
            },
        )
        self.assertEqual(warm_resp.status, 202)
        warm_id = (await warm_resp.json())["job"]["id"]
        for _ in range(50):
            warm_done = (await (await self.client.get(f"/v1/jobs/{warm_id}", headers=self._auth())).json())["job"]
            if warm_done["status"] == "completed":
                break
            await asyncio.sleep(0.03)
        self.assertEqual(warm_done["status"], "completed")
        self.assertTrue(warm_done["warm_worker"])
        self.assertIsNone(warm_done["stage1_s"])

    async def test_parallel_dual_gpu_dispatch_and_edit_upload(self) -> None:
        img_b64 = base64.b64encode(TINY_PNG).decode("ascii")
        r0 = await self.client.post(
            "/v1/jobs",
            headers=self._auth(),
            json={"mode": "t2i", "profile": "fast", "prompt": "Job zero", "preferred_gpu": 0},
        )
        r1 = await self.client.post(
            "/v1/jobs",
            headers=self._auth(),
            json={
                "mode": "edit",
                "profile": "fast",
                "prompt": "Make <image1> golden",
                "images": [{"filename": "input.png", "data_url": f"data:image/png;base64,{img_b64}"}],
                "preferred_gpu": 1,
            },
        )
        self.assertEqual(r0.status, 202)
        self.assertEqual(r1.status, 202)
        j0 = (await r0.json())["job"]["id"]
        j1 = (await r1.json())["job"]["id"]

        for _ in range(50):
            s0 = (await (await self.client.get(f"/v1/jobs/{j0}", headers=self._auth())).json())["job"]
            s1 = (await (await self.client.get(f"/v1/jobs/{j1}", headers=self._auth())).json())["job"]
            if s0["status"] == "completed" and s1["status"] == "completed":
                break
            await asyncio.sleep(0.03)

        self.assertEqual(s0["gpu_index"], 0)
        self.assertEqual(s1["gpu_index"], 1)
        self.assertEqual(len(self.worker1.uploaded_names), 1)

    async def test_identity_blend_mode_validation_and_dispatch(self) -> None:
        img_b64 = base64.b64encode(TINY_PNG).decode("ascii")
        imgs = [
            {"filename": "ref1.png", "data_url": f"data:image/png;base64,{img_b64}"},
            {"filename": "ref2.png", "data_url": f"data:image/png;base64,{img_b64}"},
        ]

        bad_mode = await self.client.post(
            "/v1/jobs",
            headers=self._auth(),
            json={
                "mode": "edit",
                "profile": "fast",
                "prompt": "Blend <image1> and <image2>",
                "images": imgs,
                "identity_blend_mode": "invalid_mode",
            },
        )
        self.assertEqual(bad_mode.status, 400)

        bad_ratio = await self.client.post(
            "/v1/jobs",
            headers=self._auth(),
            json={
                "mode": "edit",
                "profile": "fast",
                "prompt": "Blend <image1> and <image2>",
                "images": imgs,
                "identity_blend_mode": "velocity_blend",
                "identity_blend_ratio": 0.99,
            },
        )
        self.assertEqual(bad_ratio.status, 400)

        ok_resp = await self.client.post(
            "/v1/jobs",
            headers=self._auth(),
            json={
                "mode": "edit",
                "profile": "fast",
                "prompt": "Blend <image1> and <image2>",
                "images": imgs,
                "use_image_latent": False,
                "identity_blend_mode": "velocity_blend",
                "identity_blend_ratio": 0.5,
                "preferred_gpu": 0,
            },
        )
        self.assertEqual(ok_resp.status, 202)
        job_id = (await ok_resp.json())["job"]["id"]
        for _ in range(50):
            done = (await (await self.client.get(f"/v1/jobs/{job_id}", headers=self._auth())).json())["job"]
            if done["status"] == "completed":
                break
            await asyncio.sleep(0.03)
        self.assertEqual(done["status"], "completed")
        self.assertEqual(done["identity_blend_mode"], "velocity_blend")
        self.assertEqual(done["identity_blend_ratio"], 0.5)
        submitted = self.worker0.submitted_prompts[-1]
        self.assertTrue(any(n["class_type"] == "ConditioningSetAreaStrength" for n in submitted.values()))
        self.assertTrue(any(n["class_type"] == "ConditioningCombine" for n in submitted.values()))

    async def test_status_capabilities_and_face_crop_normalization_dispatch(self) -> None:
        from unittest import mock
        from qwen_kaggle.gateway import (
            _select_primary_face,
            compute_face_centered_square_crop,
        )

        # 1. Pure geometric parity check: wide portrait (30% face) vs tight close-up (54% face)
        #    must yield equal face-to-crop ratio (~0.44) so neither face dominates encoder tokens.
        wide_crop = compute_face_centered_square_crop(1024, 1024, (360, 240, 300, 300), target_face_ratio=0.44)
        tight_crop = compute_face_centered_square_crop(230, 350, (70, 80, 125, 125), target_face_ratio=0.44)
        wide_side = wide_crop[2] - wide_crop[0]
        tight_side = tight_crop[2] - tight_crop[0]
        self.assertEqual(wide_side, wide_crop[3] - wide_crop[1])
        self.assertEqual(tight_side, tight_crop[3] - tight_crop[1])
        self.assertAlmostEqual(300.0 / wide_side, 0.44, places=2)
        self.assertAlmostEqual(125.0 / tight_side, 0.44, places=2)
        # Wide crop stays strictly inside image bounds; tight crop allows gentle out-of-bounds pad
        self.assertGreaterEqual(wide_crop[0], 0)
        self.assertLessEqual(wide_crop[2], 1024)
        self.assertLess(tight_crop[0], 0)

        # Fallback when no face is detected returns clean 1:1 upper-biased cover crop
        fallback = compute_face_centered_square_crop(400, 800, None)
        self.assertEqual(fallback, (0, 100, 400, 500))

        # Primary face selector favors large central subject over tiny edge/mirror face
        chosen = _select_primary_face([(5, 5, 60, 60), (200, 180, 140, 140)], 600, 600)
        self.assertEqual(chosen, (200, 180, 140, 140))

        # 2. Gateway status advertises capabilities
        st = await (await self.client.get("/v1/status", headers=self._auth())).json()
        caps = st.get("capabilities") or {}
        self.assertIn("velocity_blend", caps.get("identity_blend_modes", []))
        self.assertIn("step_swap", caps.get("identity_blend_modes", []))
        self.assertTrue(caps.get("face_scale_normalization"))
        self.assertTrue(caps.get("idle_extend"))

        # 3. normalize_face_crops=True invokes normalize_reference_face_crops before uploading
        img_b64 = base64.b64encode(TINY_PNG).decode("ascii")
        imgs = [
            {"filename": "ref1.png", "data_url": f"data:image/png;base64,{img_b64}"},
            {"filename": "ref2.png", "data_url": f"data:image/png;base64,{img_b64}"},
        ]
        norm_called: list[int] = []

        def fake_norm(items: list[tuple[str, bytes]]) -> tuple[list[tuple[str, bytes]], int]:
            norm_called.append(len(items))
            return [(f"fc_{idx}.png", raw) for idx, (_, raw) in enumerate(items)], len(items)

        with mock.patch("qwen_kaggle.gateway.normalize_reference_face_crops", side_effect=fake_norm):
            resp = await self.client.post(
                "/v1/jobs",
                headers=self._auth(),
                json={
                    "mode": "edit",
                    "profile": "fast",
                    "prompt": "Blend <image1> and <image2>",
                    "images": imgs,
                    "use_image_latent": False,
                    "identity_blend_mode": "velocity_blend",
                    "identity_blend_ratio": 0.4,
                    "normalize_face_crops": True,
                    "preferred_gpu": 0,
                },
            )
            self.assertEqual(resp.status, 202)
            job_id = (await resp.json())["job"]["id"]
            for _ in range(50):
                done = (await (await self.client.get(f"/v1/jobs/{job_id}", headers=self._auth())).json())["job"]
                if done["status"] == "completed":
                    break
                await asyncio.sleep(0.03)

        self.assertEqual(norm_called, [2])
        self.assertTrue(done["normalize_face_crops"])
        self.assertTrue(done["face_crops_applied"])
        self.assertTrue(any("fc_0.png" in name for name in self.worker0.uploaded_names))
        self.assertTrue(any("fc_1.png" in name for name in self.worker0.uploaded_names))

    async def test_missing_base_model_returns_400(self) -> None:
        resp = await self.client.post(
            "/v1/jobs",
            headers=self._auth(),
            json={"mode": "t2i", "profile": "quality", "prompt": "Test base quality"},
        )
        self.assertEqual(resp.status, 400)
        payload = await resp.json()
        self.assertEqual(payload["error"], "MISSING_MODELS")
        self.assertIn("unet_base", payload["missing_models"])

    async def test_ensure_models_endpoint_matches_bootstrap_signature(self) -> None:
        import inspect
        from unittest import mock
        from qwen_kaggle import bootstrap

        sig = inspect.signature(bootstrap.ensure_models_and_templates)
        recorded: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def fake_ensure(*args: Any, **kwargs: Any) -> dict[str, str]:
            sig.bind(*args, **kwargs)
            recorded.append((args, kwargs))
            return {}

        with mock.patch("qwen_kaggle.gateway.ensure_models_and_templates", side_effect=fake_ensure):
            resp = await self.client.post(
                "/v1/models/ensure",
                headers=self._auth(),
                json={"preset": "full_i2i"},
            )
            self.assertEqual(resp.status, 202)
            for _ in range(30):
                if recorded:
                    break
                await asyncio.sleep(0.02)
        self.assertEqual(len(recorded), 1)

    async def test_recovers_from_silent_websocket_via_history_poll_without_deadlocking_second_gpu(self) -> None:
        self.worker0.suppress_ws_events = True
        r0 = await self.client.post(
            "/v1/jobs",
            headers=self._auth(),
            json={"mode": "t2i", "profile": "fast", "prompt": "Silent WS on GPU0", "preferred_gpu": 0},
        )
        r1 = await self.client.post(
            "/v1/jobs",
            headers=self._auth(),
            json={"mode": "t2i", "profile": "fast", "prompt": "Waiting on GPU1", "preferred_gpu": 1},
        )
        j0 = (await r0.json())["job"]["id"]
        j1 = (await r1.json())["job"]["id"]

        for _ in range(60):
            s0 = (await (await self.client.get(f"/v1/jobs/{j0}", headers=self._auth())).json())["job"]
            s1 = (await (await self.client.get(f"/v1/jobs/{j1}", headers=self._auth())).json())["job"]
            if s0["status"] == "completed" and s1["status"] == "completed":
                break
            await asyncio.sleep(0.03)

        self.assertEqual(s0["status"], "completed")
        self.assertEqual(s1["status"], "completed")

    async def test_cancel_running_and_load_lock_waiting_jobs_unblocks_workers(self) -> None:
        self.worker0.stall_event = asyncio.Event()
        self.worker0.suppress_ws_events = True

        r0 = await self.client.post(
            "/v1/jobs",
            headers=self._auth(),
            json={"mode": "t2i", "profile": "fast", "prompt": "Stalled on GPU0", "preferred_gpu": 0},
        )
        j0 = (await r0.json())["job"]["id"]
        await asyncio.sleep(0.05)

        r1 = await self.client.post(
            "/v1/jobs",
            headers=self._auth(),
            json={"mode": "t2i", "profile": "fast", "prompt": "Blocked on load lock GPU1", "preferred_gpu": 1},
        )
        j1 = (await r1.json())["job"]["id"]
        await asyncio.sleep(0.05)

        # Cancel GPU1 job while it is waiting on _load_lock, then cancel GPU0 job while stuck on WS
        c1 = await self.client.post(f"/v1/jobs/{j1}/cancel", headers=self._auth())
        self.assertEqual(c1.status, 200)
        c0 = await self.client.post(f"/v1/jobs/{j0}/cancel", headers=self._auth())
        self.assertEqual(c0.status, 200)

        for _ in range(40):
            st = await (await self.client.get("/v1/status", headers=self._auth())).json()
            if all(w["state"] == "idle" for w in st["workers"]):
                break
            await asyncio.sleep(0.03)

        s0 = (await (await self.client.get(f"/v1/jobs/{j0}", headers=self._auth())).json())["job"]
        s1 = (await (await self.client.get(f"/v1/jobs/{j1}", headers=self._auth())).json())["job"]
        self.assertEqual(s0["status"], "cancelled")
        self.assertEqual(s1["status"], "cancelled")
        st = await (await self.client.get("/v1/status", headers=self._auth())).json()
        self.assertTrue(all(w["state"] == "idle" for w in st["workers"]))
        self.worker0.stall_event.set()

    async def test_shutdown_requires_auth_and_cancels_active_work(self) -> None:
        unauth = await self.client.post("/v1/shutdown")
        self.assertEqual(unauth.status, 401)
        self.assertFalse(self.gateway.shutdown_event.is_set())

        resp = await self.client.post("/v1/shutdown", headers=self._auth(), json={"reason": "bos stop!"})
        self.assertEqual(resp.status, 202)
        data = await resp.json()
        self.assertTrue(data["shutting_down"])
        self.assertFalse(data["already_requested"])
        self.assertEqual(data["reason"], "bos_stop_")
        self.assertTrue(self.gateway.shutdown_event.is_set())

        again = await (await self.client.post("/v1/shutdown", headers=self._auth())).json()
        self.assertTrue(again["already_requested"])

        st = await (await self.client.get("/v1/status", headers=self._auth())).json()
        self.assertTrue(st["idle"]["shutting_down"])
        self.assertEqual(st["idle"]["timeout_s"], 600.0)

    async def test_status_polling_does_not_reset_idle_clock(self) -> None:
        self.gateway._last_busy_mono -= 100.0
        for _ in range(3):
            st = await (await self.client.get("/v1/status", headers=self._auth())).json()
        self.assertGreaterEqual(st["idle"]["idle_s"], 100.0)
        self.assertLessEqual(st["idle"]["shutdown_in_s"], 500.0)
        self.assertFalse(st["idle"]["busy"])

    async def test_idle_watchdog_fires_only_when_idle(self) -> None:
        self.gateway.config.idle_timeout_s = 0.2
        # Busy worker keeps the clock reset.
        self.gateway.workers[0].state = "busy"
        task = asyncio.create_task(self.gateway.idle_watchdog())
        await asyncio.sleep(0.5)
        self.assertFalse(self.gateway.shutdown_event.is_set())
        self.gateway.workers[0].state = "idle"
        await asyncio.wait_for(self.gateway.shutdown_event.wait(), timeout=3.0)
        await task
        self.assertEqual(self.gateway.shutdown_reason, "idle_timeout_0s")

    async def test_idle_extend_requires_auth_validates_and_adds_time(self) -> None:
        unauth = await self.client.post("/v1/idle/extend")
        self.assertEqual(unauth.status, 401)

        for bad_body in (
            {"seconds": 0},
            {"seconds": -10},
            {"seconds": True},
            {"seconds": 99999},
        ):
            bad_resp = await self.client.post("/v1/idle/extend", headers=self._auth(), json=bad_body)
            self.assertEqual(bad_resp.status, 400, msg=str(bad_body))

        bad_json = await self.client.post(
            "/v1/idle/extend",
            headers={**self._auth(), "Content-Type": "application/json"},
            data="{not-json",
        )
        self.assertEqual(bad_json.status, 400)

        # Simulate 550s idle out of 600s -> <= 50s remaining.
        self.gateway._last_busy_mono -= 550.0
        st_before = await (await self.client.get("/v1/status", headers=self._auth())).json()
        self.assertLessEqual(st_before["idle"]["shutdown_in_s"], 50.0)

        ext_resp = await self.client.post("/v1/idle/extend", headers=self._auth(), json={})
        self.assertEqual(ext_resp.status, 200)
        ext_data = await ext_resp.json()
        self.assertTrue(ext_data["ok"])
        self.assertEqual(ext_data["extended_s"], 600.0)
        self.assertGreaterEqual(ext_data["idle"]["shutdown_in_s"], 640.0)
        self.assertLess(ext_data["idle"]["idle_s"], 5.0)

        # Real activity resets the bonus back to base timeout_s.
        self.gateway.touch_activity()
        self.assertEqual(self.gateway._idle_bonus_s, 0.0)
        st_after_activity = await (await self.client.get("/v1/status", headers=self._auth())).json()
        self.assertLessEqual(st_after_activity["idle"]["shutdown_in_s"], 600.0)

        # Once shutdown is requested, extend returns 409 SHUTTING_DOWN.
        await self.client.post("/v1/shutdown", headers=self._auth(), json={"reason": "stop_now"})
        after_stop = await self.client.post("/v1/idle/extend", headers=self._auth(), json={})
        self.assertEqual(after_stop.status, 409)
        after_stop_data = await after_stop.json()
        self.assertFalse(after_stop_data["ok"])
        self.assertEqual(after_stop_data["error"], "SHUTTING_DOWN")

    async def test_idle_extend_delays_idle_watchdog_shutdown(self) -> None:
        self.gateway.config.idle_timeout_s = 0.25
        self.gateway.touch_activity()
        task = asyncio.create_task(self.gateway.idle_watchdog())
        try:
            await asyncio.sleep(0.12)
            ext_resp = await self.client.post(
                "/v1/idle/extend",
                headers=self._auth(),
                json={"seconds": 0.45},
            )
            self.assertEqual(ext_resp.status, 200)
            await asyncio.sleep(0.25)
            self.assertFalse(self.gateway.shutdown_event.is_set())
            await asyncio.wait_for(self.gateway.shutdown_event.wait(), timeout=2.0)
            await task
            self.assertEqual(self.gateway.shutdown_reason, "idle_timeout_0s")
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def test_session_http_client_has_no_600s_total_timeout(self) -> None:
        default_cfg = GatewayConfig()
        self.assertEqual(default_cfg.prompt_timeout_s, 1800.0)
        self.assertIsNone(self.gateway._http_client.timeout.total)
        self.assertEqual(self.gateway._http_client.timeout.sock_connect, 15)

    async def test_prompt_timeout_interrupts_clears_queue_and_recovers_for_next_job(self) -> None:
        self.gateway.config.prompt_timeout_s = 0.6
        self.gateway.config.ws_poll_interval_s = 1.5
        self.worker0.stall_event = asyncio.Event()
        self.worker0.track_queue = True
        self.worker0.reject_prompt_if_busy = True
        self.worker0.interrupt_clear_delay_s = 0.12
        self.worker0.continuous_ws_ticks = True

        r0 = await self.client.post(
            "/v1/jobs",
            headers=self._auth(),
            json={"mode": "t2i", "profile": "fast", "prompt": "Too slow on GPU0", "preferred_gpu": 0},
        )
        r1 = await self.client.post(
            "/v1/jobs",
            headers=self._auth(),
            json={"mode": "t2i", "profile": "fast", "prompt": "Next queued job on GPU0", "preferred_gpu": 0},
        )
        j0 = (await r0.json())["job"]["id"]
        j1 = (await r1.json())["job"]["id"]

        for _ in range(80):
            s0 = (await (await self.client.get(f"/v1/jobs/{j0}", headers=self._auth())).json())["job"]
            s1 = (await (await self.client.get(f"/v1/jobs/{j1}", headers=self._auth())).json())["job"]
            if s0["status"] in ("failed", "completed", "cancelled") and s1["status"] in ("completed", "failed", "cancelled"):
                break
            await asyncio.sleep(0.03)

        self.assertEqual(s0["status"], "failed", msg=str(s0))
        self.assertIn("ComfyUI execution timed out after", s0["error"])
        self.assertIn("on GPU0", s0["error"])
        self.assertGreaterEqual(self.worker0.interrupt_calls, 1)
        self.assertGreaterEqual(self.worker0.queue_clear_calls, 1)
        self.assertEqual(s1["status"], "completed", msg=str(s1))
        self.assertTrue(s1["image_available"])


class TestLaunchHelperSignature(unittest.TestCase):
    def test_prepare_and_launch_binds_real_bootstrap_signature(self) -> None:
        import inspect
        from unittest import mock
        from qwen_kaggle import bootstrap
        from qwen_kaggle.gateway import prepare_and_launch_gateway_subprocess

        sig = inspect.signature(bootstrap.ensure_models_and_templates)
        bound_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def fake_ensure(*args: Any, **kwargs: Any) -> dict[str, str]:
            sig.bind(*args, **kwargs)
            bound_calls.append((args, kwargs))
            return {}

        fake_proc = mock.MagicMock()
        fake_proc.stdout = iter(["[GATEWAY] ok\n"])
        fake_proc.wait.return_value = 0

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with (
                mock.patch("qwen_kaggle.gateway.collect_system_diagnostics", return_value={"nvidia_smi": ["0, T4", "1, T4"]}),
                mock.patch("qwen_kaggle.gateway.setup_comfy_and_venv", return_value=root / "python"),
                mock.patch("qwen_kaggle.gateway.ensure_models_and_templates", side_effect=fake_ensure),
                mock.patch("qwen_kaggle.gateway.subprocess.Popen", return_value=fake_proc) as popen_mock,
            ):
                rc = prepare_and_launch_gateway_subprocess(
                    comfy_dir=root / "ComfyUI",
                    venv_dir=root / "venv",
                    input_dir=root / "input",
                    preset="full_i2i",
                    bearer_token="tok",
                    use_tunnel=False,
                )
        self.assertEqual(rc, 0)
        self.assertEqual(len(bound_calls), 1)
        popen_args = popen_mock.call_args.args[0]
        self.assertIn("--no-tunnel", popen_args)
        idx = popen_args.index("--idle-timeout")
        self.assertEqual(popen_args[idx + 1], "600.0")
        p_idx = popen_args.index("--prompt-timeout")
        self.assertEqual(popen_args[p_idx + 1], "1800.0")

    def test_ensure_cloudflared_binary_uses_cached_asset_without_download(self) -> None:
        from unittest import mock
        from qwen_kaggle.constants import RUNTIME_CLOUDFLARED_FILENAME
        from qwen_kaggle.gateway import ensure_cloudflared_binary

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp = root / "input" / "cache"
            inp.mkdir(parents=True)
            cached = inp / RUNTIME_CLOUDFLARED_FILENAME
            cached.write_bytes(b"ELF" + b"\x00" * 1_100_000)
            dst = root / "tmp" / "cloudflared"

            with mock.patch("qwen_kaggle.gateway.urllib.request.urlopen") as url_mock:
                out = ensure_cloudflared_binary(dst, input_root=root / "input")

            self.assertEqual(out, dst)
            self.assertTrue(dst.exists())
            self.assertEqual(dst.stat().st_size, cached.stat().st_size)
            url_mock.assert_not_called()

    def test_run_gateway_server_starts_tunnel_concurrently_with_workers(self) -> None:
        from unittest import mock
        from qwen_kaggle.gateway import CloudflaredTunnel, run_gateway_server

        events: list[str] = []

        async def _scenario() -> None:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                cfg = GatewayConfig(
                    comfy_dir=root / "ComfyUI",
                    venv_dir=root / "venv",
                    input_dir=root / "input",
                    log_dir=root / "logs",
                    host="127.0.0.1",
                    port=0,
                    num_workers=1,
                    worker_urls=("http://127.0.0.1:8188",),
                    idle_timeout_s=0.0,
                )

                async def fake_gw_start(gw_self: QwenGateway) -> None:
                    events.append("gateway_start_begin")
                    await asyncio.sleep(0.08)
                    events.append("gateway_start_end")
                    gw_self.request_shutdown("test_done")

                async def fake_tunnel_start(tun_self: CloudflaredTunnel, timeout_s: float = 45.0) -> str:
                    events.append("tunnel_start_begin")
                    await asyncio.sleep(0.02)
                    events.append("tunnel_start_end")
                    tun_self.public_url = "https://early-test.trycloudflare.com"
                    return tun_self.public_url

                async def fake_tunnel_stop(tun_self: CloudflaredTunnel) -> None:
                    events.append("tunnel_stop")

                with (
                    mock.patch.object(QwenGateway, "start", new=fake_gw_start),
                    mock.patch.object(QwenGateway, "stop", new=lambda _self: asyncio.sleep(0)),
                    mock.patch.object(CloudflaredTunnel, "start", new=fake_tunnel_start),
                    mock.patch.object(CloudflaredTunnel, "stop", new=fake_tunnel_stop),
                ):
                    await run_gateway_server(cfg, use_tunnel=True)

        asyncio.run(_scenario())
        self.assertIn("tunnel_start_begin", events)
        self.assertIn("gateway_start_end", events)
        self.assertLess(events.index("tunnel_start_begin"), events.index("gateway_start_end"))
        self.assertEqual(events[-1], "tunnel_stop")
