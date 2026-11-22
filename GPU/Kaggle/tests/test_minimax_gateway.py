"""Async contract and end-to-end mocked TPU & T4-GGUF tests for `minimax_kaggle.gateway`."""

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

from minimax_kaggle.constants import MODELS
from minimax_kaggle.gateway import (
    BINARY_EVENT_PREVIEW_IMAGE,
    BINARY_EVENT_PREVIEW_IMAGE_WITH_METADATA,
    GatewayConfig,
    MiniMaxGateway,
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
    """In-memory mock ComfyUI HTTP + WebSocket worker verifying MiniMax-H3-Image gateway requests."""

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
        prompt_id = f"mm{self.gpu_index}_{len(self.submitted_prompts) + 1}"
        prompt_graph = body["prompt"]
        client_id = body.get("client_id", "")
        self.submitted_prompts.append(prompt_graph)
        if self.track_queue:
            self.queue_running.append([0, prompt_id])
        asyncio.create_task(self._simulate_execution(prompt_id, client_id))
        return web.json_response({"prompt_id": prompt_id})

    async def _simulate_execution(self, prompt_id: str, client_id: str) -> None:
        ws = self._ws_clients.get(client_id)
        if self.stall_event is not None:
            while not self.stall_event.is_set():
                if prompt_id in self.interrupted_prompts:
                    return
                await asyncio.sleep(0.03)
            if prompt_id in self.interrupted_prompts:
                return
        await asyncio.sleep(self.delay_s)
        if not self.suppress_ws_events and ws is not None and not ws.closed:
            await ws.send_json({"type": "executing", "data": {"prompt_id": prompt_id, "node": "11"}})
            await ws.send_json({"type": "progress", "data": {"prompt_id": prompt_id, "value": 1, "max": 6}})
            await asyncio.sleep(self.delay_s)
            await ws.send_json({"type": "progress", "data": {"prompt_id": prompt_id, "value": 6, "max": 6}})
            preview_pkt = struct.pack(">II", BINARY_EVENT_PREVIEW_IMAGE, 2) + TINY_PNG
            await ws.send_bytes(preview_pkt)
            await ws.send_json({"type": "executing", "data": {"prompt_id": prompt_id, "node": "12"}})
            await asyncio.sleep(self.delay_s)
        self.history[prompt_id] = {
            "status": {"status_str": "success"},
            "outputs": {
                "13": {
                    "images": [
                        {"filename": f"minimax_out_{prompt_id}.png", "subfolder": "", "type": "output"}
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


def _seed_models(comfy_dir: Path, keys: tuple[str, ...]) -> None:
    for key in keys:
        spec = MODELS[key]
        path = comfy_dir / "models" / spec.subdir / spec.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights")


class TestPreviewAndDimensions(unittest.TestCase):
    def test_parses_preview_frames_and_png_ihdr(self) -> None:
        v1 = struct.pack(">II", BINARY_EVENT_PREVIEW_IMAGE, 2) + TINY_PNG
        self.assertEqual(parse_comfy_preview_frame(v1), ("image/png", TINY_PNG))
        meta = json.dumps({"image_type": "image/jpeg"}).encode("utf-8")
        v4 = struct.pack(">II", BINARY_EVENT_PREVIEW_IMAGE_WITH_METADATA, len(meta)) + meta + b"JPG"
        self.assertEqual(parse_comfy_preview_frame(v4), ("image/jpeg", b"JPG"))
        self.assertEqual(parse_png_dimensions(TINY_PNG), (1, 1))


class TestMiniMaxTpuGateway(AioHTTPTestCase):
    async def get_application(self) -> web.Application:
        self._tmp = tempfile.TemporaryDirectory()
        self.comfy_dir = Path(self._tmp.name) / "ComfyUI"
        _seed_models(self.comfy_dir, ("dit_int8", "clip_int8", "vae"))

        self.worker0 = _FakeComfyWorker(0, delay_s=0.03)
        self.w_srv0 = TestServer(self.worker0.app)
        await self.w_srv0.start_server()

        url0 = str(self.w_srv0.make_url("")).rstrip("/")
        cfg = GatewayConfig(
            comfy_dir=self.comfy_dir,
            venv_dir=Path(self._tmp.name) / "venv",
            input_dir=Path(self._tmp.name) / "input",
            log_dir=Path(self._tmp.name) / "logs",
            bearer_token="secret-minimax-tpu",
            worker_urls=(url0,),
            accelerator="tpu",
            model_preset="tpu_int8",
            ws_poll_interval_s=0.08,
        )
        self.gateway = MiniMaxGateway(cfg)
        return create_gateway_app(self.gateway)

    async def tearDownAsync(self) -> None:
        await self.w_srv0.close()
        await super().tearDownAsync()
        self._tmp.cleanup()

    def _auth(self) -> dict[str, str]:
        return {"Authorization": "Bearer secret-minimax-tpu"}

    async def test_auth_status_and_pe_rejection(self) -> None:
        self.assertEqual((await self.client.get("/health")).status, 200)
        self.assertEqual((await self.client.get("/v1/status")).status, 401)
        self.assertEqual((await self.client.get("/v1/status", headers={"Authorization": "Bearer bad"})).status, 401)

        st_resp = await self.client.get("/v1/status", headers=self._auth())
        self.assertEqual(st_resp.status, 200)
        st = await st_resp.json()
        self.assertEqual(st["service"], "minimax-kaggle-gateway")
        self.assertEqual(st["accelerator"], "tpu_v5e_8")
        self.assertFalse(st["capabilities"]["prompt_enhancer"])
        self.assertEqual(st["active_models"], {"dit": "dit_int8", "clip": "clip_int8", "vae": "vae"})

        # Prompt Enhancer routes/flags must fail closed
        r_ref = await self.client.post(
            "/v1/jobs",
            headers=self._auth(),
            json={"action": "refine", "prompt": "short prompt"},
        )
        self.assertEqual(r_ref.status, 400)
        r_enh = await self.client.post(
            "/v1/jobs",
            headers=self._auth(),
            json={"action": "generate", "prompt": "short prompt", "enhance_prompt": True},
        )
        self.assertEqual(r_enh.status, 400)

        # Exceeding 49 MP cap fails validation on TPU
        r_huge = await self.client.post(
            "/v1/jobs",
            headers=self._auth(),
            json={"mode": "t2i", "prompt": "Too big", "width": 7680, "height": 7680},
        )
        self.assertEqual(r_huge.status, 400)

        # Up to ~48 MP (e.g. 6912x6912 for multi-character scenes) is accepted on TPU v5e-8
        self.assertEqual(st["capabilities"]["max_megapixels"], 49.0)
        self.assertTrue(st["capabilities"]["inpainting_with_references"])
        self.assertTrue(st["capabilities"]["outpainting"])
        r_48mp = await self.client.post(
            "/v1/jobs",
            headers=self._auth(),
            json={"mode": "t2i", "prompt": "Four people group photo", "width": 6912, "height": 6912},
        )
        self.assertEqual(r_48mp.status, 202)
        j_48 = (await r_48mp.json())["job"]["id"]
        for _ in range(50):
            d48 = (await (await self.client.get(f"/v1/jobs/{j_48}", headers=self._auth())).json())["job"]
            if d48["status"] == "completed":
                break
            await asyncio.sleep(0.02)
        self.assertEqual(d48["status"], "completed")
        self.assertEqual((d48["width"], d48["height"]), (6912, 6912))

    async def test_inpaint_with_mask_and_refs_and_outpaint_dispatch(self) -> None:
        img_b64 = base64.b64encode(TINY_PNG).decode("ascii")
        img_obj = {"filename": "base.png", "data_url": f"data:image/png;base64,{img_b64}"}
        ref_obj = {"filename": "vest.png", "data_url": f"data:image/png;base64,{img_b64}"}
        mask_obj = {"filename": "mask.png", "data_url": f"data:image/png;base64,{img_b64}"}

        # Missing mask_image must fail closed
        r_nomask = await self.client.post(
            "/v1/jobs",
            headers=self._auth(),
            json={"mode": "inpaint", "prompt": "Add vest", "images": [img_obj]},
        )
        self.assertEqual(r_nomask.status, 400)

        # Valid inpaint with mask + extra reference
        r_inp = await self.client.post(
            "/v1/jobs",
            headers=self._auth(),
            json={
                "mode": "inpaint",
                "prompt": "Place pink vest from <image2> onto person in <image1>",
                "images": [img_obj, ref_obj],
                "mask_image": mask_obj,
                "mask_expand": 20,
                "mask_blur": 10,
            },
        )
        self.assertEqual(r_inp.status, 202)
        j_inp = (await r_inp.json())["job"]["id"]
        for _ in range(50):
            di = (await (await self.client.get(f"/v1/jobs/{j_inp}", headers=self._auth())).json())["job"]
            if di["status"] == "completed":
                break
            await asyncio.sleep(0.02)
        self.assertEqual(di["status"], "completed")
        self.assertTrue(di["has_mask"])
        self.assertEqual(di["mask_expand"], 20)
        inp_graph = self.worker0.submitted_prompts[-1]
        self.assertEqual(inp_graph["40"]["class_type"], "LoadImageMask")
        self.assertEqual(inp_graph["14"]["class_type"], "ImageCompositeMasked")
        self.assertIn("images.image_2", inp_graph["5"]["inputs"])

        # Outpaint with zero padding fails closed; positive left/right succeeds
        r_bad_out = await self.client.post(
            "/v1/jobs",
            headers=self._auth(),
            json={"mode": "outpaint", "prompt": "Extend scene", "images": [img_obj]},
        )
        self.assertEqual(r_bad_out.status, 400)

        r_out = await self.client.post(
            "/v1/jobs",
            headers=self._auth(),
            json={
                "mode": "outpaint",
                "prompt": "Extend scenic landscape to 16:9",
                "images": [img_obj],
                "outpaint_left": 512,
                "outpaint_right": 512,
                "outpaint_feather": 80,
            },
        )
        self.assertEqual(r_out.status, 202)
        j_out = (await r_out.json())["job"]["id"]
        for _ in range(50):
            do = (await (await self.client.get(f"/v1/jobs/{j_out}", headers=self._auth())).json())["job"]
            if do["status"] == "completed":
                break
            await asyncio.sleep(0.02)
        self.assertEqual(do["status"], "completed")
        self.assertEqual(do["outpaint"]["feather"], 80)
        out_graph = self.worker0.submitted_prompts[-1]
        self.assertEqual(out_graph["30"]["class_type"], "ImagePadForOutpaint")
        self.assertEqual(out_graph["14"]["class_type"], "ImageCompositeMasked")

    async def test_single_stage_4k_t2i_without_free_cycle_and_sse(self) -> None:
        resp = await self.client.post(
            "/v1/jobs",
            headers=self._auth(),
            json={
                "mode": "t2i",
                "profile": "quality",  # defaults to 4096x4096 on TPU
                "prompt": "Architectural photography of a sunlit travertine museum atrium.",
                "seed": -1,
            },
        )
        self.assertEqual(resp.status, 202)
        job = (await resp.json())["job"]
        job_id = job["id"]
        self.assertEqual(job["width"], 4096)
        self.assertEqual(job["height"], 4096)
        self.assertEqual(job["effective_steps"], 6)
        self.assertEqual(job["effective_cfg"], 1.0)

        sse = await self.client.get(f"/v1/jobs/{job_id}/events", headers=self._auth())
        self.assertEqual(sse.status, 200)
        sse_body = (await sse.read()).decode("utf-8")
        self.assertIn('"status": "completed"', sse_body)

        done = (await (await self.client.get(f"/v1/jobs/{job_id}", headers=self._auth())).json())["job"]
        self.assertEqual(done["status"], "completed")
        self.assertIsNone(done["stage1_s"])
        self.assertTrue(done["image_available"])
        self.assertTrue(done["preview_data_url"].startswith("data:image/png;base64,"))
        self.assertFalse(done["warm_worker"])

        # Zero Stage-1 /free calls and single prompt submission
        self.assertEqual(len(self.worker0.free_calls), 0)
        self.assertEqual(len(self.worker0.submitted_prompts), 1)
        submitted = self.worker0.submitted_prompts[0]
        self.assertEqual(submitted["1"]["class_type"], "UNETLoader")
        self.assertEqual(submitted["3"]["class_type"], "CLIPLoader")
        self.assertEqual(submitted["5"]["class_type"], "MiniMaxH3ImageConditioning")
        self.assertEqual(submitted["11"]["inputs"]["steps"], 6)

        img_resp = await self.client.get(f"/v1/jobs/{job_id}/image", headers=self._auth())
        self.assertEqual(img_resp.status, 200)
        self.assertEqual(await img_resp.read(), TINY_PNG)

        # Second job marks warm_worker=True
        r2 = await self.client.post(
            "/v1/jobs",
            headers=self._auth(),
            json={"mode": "t2i", "profile": "fast", "prompt": "Warm follow-up"},
        )
        j2 = (await r2.json())["job"]["id"]
        for _ in range(50):
            d2 = (await (await self.client.get(f"/v1/jobs/{j2}", headers=self._auth())).json())["job"]
            if d2["status"] == "completed":
                break
            await asyncio.sleep(0.03)
        self.assertEqual(d2["status"], "completed")
        self.assertTrue(d2["warm_worker"])
        self.assertEqual(d2["width"], 2048)

    async def test_idle_extend_shutdown_and_timeout_recovery(self) -> None:
        self.gateway.config.prompt_timeout_s = 0.45
        self.gateway.config.ws_poll_interval_s = 0.8
        self.worker0.stall_event = asyncio.Event()
        self.worker0.track_queue = True
        self.worker0.reject_prompt_if_busy = True
        self.worker0.interrupt_clear_delay_s = 0.08

        r0 = await self.client.post("/v1/jobs", headers=self._auth(), json={"prompt": "Stall me"})
        r1 = await self.client.post("/v1/jobs", headers=self._auth(), json={"prompt": "Recover next"})
        j0 = (await r0.json())["job"]["id"]
        j1 = (await r1.json())["job"]["id"]

        for _ in range(80):
            s0 = (await (await self.client.get(f"/v1/jobs/{j0}", headers=self._auth())).json())["job"]
            s1 = (await (await self.client.get(f"/v1/jobs/{j1}", headers=self._auth())).json())["job"]
            if s0["status"] == "failed" and s1["status"] == "completed":
                break
            await asyncio.sleep(0.03)
        self.assertEqual(s0["status"], "failed")
        self.assertIn("timed out", s0["error"])
        self.assertEqual(s1["status"], "completed")

        ext = await self.client.post("/v1/idle/extend", headers=self._auth(), json={"seconds": 300})
        self.assertEqual(ext.status, 200)

        stop = await self.client.post("/v1/shutdown", headers=self._auth(), json={"reason": "done"})
        self.assertEqual(stop.status, 202)
        self.assertTrue(self.gateway.shutdown_event.is_set())
        after = await self.client.post("/v1/idle/extend", headers=self._auth(), json={"seconds": 60})
        self.assertEqual(after.status, 409)


class TestMiniMaxT4GgufGateway(AioHTTPTestCase):
    async def get_application(self) -> web.Application:
        self._tmp = tempfile.TemporaryDirectory()
        self.comfy_dir = Path(self._tmp.name) / "ComfyUI"
        # Deliberately omit clip_gguf_mmproj initially to test fail-closed Edit guard!
        _seed_models(self.comfy_dir, ("dit_gguf_q4", "clip_gguf_iq3", "vae"))

        self.worker0 = _FakeComfyWorker(0, delay_s=0.02)
        self.w_srv0 = TestServer(self.worker0.app)
        await self.w_srv0.start_server()

        url0 = str(self.w_srv0.make_url("")).rstrip("/")
        cfg = GatewayConfig(
            comfy_dir=self.comfy_dir,
            venv_dir=Path(self._tmp.name) / "venv",
            input_dir=Path(self._tmp.name) / "input",
            log_dir=Path(self._tmp.name) / "logs",
            bearer_token="secret-minimax-t4",
            worker_urls=(url0,),
            accelerator="gpu",
            model_preset="t4_gguf",
            ws_poll_interval_s=0.08,
        )
        self.gateway = MiniMaxGateway(cfg)
        return create_gateway_app(self.gateway)

    async def tearDownAsync(self) -> None:
        await self.w_srv0.close()
        await super().tearDownAsync()
        self._tmp.cleanup()

    def _auth(self) -> dict[str, str]:
        return {"Authorization": "Bearer secret-minimax-t4"}

    async def test_t4_gguf_caps_resolution_and_requires_mmproj_only_for_edit(self) -> None:
        # 4K rejected on T4
        r_4k = await self.client.post(
            "/v1/jobs",
            headers=self._auth(),
            json={"mode": "t2i", "prompt": "Too big for T4", "width": 4096, "height": 4096},
        )
        self.assertEqual(r_4k.status, 400)

        # Edit fails closed when mmproj is absent
        img_b64 = base64.b64encode(TINY_PNG).decode("ascii")
        r_no_mm = await self.client.post(
            "/v1/jobs",
            headers=self._auth(),
            json={
                "mode": "edit",
                "prompt": "Recolor <image1>",
                "images": [{"filename": "ref.png", "data_url": f"data:image/png;base64,{img_b64}"}],
            },
        )
        self.assertEqual(r_no_mm.status, 400)
        payload = await r_no_mm.json()
        self.assertEqual(payload["error"], "MISSING_MODELS")
        self.assertIn("clip_gguf_mmproj", payload["missing_models"])

        # T2I succeeds without mmproj using GGUF loaders
        r_t2i = await self.client.post(
            "/v1/jobs",
            headers=self._auth(),
            json={"mode": "t2i", "profile": "fast", "prompt": "T4 GGUF test"},
        )
        self.assertEqual(r_t2i.status, 202)
        j_t2i = (await r_t2i.json())["job"]["id"]
        for _ in range(50):
            d = (await (await self.client.get(f"/v1/jobs/{j_t2i}", headers=self._auth())).json())["job"]
            if d["status"] == "completed":
                break
            await asyncio.sleep(0.02)
        self.assertEqual(d["status"], "completed")
        self.assertEqual(d["width"], 1024)
        self.assertEqual(self.worker0.submitted_prompts[-1]["1"]["class_type"], "UnetLoaderGGUF")
        self.assertEqual(self.worker0.submitted_prompts[-1]["3"]["class_type"], "CLIPLoaderGGUF")

        # Now seed mmproj -> Edit succeeds with auto 0x0 canvas & <Picture 1> normalization
        _seed_models(self.comfy_dir, ("clip_gguf_mmproj",))
        r_edit = await self.client.post(
            "/v1/jobs",
            headers=self._auth(),
            json={
                "mode": "edit",
                "prompt": "Make <image1> golden",
                "images": [{"filename": "ref.png", "data_url": f"data:image/png;base64,{img_b64}"}],
            },
        )
        self.assertEqual(r_edit.status, 202)
        j_edit = (await r_edit.json())["job"]["id"]
        for _ in range(50):
            de = (await (await self.client.get(f"/v1/jobs/{j_edit}", headers=self._auth())).json())["job"]
            if de["status"] == "completed":
                break
            await asyncio.sleep(0.02)
        self.assertEqual(de["status"], "completed")
        self.assertEqual(de["width"], 0)
        self.assertEqual(de["height"], 0)
        edit_graph = self.worker0.submitted_prompts[-1]
        self.assertEqual(edit_graph["5"]["inputs"]["prompt"], "Make <Picture 1> golden")
        self.assertEqual(len(self.worker0.uploaded_names), 1)


if __name__ == "__main__":
    unittest.main()
