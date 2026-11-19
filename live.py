"""Live camera demo: browser webcam -> WebSocket -> StreamQwen -> browser, with live speed stats.

    python live.py --fp8 --compile   # then open the printed URL (camera access needs https or localhost)
"""

import argparse
import asyncio
import json
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from torchvision.io import decode_jpeg, encode_jpeg

from qwen_stream import SimilarityFilter, StreamQwen
from qwen_stream.stream import LORAS

TOKEN = secrets.token_urlsafe(12)  # the page is reachable through a public tunnel, so gate it
app = FastAPI()
stream: StreamQwen = None
gpu_lock = asyncio.Lock()  # one stream at a time: StreamQwen holds a single prompt cache
active = {"ws": None}  # the connection holding the stream; a new one takes over (page reloads, a second tab)
# All GPU work runs on this one thread. torch.compile's CUDA graphs are recorded per thread, so warmup and
# every frame must share it; a thread pool that hops threads re-records or hangs.
gpu_thread = ThreadPoolExecutor(max_workers=1)


async def on_gpu(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(gpu_thread, fn, *args)


def timed(fn, *args):
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn(*args)
    torch.cuda.synchronize()
    return out, (time.perf_counter() - start) * 1000


class Session:
    """GPU-side state for one browser connection. Only the GPU worker thread calls into it."""

    def __init__(self):
        self.filter, self.use_filter, self.last_output = SimilarityFilter(), False, None
        self.prompt, self.edit, self.prompt_changed = "", True, False

    def configure(self, settings):
        if "prompt" in settings or "mode" in settings:
            self.prompt = settings.get("prompt", self.prompt)
            self.edit = settings.get("mode", "edit" if self.edit else "img2img") == "edit"
            self.prompt_changed = True  # applied on the next frame: edit mode needs one as the VLM's reference
        if "schedule" in settings:
            stream.set_schedule(settings["schedule"])
        if "filter" in settings:
            self.use_filter = bool(settings["filter"])

    def process(self, jpeg: bytes):
        frame, t_io = timed(lambda: decode_jpeg(torch.frombuffer(bytearray(jpeg), dtype=torch.uint8), device="cuda"))
        frame = frame.permute(1, 2, 0)  # (H, W, 3)
        stats = {"skipped": False, "nfe": len(stream.sigmas) - 1}
        if self.prompt_changed:
            _, stats["prompt"] = timed(stream.set_prompt, self.prompt, frame if self.edit else None)
            self.prompt_changed = False
        if self.use_filter and self.last_output is not None and self.filter.should_skip(frame):
            stats["skipped"] = True
            return self.last_output, stats
        latents, stats["encode"] = timed(stream.encode, frame)
        latents, stats["denoise"] = timed(stream.denoise, latents)
        image, stats["decode"] = timed(stream.decode, latents)
        jpeg_out, t_jpeg = timed(lambda: encode_jpeg(image.permute(2, 0, 1), quality=85).cpu().numpy().tobytes())
        stats["jpeg"] = t_io + t_jpeg
        self.last_output = jpeg_out
        return jpeg_out, stats


@app.get("/")
def index(token: str = ""):
    if token != TOKEN:
        raise HTTPException(403)
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.websocket("/ws")
async def websocket(ws: WebSocket, token: str = ""):
    if token != TOKEN:
        return await ws.close(code=1008)
    await ws.accept()
    if active["ws"] is not None:
        try:
            await active["ws"].close(code=1012, reason="taken over by a newer connection")
        except RuntimeError:  # it already disconnected on its own
            pass
    async with gpu_lock:
        active["ws"] = ws
        session, pending = Session(), {"frame": None, "settings": {}}
        ready = asyncio.Event()
        await ws.send_text(json.dumps({"size": stream.latent_hw[0] * 16}))

        async def receive():
            # Keep only the newest frame and the newest settings; stale frames are dropped, never queued.
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect()
                if message.get("bytes"):
                    pending["frame"] = message["bytes"]
                elif message.get("text"):
                    pending["settings"].update(json.loads(message["text"]))
                ready.set()

        async def work():
            while True:
                await ready.wait()
                ready.clear()
                settings, pending["settings"] = pending["settings"], {}
                frame, pending["frame"] = pending["frame"], None
                if settings:
                    await on_gpu(session.configure, settings)
                    await ws.send_text(json.dumps({"configured": list(settings)}))
                if frame is not None:
                    image, stats = await on_gpu(session.process, frame)
                    await ws.send_text(json.dumps(stats))
                    await ws.send_bytes(image)

        tasks = [asyncio.create_task(receive()), asyncio.create_task(work())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)  # let an in-flight GPU call finish first
            if active["ws"] is ws:
                active["ws"] = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--fp8", choices=["row", "tensor"], help="FP8 DiT with per-row or per-tensor scaling")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--text-encoder-device", default="cpu", help="cuda: faster prompt changes, +17.5 GB")
    parser.add_argument("--attention", default="cudnn", choices=["cudnn", "flash", "sage"], help="sage: Ada GPUs")
    parser.add_argument("--lora", default="dmd2-2step", choices=list(LORAS))
    parser.add_argument("--schedule", default="edit-2", help="starting schedule; switchable in the UI")
    parser.add_argument("--token", help="URL token; default: a new random one per start")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    global TOKEN
    TOKEN = args.token or TOKEN

    def warmup():
        # Two prompt lengths, so compile has generalized the prefix length before anyone connects.
        global stream
        stream = StreamQwen(args.size, args.size, schedule=args.schedule, fp8=args.fp8, compile=args.compile,
                             text_encoder_device=args.text_encoder_device, attention=args.attention, lora=args.lora)
        dummy = torch.zeros(args.size, args.size, 3, dtype=torch.uint8, device="cuda")
        for reference in [None, dummy]:  # img2img and edit layouts
            for prompt in ["a watercolor painting", "an oil painting with thick impasto brush strokes and warm evening light"]:
                stream.set_prompt(prompt, reference)
                stream(dummy)

    gpu_thread.submit(warmup).result()
    print(f"ready: http://127.0.0.1:{args.port}/?token={TOKEN}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
