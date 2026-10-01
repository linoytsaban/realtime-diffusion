# qwen-stream

StreamDiffusion-style real-time editing with Qwen-Image 2.1: a video or camera stream goes in, an edited stream
comes out. Prototype stage. Every design choice and the optimization backlog are in [DESIGN.md](DESIGN.md).

## How it works

- **Edit mode (default).** Each frame is encoded by [TAEQI2.1](https://huggingface.co/madebyollin/taeqi2_1) and given
  to Qwen-Image 2.1 as the condition image of an edit instruction. The output is denoised from fixed noise in 4-6 steps
  with the [Viggle turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) LoRA fused in, then decoded by
  TAEQI2.1.
- **img2img mode.** The StreamDiffusion recipe: the frame is noised to a mid-schedule sigma and denoised in 2-4 steps.
  Faster, but with weak prompt adherence.
- The prompt goes through the 8B VLM once per prompt change, never per frame.

```python
from qwen_stream import StreamQwen

stream = StreamQwen(512, 512, schedule="edit-3", fp8="row", compile=True)
stream.set_prompt("Turn this photo into a watercolor painting.", reference=first_frame)  # img2img: no reference
out = stream(frame)  # (H, W, 3) uint8 CUDA tensor in and out
```

## Benchmark

```bash
python bench.py --video input.mp4 --frames 48 [--edit --schedule edit-3] [--compile] [--fp8 row|tensor] [--max-memory-gb 24] --out out.mp4
```

Prints per-stage latency and writes an `input | output` side-by-side video.
Current numbers at 512px, compiled with FP8, within a 24 GB budget on an H100 PCIe: edit mode 7.5 FPS at 3 steps,
img2img 9.8 FPS at 3 steps (see DESIGN.md).

## Live camera demo

```bash
python live.py --fp8 row --compile   # ~3 min warmup, then prints http://127.0.0.1:8765/?token=...
```

Browsers only allow camera access on https or localhost. From a remote box, either forward the port
(`ssh -L 8765:127.0.0.1:8765 box`) or put a tunnel in front, e.g.
`touch empty.yml && cloudflared tunnel --config empty.yml --url http://127.0.0.1:8765`. The empty config keeps
cloudflared from inheriting a system `config.yml`, whose catch-all ingress rule would 404 every request.

## Requirements

`torch`, `torchvision`, `diffusers` with Qwen-Image 2.1 (git `main`), `transformers>=5`, `safetensors`, `av`;
`torchao` for `--fp8`; `fastapi` + `uvicorn` for the live demo.
peft is not required.

Licenses: Qwen-Image 2.1 and the Viggle turbo LoRA are under the Qwen Research License (non-commercial).
TAEQI2.1 is MIT.
