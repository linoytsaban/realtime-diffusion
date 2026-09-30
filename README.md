# qwen-stream

StreamDiffusion-style real-time img2img with Qwen-Image 2.1: a video or camera stream goes in, a restyled stream
comes out. Prototype stage. Every design choice and the optimization backlog are in [DESIGN.md](DESIGN.md).

## How it works

- The frame is encoded by [TAEQI2.1](https://huggingface.co/madebyollin/taeqi2_1), noised to a mid-schedule sigma, and
  denoised in 3 steps by Qwen-Image 2.1 with the
  [Viggle turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) LoRA fused in.
- It is decoded again by TAEQI2.1.
- The prompt is encoded and its K/V prefilled once, then reused for every frame.

```python
from qwen_stream import StreamQwen

stream = StreamQwen(512, 512, start_node=0.75)
stream.set_prompt("a watercolor painting, soft pastel colors, paper texture")
out = stream(frame)  # (H, W, 3) uint8 CUDA tensor in and out
```

## Benchmark

```bash
python bench.py --video input.mp4 --frames 48 [--start-node 0.75] [--compile] [--no-fuse] --out out.mp4
```

Prints per-stage latency and writes an `input | output` side-by-side video.
Current numbers: 7.3 FPS at 512px with 3 steps on an H100 PCIe (see DESIGN.md).

## Requirements

`torch`, `diffusers` with Qwen-Image 2.1 (git `main`), `transformers>=5`, `safetensors`, `av`.
peft is not required.

Licenses: Qwen-Image 2.1 and the Viggle turbo LoRA are under the Qwen Research License (non-commercial).
TAEQI2.1 is MIT.
