# Qwen Stream

Real-time interactive generation with Qwen Image 2.1 based on and inspired by [StreamDiffusion](https://github.com/cumulo-autumn/streamdiffusion) and [StreamDiffusion V2](https://github.com/chenfengxu714/StreamDiffusionV2).

Design decisions and backlog: [DESIGN.md](DESIGN.md).

- **Edit mode (default):** each frame is the condition image of an edit instruction, denoised in 3 steps with the
  [Viggle turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) LoRA and decoded by
  [TAEQI2.1](https://huggingface.co/madebyollin/taeqi2_1). 7.5 FPS at 512px on an H100.
- **img2img mode:** the StreamDiffusion recipe (noise the frame, denoise 2-4 steps). 9.8 FPS, weaker adherence.
- **Gaze focus (experimental):** the edit follows where you look. See [GAZE.md](GAZE.md).

```python
from qwen_stream import StreamQwen

stream = StreamQwen(512, 512, schedule="edit-3", fp8="row", compile=True)
stream.set_prompt("Turn this photo into a watercolor painting.", reference=first_frame)  # img2img: no reference
out = stream(frame)  # (H, W, 3) uint8 CUDA tensor in and out
```

## Benchmark

```bash
python bench.py --video in.mp4 --edit --schedule edit-3 --compile --fp8 row [--max-memory-gb 24] --out out.mp4
```

## Live camera demo

```bash
python live.py --fp8 row --compile [--text-encoder-device cuda]   # ~11 min warmup, then prints the URL
```

Camera access needs https or localhost: use `ssh -L 8765:127.0.0.1:8765 box`, or
`touch empty.yml && cloudflared tunnel --config empty.yml --url http://127.0.0.1:8765`
(the empty config avoids inheriting a system `config.yml` that 404s everything).

## Tests

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python tests/test_focus.py && node --test tests/
```

## Requirements

`torch`, `torchvision`, `diffusers` (git `main`), `transformers>=5`, `safetensors`, `av`, `torchao` (FP8),
`fastapi` and `uvicorn` (live demo).

Qwen-Image 2.1 and the Viggle LoRA: Qwen Research License (non-commercial). TAEQI2.1: MIT.
