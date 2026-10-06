# Qwen Stream

Real-time, instruction-based editing of a live camera feed with [Qwen-Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1).
Type an edit like "Turn the person into a marble statue" and watch it applied to your webcam at ~7 FPS.

Based on and inspired by [StreamDiffusion](https://github.com/cumulo-autumn/streamdiffusion) and
[StreamDiffusionV2](https://github.com/chenfengxu714/StreamDiffusionV2).

<!-- demo clip goes here -->

## How it works

```
camera frame -> tiny VAE -> prefill [instruction + frame] -> 3 denoising steps -> tiny VAE -> browser
```

- **Edit mode (default):** each frame is the condition image of an edit instruction. Pose and motion follow the
  camera; the instruction decides the look.
- **Few steps:** the [Pruna 5-step LoRA](https://huggingface.co/PrunaAI/Pruna-Qwen-Image-2.1), run at 3 steps.
  The [Viggle turbo LoRA](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) is an option.
- **Fast ends:** [TAEQI2.1](https://huggingface.co/madebyollin/taeqi2_1) encodes and decodes in ~2 ms each.
- **Fast middle:** the instruction is encoded once per prompt; each step runs a lean, compiled forward with CUDA
  graphs and FP8 weights.
- **img2img mode:** the classic StreamDiffusion recipe (noise the frame, denoise 2-4 steps). Faster, but follows the
  prompt less closely.

Every non-obvious choice is written up, with measurements, in [DESIGN.md](DESIGN.md).

## Performance

One H100 PCIe, 512px, compiled, FP8:

| mode | GPU ms per frame | FPS |
|---|---|---|
| edit, 3 steps (default) | 134 | ~7 |
| img2img, 3 steps | 91-97 | ~10 |

Other GPUs are not measured yet. The design targets a 24 GB RTX 4090 (text encoder on CPU, ~14 GB on the GPU).

## Install

Needs a CUDA GPU and Python 3.10+.

```bash
git clone https://github.com/linoytsaban/qwen-stream && cd qwen-stream
pip install -r requirements.txt
```

Weights (~31 GB, mostly the DiT and the text encoder) download from the Hugging Face Hub on first run.

## Live camera demo

```bash
python live.py --fp8 row --compile --text-encoder-device cuda
```

Warmup takes ~11 minutes (compilation). It then prints a URL with an access token; open it in Chrome and allow the
camera. Use the **Shuffle prompt** button for ready-made edits, or type your own instruction.

- `--text-encoder-device cuda` makes prompt changes near-instant but needs ~17 GB more GPU memory; leave it out on a
  24 GB card.
- Camera access needs https or localhost. Either forward the port (`ssh -L 8765:127.0.0.1:8765 <server>`) or put a
  tunnel in front:
  `touch empty.yml && cloudflared tunnel --config empty.yml --url http://127.0.0.1:8765`
  (the empty config avoids inheriting a system `config.yml`).
- One viewer at a time; a new tab takes over.

## Python API

```python
from qwen_stream import StreamQwen

stream = StreamQwen(512, 512, fp8="row", compile=True)          # edit mode, Pruna 5-step LoRA, 3 steps
stream.set_prompt("Turn the person into a marble statue in a museum.", reference=first_frame)
out = stream(frame)  # (H, W, 3) uint8 CUDA tensor in and out
```

For img2img, pass `schedule="img2img-medium"` and call `set_prompt(description)` without a reference.

| option | values |
|---|---|
| `lora` | `"pruna-5"` (default), `"pruna-8"`, `"viggle"` |
| `schedule` | `"edit-3"` (default), `"edit-4"`, `"edit-6"`, `"img2img-subtle"`, `"img2img-medium"`, `"img2img-strong"` |
| `fp8` | `None`, `"row"`, `"tensor"` |
| `attention` | `"cudnn"` (default), `"flash"`, `"sage"` (aimed at RTX 4090; needs `sageattention`) |
| `text_encoder_device` | `"cpu"` (default), `"cuda"` |

## Benchmark

```bash
python bench.py --video in.mp4 --compile --fp8 row [--max-memory-gb 24] --out out.mp4
```

Prints per-stage latency and writes an input | output comparison video. `--img2img`, `--lora`, `--schedule` and
`--attention` select the same options as above.

## Limitations

- Fixed noise keeps a still scene still, but its patterns stay fixed in screen space.
- The Pruna LoRA was trained at 1024px; at 512px and 3 steps it is outside its training setup and was chosen by eye.
- Prompt changes take ~5 s with the text encoder on CPU.

## License

Code: [Apache-2.0](LICENSE). Models have their own licenses: Qwen-Image 2.1 and the Viggle and Pruna LoRAs are
under the Qwen Research License (non-commercial); TAEQI2.1 is MIT.
