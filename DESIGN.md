# Design notes

Each non-obvious choice in the code is tagged `D<n>` here. Open ideas are `O<n>`.
Numbers are from one H100 PCIe at 512px unless noted.

## Targets

| GPU | Arch | Memory | Notes |
|---|---|---|---|
| RTX 4090 ("Audace") | Ada | 24 GB | FP8; the memory budget to design for |
| RTX Pro 6000 (ZeroGPU) | Blackwell | 96 GB | FP8, NVFP4; GPU attached per request |
| DGX Spark | Blackwell GB10 | 128 GB unified, ~273 GB/s | bandwidth-bound |
| H100 PCIe | Hopper | 80 GB | dev box only; no Hopper-only kernels by default |

## Pipeline

```
edit (default): frame -> tiny VAE -> prefill [instruction, frame] -> 3 steps from fixed noise -> tiny VAE
img2img:        frame -> tiny VAE -> noise to mid-schedule -> 2-4 steps -> tiny VAE
prompt change:  Qwen3-VL once (edit: with the current frame as its reference image)
```

## Current numbers

| mode | GPU ms/frame | FPS (offline / live) | prompt change (encoder on CPU / GPU) |
|---|---|---|---|
| edit-3 | 134 | 7.5 / 7.0 | ~5 s / 0.14 s |
| edit-4 | 162 | - / 5.8 | ~5 s / 0.16 s |
| img2img-medium | 91-97 | 9.8 / 10.4 | ~0.8 s / 0.06 s |

Compiled, FP8 per-row, cuDNN attention. GPU memory: 14 GB with the text encoder on CPU, 27 GB on GPU.

## Decisions

**Model and sampling**
- **D1. Tiny VAE (TAEQI2.1) on both ends.** ~2 ms each vs 46/166 ms for the full VAE (at 1024px). Lossier on fine
  detail. `tiny_vae.py` matches Ollin's `taesd.py` exactly.
- **D2. Viggle turbo LoRA (DMD, no CFG).** Its raw sigma nodes get the pipeline's resolution time shift
  (`shift_terminal=None`, per its card).
- **D3. LoRA fused by hand (`lora.py`).** No peft needed. Fused in fp32, rounded once. Unfused hook path kept for
  offline comparison only. Open: fused vs unfused quality.
- **D5. Fixed noise for the whole stream.** Static input gives static output. Cost: noise patterns are fixed in
  screen space.
- **D13. Edit mode (default).** The frame is the condition image and the prompt is an instruction.
  - Prompt adherence is far better than img2img; pose and motion follow the input.
  - Must start from pure noise: a noised frame as start makes the model copy the input.
  - `edit-3` = `[1, .75, .25]` looks nearly identical to `edit-4`. Two steps break down on hard edits (lego).
  - The VLM runs once per prompt: reusing its output across frames changed the result by under 0.35/255.
  - The image label in the edit template is pinned to "Picture" (it is random upstream).
- **D21. Pruna 5-step LoRA, live default (`live.py --lora pruna-5`).** Pruna's DMD LoRAs use their nodes as given (no
  resolution shift). At 3 steps both beat Viggle by eye (5 photos x 4 prompts); head to head (8 prompts x 3 seeds) the
  5-step LoRA at (1, 6/7, 0.4) was preferred over the 8-step one at (1, 10/13, 0.4). Same speed as Viggle. Its edit-4
  and edit-6 presets are untested ("edit-6" is its full 5 steps); img2img with Pruna is untested.
- **D4. img2img mode.** SDEdit from a turbo node (`img2img-subtle/medium/strong`). Fast but weak adherence:
  only the noise level trades frame against prompt.

**Caching and compute**
- **D6. Prompt encoded once per prompt.** Text tokens are modulated at t=0 and never attend to image tokens, so
  their K/V are reused across steps and frames. In edit mode the frame is part of the prefix, so it is re-prefilled
  per frame (D14).
- **D9. Text encoder placement (`text_encoder_device`).** CPU by default (fits 24 GB, slow prompt changes);
  `cuda` for near-instant prompt changes at +17.5 GB.
- **D10. FP8 DiT (`fp8="row"` or `"tensor"`).** torchao dynamic FP8, after the LoRA is fused. Mean drift ~1.4/255,
  not visible. Per-tensor is 2% faster and looks the same; per-row stays the default.
- **D11. Lean target-only step (`fast.py`).** Precomputes rotary tables and indices once per prompt, so a step is
  pure tensor math. Within 0.6% of diffusers.
  - Compiled `fullgraph`, `mode="reduce-overhead"` (CUDA graphs). Graphs are required in a server: without them a
    busy Python thread pushed a pass from 30 ms to 2.6 s (GIL).
  - One compiled instance per stream; the prefix length generalizes after two prompt lengths.
  - Graphs copy in swapped prefix tensors, verified, so a new prompt cannot use a stale prefix.
- **D14. Lean prefill.** Per frame: `img_in(frame)`, a scatter into fixed slots, 32 blocks emitting K/V.
  ~32 ms vs ~200 ms through diffusers. Within 0.6% of diffusers. Outputs are cloned out of CUDA-graph buffers.
- **D15. cuDNN attention** for unmasked calls (`attention="cudnn"`, or `"flash"`). Portable from Ampere to
  Blackwell, -2 ms per step vs FlashAttention-2. SageAttention's CUDA builds failed (D20); FA3 is
  Hopper-only.
- **D20. SageAttention (`attention="sage"`), for the RTX 4090.** SageAttention's Triton kernel (INT8 Q/K), wrapped
  as a custom op so it compiles and records in CUDA graphs. LPIPS 0.010 vs cuDNN (mean of 20 edits; max 0.023),
  not visible. Slower on the H100 (158.7 vs 134.6 ms per frame); its gain is expected on Ada, not yet measured.
  The CUDA builds failed: pip `sageattention` 2.2.0 gives wrong output here, and `kernels-community/sage-attention`
  launches outside the current stream, so CUDA-graph replays skip it (NaN).
- **D16. Exact split of the prefill attention.** Image rows need no mask, so they use the fast kernel; only ~25 text
  rows keep the mask. -3 ms.
- **D17. Fused projections.** One GEMM for Q/K/V and one for gate/up. The originals are deleted, so the stock
  diffusers forward no longer runs on the model. -0.5 ms per pass.
- **D7. No attention mask in the steps.** A single unpadded prompt needs none.
- **D19. Pooled middle blocks (`set_bottleneck`, off by default).** Training-free DiT-to-U-Net retrofit: blocks
  `start..end-1` run on 2x2-pooled tokens, only their residual delta is upsampled. Fast but not usable without
  calibration (edit-3, FP8, 5 photos x 4 prompts x 3 seeds, LPIPS vs dense; another seed of dense is 0.21):

  | pooled blocks | FPS | LPIPS |
  |---|---|---|
  | none | 7.4 | 0 |
  | 12-20 | 8.7 | 0.34 |
  | 8-24 | 10.7 | 0.48 |
  | 4-28 | 13.8 | 0.62 |

  Faces and texture smear even at 12-20. One pooled block costs 0.02-0.06 (rounding noise: 0.003); blocks 1-2 and 31
  are the worst, 20-23 the most tolerant. Nearest upsampling beats bilinear; pooling the condition image too is
  better in the middle blocks, worse in the early ones.

**Serving (`live.py`, `static/index.html`)**
- **D12.** Newest frame wins (stale frames dropped), two frames in flight (hides network round trip: 9.5 vs 5.7 FPS
  through the tunnel), one dedicated GPU thread (CUDA graphs are per thread), JPEG on the GPU, one user at a time
  (a new tab takes over), URL token gate.
- **D8. Similarity filter** (StreamDiffusion): skips near-identical frames. Built, not yet measured live.

## Measurement notes

- A compiled bf16 pass is 77% GEMM; FP8 GEMMs run at ~40% of FP8 peak.
- In edit mode attention is ~25% of a step (2072 keys); in img2img ~10%.
- Kernel changes on edit-3 (GPU ms per frame): 138 baseline, 132 cuDNN, 128 split prefill, 126 fused projections,
  122 with per-tensor FP8.
- Live warmup takes ~11 min (compiles both layouts at two prompt lengths).

## Backlog

- **O1. Fewer edit steps.** A 1-2 step edit student (rCM or TwinFlow style). Biggest lever; a training project.
- **O2. NVFP4 on Blackwell** (RTX Pro 6000, 5090, Spark).
- **O3. Faster GEMMs.** Autotuned FP8 GEMM templates; FA3 as a Hopper opt-in; measure SageAttention (D20) on a 4090.
- **O4. Partial prefill** (FluxRT-style): recompute only changed frame tokens. Gain is capped at the prefill share
  (~20%), and it is an approximation.
- **O5. Measure the similarity filter live**, and add motion-aware steps (StreamDiffusionV2).
- **O6. Frame interpolation (RIFE): dropped.** Smoother (~5 ms per extra frame) but adds ~140 ms delay, since it
  waits for the next frame; delay matters more for a live camera.
- **O7. Persistent compile cache** to cut the ~11 min warmup.
- **O8. Gaze-targeted edits.** Prototype removed after a first live test (gaze off target, edit not limited);
  it is in commit `fe19c86` (D18).
- **O9. ZeroGPU serving:** the GPU is attached per call, so frames would be processed in chunks.
- **O10. Calibrate D19:** a short finetune (~2.6k steps, as a LoRA) so pooled blocks 4-28 (13.8 FPS) match dense.
