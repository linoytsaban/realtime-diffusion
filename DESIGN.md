# Design notes

Every non-obvious choice in the code is tagged `D<n>` and explained here, with what it costs and what could replace it.
Open optimization ideas are `O<n>`. Measurements are from an H100 PCIe (the only GPU tested so far) unless noted.

## Targets

| GPU | Arch | Memory | Notes |
|---|---|---|---|
| RTX 4090 ("Audace") | Ada, sm89 | 24 GB | FP8 tensor cores; 24 GB is the budget to design for |
| RTX Pro 6000 (ZeroGPU) | Blackwell, sm120 | 96 GB | NVFP4 + FP8; ZeroGPU attaches the GPU per request (see O7) |
| DGX Spark | Blackwell GB10 | 128 GB unified, ~273 GB/s | NVFP4; bandwidth-bound: reading bf16 DiT weights alone is ~50 ms/pass |
| H100 (dev box) | Hopper, sm90 | 80 GB | development only; avoid Hopper-only kernels such as FA3 |

## Pipeline per frame

```
frame (H,W,3 uint8) -> TAEQI encode -> noise to sigma_start (fixed noise) -> k Euler steps of the DiT
                    -> TAEQI decode -> frame
prompt change       -> Qwen3-VL encode + prefill prefix K/V once
```

## Decisions

**D1. Tiny VAE (TAEQI2.1) on both ends; the full VAE is never loaded.**
The full Qwen-Image 2.1 VAE takes 46 ms to encode and 166 ms to decode at 1024px on H100 (7 GiB peak). TAEQI2.1 takes
~2 ms each at 512px. It works directly on the normalized latents, so no `latents_mean/std` is applied.
Cost: visibly lossier than the full VAE (fine detail, text). `tiny_vae.py` is a self-contained port of Ollin's `f16`
variant, verified bit-exact (max diff 0.0) against `taesd.py`.

**D2. Viggle turbo (DMD) student with its own sigma nodes.**
The only public few-step distill of 2.1 that works without CFG. Its raw nodes `[1, .9375, .875, .75, .5, .25]` get
the pipeline's resolution-dependent time shift, computed with the same scheduler config (`shift_terminal=None`,
as its card requires). At 512px the shifted nodes are `[1, .963, .923, .837, .632, .364, 0]`.
Cost: the low-noise nodes `.75/.5/.25` are the only ones it was trained to land on, which limits how finely the
img2img strength can be set (D4). No CFG means there is no negative prompt.

**D3. The LoRA is applied by hand and fused by default (`lora.py`).**
peft is not needed to add `B @ A` to 227 Linear layers, and it pulled in a broken `torchao` on the dev box.
Fusing is done in fp32 and rounded to bf16 once. It costs nothing at runtime, and it is required anyway for
quantization (O2) and CUDA graphs. The model card warns that merging into bf16 is lossy. `fuse=False` keeps an exact
forward-hook path so the two can be compared. **Open: compare fused vs unfused quality.**

**D4. img2img = SDEdit from a turbo node; the default start node is 0.75 (3 NFE).**
A sweep on one frame (watercolor prompt, 512px):

| start node (shifted) | NFE | result |
|---|---|---|
| 0.5 (0.63) | 2 | near copy of the input; the prompt barely shows |
| **0.75 (0.84)** | **3** | style applied, structure kept |
| 0.875 (0.92) | 4 | structure lost |
| 1.0 | 6 | pure text-to-image (the prompt path works) |

`set_start_node()` changes it at runtime. The usable range is narrow, and it is quantized to the student's nodes.

**D5. One fixed noise tensor for the whole stream.**
StreamDiffusion does the same. With the same noise every frame, a static input gives a static output, which cuts
flicker. Cost: patterns in that noise stay fixed in screen space ("texture swimming" when the camera moves). See O6.

**D6. Prompt encoding and prefix K/V are computed once per prompt, not per frame or per step.**
Qwen-Image 2.1 modulates text tokens at t=0 (`causal_condition`) and text never attends to image tokens, so the text
K/V are independent of the frame and the timestep. `set_prompt` runs Qwen3-VL (~0.3 s) and one `extract` pass. Every
step of every frame then runs in `cached` mode, computing only the 1024 image tokens.
This is also why the frame goes in as the SDEdit starting point and not as an edit-mode condition image: a condition
image sits in the prefix, so the 8B VLM and the prefill would rerun on every frame (~6.6 TFLOP extra per frame).

**D7. No attention mask.**
With one unpadded prompt the mask would be all ones. Passing `None` lets SDPA choose a fused kernel (FlashAttention-2
on this box). Batched prompts of different lengths would need the mask back.

**D8. Stochastic Similarity Filter (StreamDiffusion), implemented but not measured.**
A frame is skipped with probability `max(0, (cos_sim - t) / (1 - t))` against the last processed frame.
The test video moves continuously, so this is untested until there is a live camera loop.

**D9. Only the DiT lives on the GPU; the text encoder stays on CPU by default (`text_encoder_device`).**
Qwen3-VL-8B (17.5 GB) runs only when the prompt changes. On CPU a prompt change takes 1.3 s instead of 0.3 s, and GPU
peak drops from 30.4 to 14.0 GiB, so it fits a 24 GB RTX 4090 (checked with `bench.py --max-memory-gb 24`).
On DGX Spark (unified memory) `text_encoder_device="cuda"` costs nothing extra. Next: a small cache of recent prompts.

**D10. Optional FP8 DiT (`fp8=True`): torchao dynamic FP8 activations + FP8 weights, per-row scales, on the 32 blocks.**
Needs FP8 tensor cores (Ada/RTX 4090, Hopper, Blackwell). Applied after the LoRA is fused (D3).
On one frame the output differs from bf16 by a mean of 1.4/255 and is visually indistinguishable.
Weights drop from 13.3 to 6.8 GiB. Speed on H100 only improves ~10% (compiled 137 -> 123 ms/frame): launch overhead,
not GEMMs, dominates until O1. The halved weight traffic should matter more on bandwidth-bound Spark.
Load-time peak is still 14 GiB, because the model is loaded in bf16 first. Loading pre-quantized weights would fix
that for 16 GB cards.

**D11. Lean target-only forward for every cached step (`fast.py`).**
`set_prompt` still uses the stock diffusers forward once, to prefill the prefix K/V. After that, each step runs
`CachedDenoiser`. It precomputes the target-token rotary table (as real cos/sin) and the per-layer prefix K/V once,
then runs `img_in` -> one shared modulation (all 32 blocks share it) -> 32 blocks -> `norm_out` / `proj_out`.
There is no data-dependent indexing and no host sync, so it compiles with `fullgraph=True`.
- **Parity:** vs the diffusers cached forward, 0.6% relative difference (bf16 rounding over 32 layers).
- **Cost:** about 50 lines that duplicate the block math. It has to track upstream changes to the 2.1 block.
- **Prompt length:** compile is specialized to the prompt's token count, so a prompt of a new length triggers a
  recompile (~30 s). Fix: pad prompts to a few bucket lengths with a key mask.

## Measurements (H100 PCIe, 512px, fused LoRA, 48 frames of a 24 fps clip)

| config | encode | denoise | decode | total | FPS |
|---|---|---|---|---|---|
| 2 NFE, eager | 1.7 ms | 110 ms | 1.8 ms | 113 ms | 8.8 |
| 3 NFE, eager | 1.7 ms | 165 ms | 1.8 ms | 168 ms | 5.9 |
| 3 NFE, `torch.compile` | 2.6 ms | 132 ms | 2.7 ms | 138 ms | 7.3 |
| 3 NFE, compile + FP8, text encoder on CPU, 24 GB cap | 2.5 ms | 123 ms | 2.7 ms | 128 ms | 7.8 |

| lean forward (D11), eager, 24 GB cap | 3.5 ms | 171 ms | 3.6 ms | 178 ms | 5.6 |
| lean forward, compiled, 24 GB cap | 2.6 ms | 127 ms | 2.8 ms | 132 ms | 7.6 |
| **lean forward, compiled + FP8, 24 GB cap** | 2.7 ms | 98 ms | 2.9 ms | 104 ms | **9.6** |

Per transformer pass (512px, 1024 tokens):

| path | ms/NFE |
|---|---|
| diffusers cached, eager | 55 |
| lean, eager | 57 |
| lean, eager + CUDA graph | 55 |
| lean, compiled (+/- CUDA graph) | 42 |
| lean, compiled, FP8 | 31 |

**CUDA graphs do not help here.** The earlier "CPU launch bound" reading was wrong: graph replay is as slow as eager,
so the time is GPU kernels. Compile helps because it fuses the ~25 ms of elementwise kernels (norms, rope, modulation,
casts) down to under 5 ms.
After compile, a bf16 pass is 37 ms of GPU time:
- GEMMs: 28.7 ms (77%)
- attention: 3.6 ms
- everything else: under 5 ms

That is near the bf16 limit of this card. The remaining levers are FP8/FP4 GEMMs (D10), fewer tokens (O4) and fewer
steps (O5).

**Quality is the open problem, not speed.** At start node 0.75, faces and fine structure get mushy with a generic
style prompt (see D4).

With `--compile`, the reported `set_prompt` time includes compile warmup (minutes), because the first compiled pass
runs there.

With the text encoder on the GPU, peak memory is 30.4 GiB (14 GB DiT plus 17.5 GB text encoder). With D9 it is
14.0 GiB, and the speed is unchanged (6.0 FPS eager, 3 NFE, under a 24 GB cap).

Profile of one cached pass (eager, ~48 ms GPU):
- GEMMs take 28 ms, i.e. ~500 TFLOP/s: already about 2/3 of bf16 peak on this card.
- The other ~20 ms is elementwise and dtype-conversion kernels.
- The CPU needs 54 ms to launch a pass whose GPU work is 48 ms, so launch overhead is a bottleneck.

## Optimization backlog (roughly in priority order)

- **O1. ~~Lean decode-only forward~~** done as D11.
  - CUDA graphs measured no gain on H100. They may still matter on hosts with weak CPUs (Spark's ARM cores), and the
    lean forward is capturable if needed.
- **O2. Low-precision DiT, chosen per GPU generation.**
  - FP8 weights and activations on Ada, Hopper and Blackwell. That covers the 4090.
  - NVFP4 on Blackwell (RTX Pro 6000, Spark).
  - Halves or quarters both the GEMM time and the 14 GB of weights; on Spark, bandwidth is the limit.
  - Needs a quality check against bf16 at 3 NFE.
- **O3. ~~Text encoder off the GPU budget~~** done as D9. Still open: 4-bit text encoder for faster prompt changes on
  small cards.
- **O4. Resolution and token count as a user-facing setting.** 384px (576 tokens) is about 0.55x the compute of 512.
  It is the most portable lever.
- **O5. 1–2 step student.**
  - Retrain or distill a student whose low-noise nodes suit img2img (e.g. rCM or TwinFlow style).
  - Removes D4's narrow strength range and cuts NFE; the largest lever overall, but a training project.
- **O6. Temporal consistency.** Try latent blending with the previous output, motion-aware start sigma
  (StreamDiffusionV2), or feature banks (StreamV2V).
- **O7. Serving.**
  - Threaded input/output queues (StreamDiffusion).
  - On ZeroGPU the GPU is attached per call, so frames have to be processed in chunks per call rather than in a
    persistent loop.
- **O8. Multi-GPU pipeline parallelism** (StreamDiffusionV2): split the blocks across GPUs, with steps in flight.
  Only relevant beyond one card.
