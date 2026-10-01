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

Two modes. **Edit** (default) treats Qwen-Image 2.1 as the image-editing model it is. **img2img** is the
StreamDiffusion-style SDEdit path.

```
edit:     frame -> TAEQI encode -> prefill prefix [instruction, frame as condition image] (D13, D14)
                -> pure fixed noise -> 4-6 Euler steps on the target -> TAEQI decode
img2img:  frame -> TAEQI encode -> noise to a mid-schedule sigma (fixed noise) -> 2-4 Euler steps -> TAEQI decode
          (the text-only prefix is prefilled once per prompt)
prompt change -> Qwen3-VL encode once (edit: with the current frame as the VLM's reference image)
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

Presets in `SCHEDULES` (`img2img-subtle/medium/strong`) pick the entry node at runtime. The usable range is narrow,
and quantized to the student's nodes. **Prompt adherence is weak in this mode**: nothing but the noise level trades
the frame against the prompt (no CFG, no RCFG). Edit mode (D13) fixes that.

**D5. One fixed noise tensor for the whole stream.**
StreamDiffusion does the same. With the same noise every frame, a static input gives a static output, which cuts
flicker. Cost: patterns in that noise stay fixed in screen space ("texture swimming" when the camera moves). See O6.

**D6. Prompt encoding and prefix K/V are computed once per prompt, not per frame or per step.**
Qwen-Image 2.1 modulates text tokens at t=0 (`causal_condition`) and text never attends to image tokens, so the text
K/V are independent of the frame and the timestep. `set_prompt` runs Qwen3-VL (~0.3 s) and one `extract` pass. Every
step of every frame then runs in `cached` mode, computing only the 1024 image tokens.
In edit mode the prefix contains the frame, so it is re-prefilled per frame (D13, D14); the VLM still runs once.

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
Each step runs `CachedDenoiser`. It precomputes the target-token rotary table (as real cos/sin) once per prompt,
takes the per-layer prefix K/V from the prefill (D14), then runs `img_in` -> one shared modulation (all 32 blocks share it) -> 32 blocks -> `norm_out` / `proj_out`.
There is no data-dependent indexing and no host sync, so it compiles with `fullgraph=True`.
- **Parity:** vs the diffusers cached forward, 0.6% relative difference (bf16 rounding over 32 layers).
- **Cost:** about 50 lines that duplicate the block math. It has to track upstream changes to the 2.1 block.
- **One instance for the stream's lifetime.** A prompt change swaps the per-prompt tensors (`set_layout`, `set_kv`).
  - torch.compile's automatic dynamic shapes generalize the prefix length after the second length seen.
  - Measured: the first two prompt lengths each compile once (~1 min). After that, a new length costs nothing extra,
    and steady speed is unchanged (~101 ms/frame).
  - `live.py` warms up with two lengths at startup.
- **Compiled with `mode="reduce-overhead"` (CUDA graphs).** Single-threaded, graphs gain nothing (30.4 vs 30.7 ms per
  pass with FP8). But with one busy Python thread competing for the GIL, the graph-less compiled pass collapses to
  2617 ms, while the graphed one holds at 47 ms. A server always has other threads, so graphs are required.
- **The graphs re-read swapped prefix tensors.** Replacing the prefix K/V with zero tensors of the same shape changes
  the compiled output by 101%. It then matches eager with the same zero K/V to 2.6% (FP8 rounding), so a prompt swap
  cannot leave a stale baked-in prefix.

**D12. Live server (`live.py` + `static/index.html`): newest frame only, two frames in flight, one GPU thread.**
- **Newest frame wins.** The browser sends JPEG frames over a WebSocket. The server keeps only the newest frame and
  the newest settings, so a slow step drops frames instead of building latency.
- **Two frames in flight.** The page sends the next frame when a result arrives, with up to 2 frames in flight. One
  frame hides the network round trip while the other is on the GPU. Through the cloudflared tunnel this measured
  9.5 FPS, vs 5.7 FPS with one frame in flight (~67 ms round trip).
- **One dedicated GPU thread.** It runs warmup, prompt changes and every frame. CUDA graphs in torch.compile are
  recorded per thread: running frames on `asyncio.to_thread`'s pool hung on the first frame.
- **JPEG on the GPU.** `torchvision.io` decodes and encodes on the GPU (~1 ms per frame in total).
- **Single user, by design.** `StreamQwen` holds one prompt cache. A new connection takes over from the old one
  (page reloads, a second tab), and the page re-sends its mode, prompt and schedule on connect.
- **Prompt changes apply on the next frame,** because edit mode needs a real frame as the VLM's reference (D13).
- **Gated.** The page sits behind a random URL token, because it is exposed through a public quick tunnel.
- **Measured locally, end to end:** 9.2 FPS with 103 ms of GPU work per frame, flat from 1 to 3 frames in flight.
  Before the single GPU thread, GIL contention pushed it from 104 to 129 ms.
- **With edit mode (D13, D14), end to end:** edit-4 5.3 FPS (179 ms GPU), edit-6 3.9 FPS (249 ms), img2img-medium
  9.1 FPS (104 ms). A prompt change takes ~5 s in edit mode and ~0.8 s in img2img (text encoder on CPU).

**D13. Edit mode: the frame is the condition image and the prompt is an instruction; denoising starts from pure noise.**
Qwen-Image 2.1 is an image-editing model, and the Viggle turbo LoRA was distilled on edits too. StreamDiffusion's
RCFG self-negative exists only because SD-Turbo is text-to-image; here the edit training does that job.
Sweep on frame 40 of the cat clip, eager bf16:

| setting | adherence | structure | eager ms |
|---|---|---|---|
| img2img 0.75 (D4) | weak, mushy | partly lost | 171 |
| **edit, pure noise (1.0)** | **excellent: real watercolor, real lego** | **pose and framing exact** | 416 (6 steps) |
| edit, noised frame as start (0.875 and below) | none: near copy of the input | kept | 179-300 |

- **Start from pure noise.** With the frame as both the condition and the starting point, the model reconstructs it,
  so SDEdit and edit do not combine.
- **Steps.** Step sweep on the watercolor and lego instructions:
  - `edit-3` = `[1, .75, .25]` (default) is nearly identical to `edit-4`; the lego is a touch softer.
  - `[1, .5, .25]` is decent, with a little less detail.
  - `[1, .75, .5]` is blurry, because it stops too early.
  - 2 steps hold up for watercolor, but the lego falls apart.
  - `edit-4` = `[1, .75, .5, .25]` is the student's training schedule; `edit-6` is the card's.
- **VLM once per prompt.** The DiT reads the condition image from its VAE latents; the VLM only shapes the text
  tokens that follow the image. Reusing the embeddings from frame 0 on frame 40 changed the output by
  0.26-0.34/255 (mean) vs re-running the VLM on frame 40. So the 8B VLM never runs per frame.
- **The prompt changes cost more:** ~5 s on CPU, because the VLM now also processes the reference image (D9).
- **Determinism.** The edit template picks its image label ("Picture"/"Image"/"图"/"图片") at random; it is pinned
  to "Picture".
- **Temporal behavior.** Fixed noise (D5) plus a condition frame gave a stable background and style across the clip,
  and fine motion (the tongue) followed the input.

**D14. Lean prefill (`CachedDenoiser.prefill`).**
`set_layout` precomputes, once per prompt:
- the text rows (`txt_in` of the prompt embeddings) and fixed scatter indices for the image rows;
- the prefix rotary table and block-causal mask (causal, with the condition image bidirectional);
- the t=0 modulation of the prefix, which is a constant.

Per frame, `prefill` is then `img_in(frame)`, a scatter, and 32 blocks that emit K/V; compiled like D11.
It also serves img2img (empty image part), so the diffusers forward is no longer used at runtime.
- **Parity vs the diffusers `extract` pass:** prefix K/V within 1.1-2.6% per layer (bf16), first-step velocity
  within 0.4% (img2img) and 0.6% (edit).
- **Speed:** the diffusers prefill-plus-first-step pass took ~200 ms eager; the lean prefill is ~35 ms, about one
  step. Edit at 4 steps went from 274 to 175 ms per frame.
- **CUDA-graph buffers.** Graph outputs are reused by the next replay, so the prefill's K/V are cloned out right away
  (~0.5 GB of copies per frame in edit mode, ~0.3 ms).

**D15. cuDNN attention for every unmasked attention call (`attention="cudnn"`; `"flash"` = FlashAttention-2).**
In edit mode the steps attend over ~2072 keys, so attention is ~25% of the frame. Backends, measured on an idle GPU:
- **cuDNN:** built into PyTorch, Ampere through Blackwell. -2 ms per step vs FA2.
- **FlashAttention-3** (HF kernels Hub): Hopper only, so not a portable default. Untested end to end.
- **SageAttention 2.2** (8-bit): wrong output on the 1048-key prefill shape (rel err 19.7), likely a block-size
  edge case. Rejected unless padded.

**D16. Exact split of the prefill attention.**
The block-causal mask forces PyTorch's memory-efficient kernel. But the 1024 image rows see every key up to the end
of the image block, so they need no mask: they run on the fast backend, and only the ~25 text rows keep the mask.
The prefill went from 35.3 to 32.0 ms.

**D17. Fused projections (`fuse_projections`).**
Q/K/V become one GEMM and the MLP's gate/up become one, instead of 3 + 2 per block. This is done after the LoRA is
fused and before FP8. The original layers are deleted (no duplicate weights), so the stock diffusers forward no
longer runs on the transformer, and the unfused-LoRA hook path (D3) is offline-only. Gain: ~0.5 ms per pass.

**FP8 granularity (D10).** Per-tensor scaling saves ~3 ms per frame over per-row (125.5 vs 128.4 ms denoise for
edit-3). The images look equally good, with a mean difference of 4.2/255 that reads as a different rendering rather
than an artifact. Per-row stays the default, as the safer choice for activation outliers; `--fp8 tensor` is opt-in.

**Kernel sweep, edit-3 GPU time per frame (idle H100, CUDA graphs):**

| variant | prefill | step | frame | step drift vs baseline |
|---|---|---|---|---|
| FP8 per-row, FA2, masked prefill (before) | 35.3 | 34.3 | 138.2 | - |
| + cuDNN attention (D15) | 35.8 | 32.2 | 132.4 | 1.8% |
| + split prefill (D16) | 32.0 | 31.9 | 127.8 | 2.1% |
| + fused projections (D17) | 32.1 | 31.4 | 126.3 | 2.6% |
| + FP8 per-tensor | 29.1 | 30.9 | 121.7 | - |

The drift is the same order as FP8's own (1.4%) and not visible.

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
| edit-4 (D13), diffusers prefill, compiled + FP8 | 2.5 ms | 269 ms | 2.7 ms | 274 ms | 3.7 |
| **edit-4, lean prefill (D14), compiled + FP8** | 1.8 ms | 175 ms | 1.9 ms | 179 ms | **5.6** |
| edit-6, lean prefill, compiled + FP8 | 1.8 ms | 245 ms | 1.9 ms | 249 ms | 4.0 |
| img2img-medium, lean prefill, compiled + FP8 | 2.6 ms | 97 ms | 2.9 ms | 102 ms | 9.8 |
| **edit-3 + D15-D17, compiled + FP8 per-row** | 2.6 ms | 128 ms | 2.8 ms | 134 ms | **7.5** |
| edit-3 + D15-D17, compiled + FP8 per-tensor | 2.6 ms | 126 ms | 2.8 ms | 131 ms | 7.6 |

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

**Quality in img2img mode is weak** (D4); edit mode (D13) trades ~1.8x the per-frame cost for real prompt adherence.

With `--compile`, the reported `set_prompt` time includes compile warmup (minutes), because the first compiled pass
runs there.

With the text encoder on the GPU, peak memory is 30.4 GiB (14 GB DiT plus 17.5 GB text encoder). With D9 it is
14.0 GiB, and the speed is unchanged (6.0 FPS eager, 3 NFE, under a 24 GB cap).

Profile of one cached pass (eager, ~48 ms GPU):
- GEMMs take 28 ms, i.e. ~500 TFLOP/s: already about 2/3 of bf16 peak on this card.
- The other ~20 ms is elementwise and dtype-conversion kernels.
- The CPU needs 54 ms to launch a pass whose GPU work is 48 ms, so launch overhead is a bottleneck.

## Optimization backlog (roughly in priority order)

- **O1. ~~Lean decode-only forward + CUDA graphs~~** done as D11. Graphs are what keep the server stable under GIL
  contention; they should also help hosts with weak CPUs (Spark's ARM cores).
- **O2. Low-precision DiT, chosen per GPU generation.**
  - FP8 weights and activations on Ada, Hopper and Blackwell. That covers the 4090.
  - NVFP4 on Blackwell (RTX Pro 6000, Spark).
  - Halves or quarters both the GEMM time and the 14 GB of weights; on Spark, bandwidth is the limit.
  - Needs a quality check against bf16 at 3 NFE.
- **O3. ~~Text encoder off the GPU budget~~** done as D9. Still open: 4-bit text encoder for faster prompt changes on
  small cards.
- **O4. Resolution and token count as a user-facing setting.** 384px (576 tokens) is about 0.55x the compute of 512.
  It is the most portable lever.
- **O5. Fewer steps for edit mode.** It always starts from pure noise, so steps are the cost.
  - Try Pruna's 5-step edit LoRA.
  - Longer term, a 1-2 step edit student distilled for streaming (e.g. rCM or TwinFlow style).
- **O5b. 1–2 step student (img2img).**
  - Retrain or distill a student whose low-noise nodes suit img2img (e.g. rCM or TwinFlow style).
  - Removes D4's narrow strength range and cuts NFE; the largest lever overall, but a training project.
- **O6. Temporal consistency.** Try latent blending with the previous output, motion-aware start sigma
  (StreamDiffusionV2), or feature banks (StreamV2V).
- **O7. Serving.**
  - The local live server is D12.
  - On ZeroGPU the GPU is attached per call, so frames have to be processed in chunks per call rather than in a
    persistent loop.
- **O10. More kernel work.**
  - FA3 as a Hopper-only opt-in.
  - SageAttention with padded key lengths (8-bit attention, Ada/Hopper; Sage3 FP4 on Blackwell).
  - The FP8 GEMMs still run at ~40% of FP8 peak: try autotuned (max-autotune) GEMM templates.
- **O9. Warmup time.** `live.py` compiles both layouts (img2img and edit) at two prompt lengths: ~11 min before the
  first frame. Try torch.compile's persistent caches (inductor FX graph cache, `torch.compiler.save_cache_artifacts`)
  or warming up only the default mode.
- **O8. Multi-GPU pipeline parallelism** (StreamDiffusionV2): split the blocks across GPUs, with steps in flight.
  Only relevant beyond one card.
