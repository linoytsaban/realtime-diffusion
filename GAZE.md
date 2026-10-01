# Gaze-targeted edits

Goal: the edit follows where the viewer looks. Look at the mug, the mug turns to gold; the rest stays the camera.

## How it works

```
browser: camera -> MediaPipe face landmarks -> eye features -> calibrated (u, v) on the output
         -> One-Euro smoothing -> fixation? -> {"focus": {u, v, radius}} over the WebSocket
server:  focus -> soft disk mask on the latent grid -> after every denoising step, restore the frame outside it
```

- **Features** (`static/gaze.js`): each iris in its eye's own frame (normalized by eye width, so distance does not
  matter), head yaw and pitch, and the face position.
- **Calibration:** follow a dot moving over the output image for 12 s (smooth pursuit, which beat 9-point grids in
  [this study](https://arxiv.org/abs/2608.11566)). Ridge regression on second-order features maps them to (u, v).
- **Smoothing and fixation:** One-Euro filter, then velocity-threshold fixation (still for 150 ms). The focus is only
  sent on fixation, when it moved, at most 10 times per second, so saccades and jitter do not reach the GPU.
- **Edit restriction** (`qwen_stream/focus.py`, D18): outside the disk, the latents are reset to the frame noised to
  the current sigma after every step (standard inpainting). The model still sees the whole frame as condition, so the
  edit blends in. Costs no extra GPU work.

## Expected accuracy

Webcam gaze with face landmarks reaches about 3 to 5 degrees as actually deployed (WebGazer: 7 to 11),
roughly 100 to 200 px at laptop distance. So targets are regions or objects, not pixels; the default focus radius
is 30% of the image.

## Status

| part | status |
|---|---|
| focus mask and inpainting step | unit-tested on CPU: frame untouched outside, edited inside |
| gaze math (features, calibration, smoothing, fixation) | unit-tested in Node (`node --test tests/`) |
| page, MediaPipe, calibration, focus messages | headless Chromium with a fake camera: face tracked, calibration done, focus sent |
| server focus setting | checked on CPU with a stub engine |
| real gaze accuracy, edit quality at the boundary, speed | **untested**: needs a real person and the GPU |

## Options not built yet

- **Name the object** under the gaze: crop around (u, v), ask Qwen3-VL (already loaded) what it is, and rewrite the
  instruction ("Turn the mug into gold"). Needs a VLM generation per new fixation, about 0.3 to 1 s on GPU.
- **Snap to objects:** a segmentation mask of the object under the gaze instead of a disk. Hides gaze error.
- **Compute only the focus:** denoise just the tokens inside the disk. Could make focused edits faster than full ones.
- **Gaze in the scene** instead of on the screen (where the person in the frame looks). Harder: needs a 3D gaze ray.
