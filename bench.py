"""Stream a video through StreamQwen, report per-stage latency, and write an input|output comparison video."""

import argparse
import time

import av
import numpy as np
import torch
import torch.nn.functional as F

from qwen_stream import StreamQwen
from qwen_stream.stream import SCHEDULES


def read_frames(path, size, limit):
    """Center-crop to square, resize to `size`, return (H, W, 3) uint8 CUDA tensors."""
    frames = []
    with av.open(path) as container:
        for frame in container.decode(video=0):
            x = torch.from_numpy(frame.to_ndarray(format="rgb24")).cuda().permute(2, 0, 1).float()
            side = min(x.shape[1:])
            top, left = (x.shape[1] - side) // 2, (x.shape[2] - side) // 2
            x = F.interpolate(x[None, :, top : top + side, left : left + side], size=size, mode="area")[0]
            frames.append(x.round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous())
            if len(frames) == limit:
                break
    return frames


def write_video(path, frames, fps):
    with av.open(path, "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.height, stream.width = frames[0].shape[:2]
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        container.mux(stream.encode())


def timed(fn, *args):
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn(*args)
    torch.cuda.synchronize()
    return out, (time.perf_counter() - start) * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--prompt", default="a watercolor painting, soft pastel colors, paper texture")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--schedule", default="img2img-medium", choices=list(SCHEDULES))
    parser.add_argument("--edit", action="store_true", help="frame as condition image; the prompt is an instruction")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--attention", default="cudnn", choices=["cudnn", "flash"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--text-encoder-device", default="cpu")
    parser.add_argument("--fp8", choices=["row", "tensor"], help="FP8 DiT with per-row or per-tensor scaling (D10)")
    parser.add_argument("--max-memory-gb", type=float, help="cap GPU memory to emulate a smaller card")
    parser.add_argument("--out", default="out.mp4")
    args = parser.parse_args()

    if args.max_memory_gb:
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(args.max_memory_gb / total)
    stream = StreamQwen(
        args.size, args.size, schedule=args.schedule, attention=args.attention,
        text_encoder_device=args.text_encoder_device, fp8=args.fp8, compile=args.compile,
    )  # fmt: skip
    frames = read_frames(args.video, args.size, args.frames)
    _, prompt_ms = timed(stream.set_prompt, args.prompt, frames[0] if args.edit else None)

    for frame in frames[:3]:  # warmup: cudnn autotune, compile
        stream(frame)

    stages = {"encode": [], "denoise": [], "decode": []}
    outputs = []
    for frame in frames:
        latents, t_enc = timed(stream.encode, frame)
        latents, t_den = timed(stream.denoise, latents)
        image, t_dec = timed(stream.decode, latents)
        for name, ms in zip(stages, (t_enc, t_den, t_dec)):
            stages[name].append(ms)
        outputs.append(torch.cat([frame, image], dim=1).cpu().numpy())

    nfe = len(stream.sigmas) - 1
    total = np.sum([np.median(v) for v in stages.values()])
    print(f"{args.size}px, {args.schedule}, edit={args.edit}, {nfe} NFE, compile={args.compile}, fp8={args.fp8}")
    print(f"set_prompt (once per prompt): {prompt_ms:.0f} ms")
    for name, values in stages.items():
        print(f"  {name:8s} {np.median(values):7.1f} ms (median)")
    print(f"  total    {total:7.1f} ms -> {1000 / total:.1f} FPS  (peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)")
    write_video(args.out, outputs, fps=24)


if __name__ == "__main__":
    main()
