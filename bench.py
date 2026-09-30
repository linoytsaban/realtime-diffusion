"""Stream a video through StreamQwen, report per-stage latency, and write an input|output comparison video."""

import argparse
import time

import av
import numpy as np
import torch
import torch.nn.functional as F

from qwen_stream import StreamQwen


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
    parser.add_argument("--start-node", type=float, default=0.75, choices=[1.0, 0.9375, 0.875, 0.75, 0.5, 0.25])
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--no-fuse", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--text-encoder-device", default="cpu")
    parser.add_argument("--max-memory-gb", type=float, help="cap GPU memory to emulate a smaller card")
    parser.add_argument("--out", default="out.mp4")
    args = parser.parse_args()

    if args.max_memory_gb:
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(args.max_memory_gb / total)
    stream = StreamQwen(
        args.size, args.size, start_node=args.start_node, fuse_lora=not args.no_fuse,
        text_encoder_device=args.text_encoder_device,
    )  # fmt: skip
    if args.compile:
        stream.transformer.compile(fullgraph=False, dynamic=False)
    _, prompt_ms = timed(stream.set_prompt, args.prompt)
    frames = read_frames(args.video, args.size, args.frames)

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
    print(f"{args.size}px, {nfe} NFE, fused={not args.no_fuse}, compile={args.compile}")
    print(f"set_prompt (once per prompt): {prompt_ms:.0f} ms")
    for name, values in stages.items():
        print(f"  {name:8s} {np.median(values):7.1f} ms (median)")
    print(f"  total    {total:7.1f} ms -> {1000 / total:.1f} FPS  (peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)")
    write_video(args.out, outputs, fps=24)


if __name__ == "__main__":
    main()
