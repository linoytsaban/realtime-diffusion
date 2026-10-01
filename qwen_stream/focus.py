"""Gaze focus (GAZE.md): limit the edit to a soft disk around where the viewer looks."""

import torch


def focus_mask(u: float, v: float, radius: float, latent_hw: tuple[int, int], feather: float = 0.08, device="cpu"):
    """Soft disk on the latent grid, as (1, h*w, 1) weights in [0, 1].

    `u`, `v` are the gaze point and `radius`/`feather` its size, all as fractions of the image width/height.
    1 inside the disk, 0 outside, with a linear ramp of width `feather` so the edit blends into the frame.
    """
    h, w = latent_hw
    ys = (torch.arange(h, device=device) + 0.5) / h
    xs = (torch.arange(w, device=device) + 0.5) / w
    distance = torch.hypot(ys[:, None] - v, xs[None, :] - u)
    weight = ((radius - distance) / feather + 0.5).clamp(0, 1)
    return weight.reshape(1, h * w, 1)


def restore_outside(x: torch.Tensor, frame_latents: torch.Tensor, noise: torch.Tensor, sigma, mask: torch.Tensor):
    """Inpainting step: outside the mask, replace `x` with the frame's own latents noised to `sigma`.

    Applied after every denoising step, so at sigma 0 the output equals the frame outside the mask, while the model
    sees (and harmonizes with) consistent surroundings at every noise level.
    """
    known = (1 - sigma) * frame_latents + sigma * noise
    return mask * x + (1 - mask) * known
