"""TAEQI2.1: madebyollin's tiny autoencoder for Qwen-Image 2.1 (16x spatial, 64 latent channels, RGBA).

Self-contained port of the `f16` variant of https://github.com/madebyollin/taesd (see DESIGN.md, D4).
Layer indices match the published `taeqi2_1.safetensors` keys, so the file loads with only a prefix rename.
"""

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Block(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.conv = nn.Sequential(conv(n, n), nn.ReLU(), conv(n, n), nn.ReLU(), conv(n, n))
        self.fuse = nn.ReLU()

    def forward(self, x):
        return self.fuse(self.conv(x) + x)


def stage(n, depth):
    return [Block(n) for _ in range(depth)]


class TinyVAE(nn.Module):
    """Encodes RGBA in [0, 1] to normalized Qwen-Image 2.1 latents and back.

    The latents are already in the normalized space the transformer works in, so no `latents_mean`/`latents_std`
    is applied (the full VAE needs them, this one was distilled against the normalized latents).
    """

    def __init__(self, latent_channels=64, image_channels=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.PixelUnshuffle(2), conv(image_channels * 4, 64), nn.ReLU(), *stage(64, 1),
            conv(64, 64, stride=2, bias=False), *stage(64, 3),
            conv(64, 128, stride=2, bias=False), *stage(128, 3),
            conv(128, 256, stride=2, bias=False), *stage(256, 3),
            conv(256, latent_channels),
        )  # fmt: skip
        self.decoder = nn.Sequential(
            conv(latent_channels, 256), nn.ReLU(), *stage(256, 3),
            nn.Upsample(scale_factor=2), conv(256, 128, bias=False), *stage(128, 3),
            nn.Upsample(scale_factor=2), conv(128, 64, bias=False), *stage(64, 3),
            nn.Upsample(scale_factor=2), conv(64, 64, bias=False), *stage(64, 1),
            conv(64, image_channels * 4), nn.PixelShuffle(2),
        )  # fmt: skip

    @classmethod
    def from_pretrained(cls, repo_id="madebyollin/taeqi2_1", filename="taeqi2_1.safetensors"):
        model = cls()
        # The file uses diffusers' `AutoencoderTiny` naming (`encoder.layers.N`); indices already match ours.
        state_dict = load_file(hf_hub_download(repo_id, filename))
        model.load_state_dict({k.replace(".layers.", ".", 1): v for k, v in state_dict.items()})
        return model.eval().requires_grad_(False)

    def encode(self, rgba: torch.Tensor) -> torch.Tensor:
        """(B, 4, H, W) in [0, 1] -> (B, 64, H/16, W/16)."""
        return self.encoder(rgba)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """(B, 64, h, w) -> (B, 4, 16h, 16w) in [0, 1]. The tanh soft-clamp matches TAESD's `Clamp` layer."""
        return self.decoder(torch.tanh(latents / 3) * 3).clamp(0, 1)
