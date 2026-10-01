"""Lean decode-only forward of the Qwen-Image 2.1 DiT for the cached (target-only) path (DESIGN.md, D11).

The diffusers forward rebuilds token metadata on every call (`nonzero`, host-to-device `torch.tensor`,
`repeat_interleave`, rotary tables). In cached mode all of it is constant for a given prompt and resolution, so it is
computed once per prompt here (`set_prefix`), and a step is pure tensor math.

One instance lives for the whole stream and is compiled once. A new prompt swaps the prefix tensors; their length is
the only shape that changes, and torch.compile's automatic dynamic shapes generalize it after the second length seen,
so prompts of new lengths stop triggering recompiles.
"""

import torch
import torch.nn.functional as F


class CachedDenoiser:
    def __init__(self, transformer):
        self.t = transformer

    def set_prefix(self, img_mask: torch.Tensor, img_shapes, kv_cache):
        """Snapshot everything a cached step needs. Call after the prefix K/V were extracted into `kv_cache`."""
        transformer = self.t
        # Same joint layout as the diffusers forward: each image slot of the text sequence stands for 2x2 tokens.
        repeats = torch.where(img_mask, 4, 1)[0]
        image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)
        freqs = transformer.pos_embed(img_shapes[0], image_pad_mask, device=img_mask.device)
        _, height, width = img_shapes[0][-1]
        # The target is the last block. Its frame-axis position follows the text, so the table depends on the prompt.
        target = freqs[-height * width :]
        self.cos = target.real[:, None].float()  # (N, 1, D/2), broadcast over heads
        self.sin = target.imag[:, None].float()
        self.prefix_kv = [layer.get() for layer in kv_cache.layer_caches]  # post-rope (B, S, H, D) per block

    def rope(self, x):
        """Rotate interleaved (real, imag) pairs; the real-valued form of diffusers' complex `apply_rotary_emb_qwen`."""
        x_r, x_i = x.float().unflatten(-1, (-1, 2)).unbind(-1)
        out = torch.stack([x_r * self.cos - x_i * self.sin, x_r * self.sin + x_i * self.cos], dim=-1)
        return out.flatten(-2).to(x.dtype)

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """x: (1, N, 64) bf16 noisy latents, sigma: (1,) -> velocity (1, N, 64)."""
        t = self.t
        h = t.img_in(x)
        temb = t.time_text_embed(sigma, h)
        scale1, gate1, scale2, gate2 = t.modulation(temb)[:, None].chunk(4, dim=-1)  # shared by all blocks
        gate1, gate2 = gate1.tanh(), gate2.tanh()

        for block, (k_prefix, v_prefix) in zip(t.transformer_blocks, self.prefix_kv):
            attn = block.attn
            n = block.img_norm1(h) * (1 + scale1)
            q, k, v = (proj(n).unflatten(-1, (attn.heads, -1)) for proj in (attn.to_q, attn.to_k, attn.to_v))
            q, k = self.rope(attn.norm_q(q).to(v.dtype)), self.rope(attn.norm_k(k).to(v.dtype))
            k, v = torch.cat([k_prefix, k], dim=1), torch.cat([v_prefix, v], dim=1)
            # Target queries see the whole prefix and every target token, so no mask is needed (D7).
            o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
            h = h + gate1 * attn.to_out[0](o.transpose(1, 2).flatten(2))
            h = h + gate2 * block.img_mlp(block.img_norm2(h) * (1 + scale2))

        h = t.norm_out(h, temb)
        return t.proj_out(h)
