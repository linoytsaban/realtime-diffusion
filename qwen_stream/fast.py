"""Lean forwards of the Qwen-Image 2.1 DiT for streaming (DESIGN.md, D11 and D14).

The diffusers forward rebuilds token metadata on every call (`nonzero`, host-to-device `torch.tensor`,
`repeat_interleave`, rotary tables). For a stream all of it is constant per prompt, so `set_layout` computes it once.
Per call only tensor math remains:

- `prefill`: the prefix K/V. Text only in img2img (once per prompt); text plus the current frame as condition image
  in edit mode (once per frame, D14).
- `__call__`: one denoising step over the target tokens, attending to the cached prefix K/V (D11).

Kernel choices: cuDNN attention (D15), an unmasked split of the prefill attention (D16), fused projections (D17).

One instance lives for the whole stream and is compiled once. A new prompt swaps the per-prompt tensors; the prefix
length is the only shape that changes, and torch.compile's automatic dynamic shapes generalize it after the second
length seen, so prompts of new lengths stop triggering recompiles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def fuse_projections(transformer):
    """D17: one GEMM for Q/K/V and one for the MLP's gate/up, instead of three and two. Run before FP8 quantization.

    The original Linear layers are deleted, so the stock diffusers forward no longer works on this transformer.
    """

    def fuse(layers):
        fused = nn.Linear(layers[0].in_features, sum(l.out_features for l in layers), bias=False)
        fused.weight.data = torch.cat([l.weight.data for l in layers])
        return fused.to(layers[0].weight.device)

    for block in transformer.transformer_blocks:
        attn, mlp = block.attn, block.img_mlp
        attn.to_qkv = fuse([attn.to_q, attn.to_k, attn.to_v])
        mlp.gate_up = fuse([mlp.gate_layer, mlp.proj])
        del attn.to_q, attn.to_k, attn.to_v, mlp.gate_layer, mlp.proj


class CachedDenoiser:
    def __init__(self, transformer, attention: str = "cudnn"):
        self.t = transformer
        # D15: the backend for unmasked attention (every step, and the image rows of the prefill). cuDNN runs on
        # Ampere through Blackwell and measured fastest here; "flash" (FlashAttention-2) is the PyTorch default.
        self.backend = {"cudnn": SDPBackend.CUDNN_ATTENTION, "flash": SDPBackend.FLASH_ATTENTION}[attention]

    @torch.no_grad()
    def set_layout(self, prompt_embeds: torch.Tensor, img_mask: torch.Tensor, img_shapes):
        """Everything that depends only on the prompt layout, not on any frame."""
        t = self.t
        # Same joint layout as the diffusers forward: each image slot of the text sequence stands for 2x2 tokens.
        repeats = torch.where(img_mask, 4, 1)[0]
        is_image = torch.repeat_interleave(img_mask[0], repeats)
        freqs = t.pos_embed(img_shapes[0], is_image, device=img_mask.device)
        _, height, width = img_shapes[0][-1]
        prefix_len = is_image.numel() - height * width  # the target image is the last block

        # Target tokens: their frame-axis position follows the text, so this table depends on the prompt.
        self.cos, self.sin = self._rope_table(freqs[prefix_len:])

        # Prefix: text rows come from the prompt; condition-image rows are filled per frame at fixed positions.
        self.prefix_cos, self.prefix_sin = self._rope_table(freqs[:prefix_len])
        prefix_is_image = is_image[:prefix_len]
        self.text_index = (~prefix_is_image).nonzero()[:, 0]
        self.image_index = prefix_is_image.nonzero()[:, 0]
        # The condition image is one contiguous block (edit mode); its rows are attended without a mask (D16).
        self.image_span = (int(self.image_index[0]), int(self.image_index[-1]) + 1) if len(self.image_index) else None
        text_rows = ~img_mask[0, : prompt_embeds.shape[1]]  # image slots are replaced by image tokens
        self.text_hidden = t.txt_in(prompt_embeds[:, text_rows])
        # Block-causal mask: causal, except that the tokens of the condition image see each other (bidirectional).
        position = torch.arange(prefix_len, device=img_mask.device)
        block = torch.where(prefix_is_image, 0, -1 - position)
        self.prefix_mask = (position[:, None] >= position[None]) | (block[:, None] == block[None])
        # Prefix tokens are modulated from t = 0 (`causal_condition`), so their modulation is a constant.
        zero = torch.zeros(1, device=img_mask.device, dtype=self.text_hidden.dtype)
        self.prefix_modulation = self._modulation(t.time_text_embed(zero, self.text_hidden))

    def set_kv(self, prefix_kv):
        """Point the cached steps at prefix K/V: one post-rope (B, S, H, D) pair per block."""
        self.prefix_kv = prefix_kv

    @staticmethod
    def _rope_table(freqs):
        return freqs.real[:, None].float(), freqs.imag[:, None].float()  # (N, 1, D/2), broadcast over heads

    @staticmethod
    def _rope(x, cos, sin):
        """Rotate interleaved (real, imag) pairs; the real-valued form of diffusers' complex `apply_rotary_emb_qwen`."""
        x_r, x_i = x.float().unflatten(-1, (-1, 2)).unbind(-1)
        out = torch.stack([x_r * cos - x_i * sin, x_r * sin + x_i * cos], dim=-1)
        return out.flatten(-2).to(x.dtype)

    def _modulation(self, temb):
        scale1, gate1, scale2, gate2 = self.t.modulation(temb)[:, None].chunk(4, dim=-1)  # shared by all blocks
        return scale1, gate1.tanh(), scale2, gate2.tanh()

    def _qkv(self, block, h, scale1, cos, sin):
        attn = block.attn
        n = block.img_norm1(h) * (1 + scale1)
        q, k, v = attn.to_qkv(n).unflatten(-1, (3, attn.heads, -1)).unbind(-3)  # fused projection (D17)
        return self._rope(attn.norm_q(q).to(v.dtype), cos, sin), self._rope(attn.norm_k(k).to(v.dtype), cos, sin), v

    @staticmethod
    def _mlp(block, x):
        gate, up = block.img_mlp.gate_up(x).chunk(2, dim=-1)  # fused projection (D17)
        return block.img_mlp.out(F.silu(gate) * up)

    def _attend(self, q, k, v, mask=None):
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is None:
            with sdpa_kernel([self.backend]):
                out = F.scaled_dot_product_attention(q, k, v)
        elif self.image_span is None:  # text-only prefix: a handful of tokens
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            # D16, exact: the image rows see every key up to the end of the image with no mask, so they get the fast
            # backend; only the few text rows need the block-causal mask.
            a, b = self.image_span
            with sdpa_kernel([self.backend]):
                image = F.scaled_dot_product_attention(q[:, :, a:b], k[:, :, :b], v[:, :, :b])
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                text_q = torch.cat([q[:, :, :a], q[:, :, b:]], dim=2)
                text = F.scaled_dot_product_attention(text_q, k, v, attn_mask=torch.cat([mask[:a], mask[b:]]))
            out = torch.cat([text[:, :, :a], image, text[:, :, a:]], dim=2)
        return out.transpose(1, 2).flatten(2)

    def prefill(self, cond: torch.Tensor | None = None):
        """cond: (1, N, 64) bf16 condition-image latents, or None for a text-only prefix -> K/V per block."""
        t = self.t
        h = self.text_hidden.new_empty(1, self.prefix_mask.shape[0], self.text_hidden.shape[-1])
        h[:, self.text_index] = self.text_hidden
        if cond is not None:
            h[:, self.image_index] = t.img_in(cond)
        scale1, gate1, scale2, gate2 = self.prefix_modulation
        prefix_kv = []
        for block in t.transformer_blocks:
            q, k, v = self._qkv(block, h, scale1, self.prefix_cos, self.prefix_sin)
            prefix_kv.append((k, v))
            h = h + gate1 * block.attn.to_out[0](self._attend(q, k, v, self.prefix_mask))
            h = h + gate2 * self._mlp(block, block.img_norm2(h) * (1 + scale2))
        return prefix_kv

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """x: (1, N, 64) bf16 noisy latents, sigma: (1,) -> velocity (1, N, 64)."""
        t = self.t
        h = t.img_in(x)
        temb = t.time_text_embed(sigma, h)
        scale1, gate1, scale2, gate2 = self._modulation(temb)

        for block, (k_prefix, v_prefix) in zip(t.transformer_blocks, self.prefix_kv):
            q, k, v = self._qkv(block, h, scale1, self.cos, self.sin)
            k, v = torch.cat([k_prefix, k], dim=1), torch.cat([v_prefix, v], dim=1)
            # Target queries see the whole prefix and every target token, so no mask is needed (D7).
            h = h + gate1 * block.attn.to_out[0](self._attend(q, k, v))
            h = h + gate2 * self._mlp(block, block.img_norm2(h) * (1 + scale2))

        h = t.norm_out(h, temb)
        return t.proj_out(h)
