"""StreamDiffusion-style img2img streaming for Qwen-Image 2.1. Decision numbers (D1...) refer to DESIGN.md."""

import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift
from huggingface_hub import hf_hub_download

from .fast import CachedDenoiser
from .lora import apply_lora
from .tiny_vae import TinyVAE

BASE_MODEL = "Qwen/Qwen-Image-2.1"
TURBO_LORA = ("Viggle/Qwen-Image-2.1-viggle-turbo", "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors")
TURBO_SIGMAS = [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]  # raw nodes of the turbo student (D2)


class StreamQwen:
    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        start_node: float = 0.75,
        fuse_lora: bool = True,
        seed: int = 0,
        device: str = "cuda",
        text_encoder_device: str = "cpu",
        fp8: bool = False,
        compile: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device, self.text_encoder_device, self.dtype = device, text_encoder_device, dtype
        self.latent_hw = (height // 16, width // 16)
        num_tokens = self.latent_hw[0] * self.latent_hw[1]

        # D1: the full VAE is never loaded, TAEQI2.1 does both ends.
        self.pipe = QwenImage21Pipeline.from_pretrained(BASE_MODEL, vae=None, torch_dtype=dtype)
        # D9: only the DiT lives on the GPU by default; the 8B text encoder runs once per prompt wherever it is put.
        self.pipe.text_encoder.to(text_encoder_device)
        self.transformer = self.pipe.transformer.to(device)
        # D3: the turbo LoRA is fused by default, trading a little precision for no per-layer LoRA matmuls.
        apply_lora(self.transformer, hf_hub_download(*TURBO_LORA), fuse=fuse_lora)
        if fp8:  # D10: FP8 weights and activations (Ada, Hopper, Blackwell); needs the LoRA fused first
            from torchao.quantization import PerRow, float8_dynamic_activation_float8_weight, quantize_

            quantize_(self.transformer.transformer_blocks, float8_dynamic_activation_float8_weight(granularity=PerRow()))
        self.vae = TinyVAE.from_pretrained().to(device, dtype)

        # D2: the turbo schedule, time-shifted for this resolution exactly as the pipeline would, entered mid-way.
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config, shift_terminal=None)
        cfg = scheduler.config
        mu = calculate_shift(num_tokens, cfg.base_image_seq_len, cfg.max_image_seq_len, cfg.base_shift, cfg.max_shift)
        scheduler.set_timesteps(sigmas=TURBO_SIGMAS, mu=mu, device=device)
        self.schedule = scheduler.sigmas.float()  # shifted TURBO_SIGMAS plus the terminal 0
        self.set_start_node(start_node)

        # D5: one fixed noise sample for the whole stream.
        generator = torch.Generator(device).manual_seed(seed)
        self.noise = torch.randn(1, num_tokens, 64, generator=generator, device=device, dtype=torch.float32)

        self.img_shapes = [[(1, *self.latent_hw)]]
        self.kv_cache = QwenImage21KVCache(len(self.transformer.transformer_blocks))
        # D11: every cached step runs this lean target-only forward; compiled once for the stream's lifetime.
        self.denoiser = CachedDenoiser(self.transformer)
        self.step = torch.compile(self.denoiser.__call__, fullgraph=True, mode="reduce-overhead") if compile else self.denoiser

    def set_start_node(self, node: float):
        """D4: where img2img enters the schedule (a raw turbo node). Higher = more prompt, less input frame, more NFE."""
        self.sigmas = self.schedule[TURBO_SIGMAS.index(node) :]

    @torch.no_grad()
    def set_prompt(self, prompt: str):
        """D6: encode the prompt and prefill its K/V once; every later step of every frame reuses them."""
        embeds, _, image_pad_mask = self.pipe.encode_prompt(prompt, device=self.text_encoder_device)
        embeds, image_pad_mask = embeds.to(self.device), image_pad_mask.to(self.device)
        num_slots = self.noise.shape[1] // 4  # the transformer maps each image slot to 2x2 latent tokens
        self.prompt_embeds = embeds
        self.img_mask = torch.cat([image_pad_mask, image_pad_mask.new_ones(1, num_slots)], dim=1)
        self.kv_cache.clear()
        self._transformer(self.noise, self.sigmas[0], kv_cache_mode="extract")
        self.denoiser.set_prefix(self.img_mask, self.img_shapes, self.kv_cache)

    def _transformer(self, x, sigma, kv_cache_mode):
        """The stock diffusers forward; used only for the prefill."""
        return self.transformer(
            hidden_states=x.to(self.dtype),
            timestep=sigma.view(1).to(self.dtype),
            encoder_hidden_states=self.prompt_embeds,
            encoder_hidden_states_mask=None,  # D7: single unpadded prompt, so attention needs no mask
            img_shapes=self.img_shapes,
            img_mask=self.img_mask,
            kv_cache=self.kv_cache,
            kv_cache_mode=kv_cache_mode,
            return_dict=False,
        )[0]

    @torch.no_grad()
    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        """(H, W, 3) uint8 on device -> packed latents (1, h*w, 64)."""
        rgb = frame.permute(2, 0, 1)[None].to(self.dtype) / 255
        rgba = torch.cat([rgb, torch.ones_like(rgb[:, :1])], dim=1)  # the VAE is RGBA; camera frames are opaque
        latents = self.vae.encode(rgba)
        return latents.flatten(2).transpose(1, 2).float()

    @torch.no_grad()
    def denoise(self, latents: torch.Tensor) -> torch.Tensor:
        """D2/D4: SDEdit. Noise the frame's latents to the start node, then Euler-integrate the velocity to 0."""
        s0 = self.sigmas[0]
        x = (1 - s0) * latents + s0 * self.noise
        for s, s_next in zip(self.sigmas[:-1], self.sigmas[1:]):
            x = x + (s_next - s) * self.step(x.to(self.dtype), s.view(1).to(self.dtype)).float()
        return x

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Packed latents -> (H, W, 3) uint8 on device. Alpha is dropped."""
        latents = latents.transpose(1, 2).unflatten(2, self.latent_hw).to(self.dtype)
        rgb = self.vae.decode(latents)[0, :3]
        return (rgb * 255).round().to(torch.uint8).permute(1, 2, 0)

    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        return self.decode(self.denoise(self.encode(frame)))


class SimilarityFilter:
    """D8: StreamDiffusion's Stochastic Similarity Filter.

    Skips a frame with probability max(0, (sim - threshold) / (1 - threshold)), where `sim` is the cosine similarity
    to the last frame that was actually processed. A static camera then costs almost nothing.
    """

    def __init__(self, threshold: float = 0.98, size: int = 64):
        self.threshold, self.size, self.reference = threshold, size, None

    def should_skip(self, frame: torch.Tensor) -> bool:
        small = F.interpolate(frame.permute(2, 0, 1)[None].float(), size=self.size, mode="area").flatten()
        if self.reference is not None:
            sim = F.cosine_similarity(small, self.reference, dim=0).item()
            if torch.rand(1).item() < (sim - self.threshold) / (1 - self.threshold):
                return True
        self.reference = small
        return False
