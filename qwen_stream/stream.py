"""Real-time streaming edit and img2img for Qwen-Image 2.1."""

from typing import NamedTuple

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift
from huggingface_hub import hf_hub_download

from .fast import CachedDenoiser, fuse_projections
from .lora import apply_lora
from .tiny_vae import TinyVAE

BASE_MODEL = "Qwen/Qwen-Image-2.1"


class LoRA(NamedTuple):
    weights: tuple[str, str]  # (Hub repo, file)
    shift: bool  # True: schedule nodes get the pipeline's resolution-dependent shift; False: used exactly as given
    presets: dict = {}  # this LoRA's own versions of SCHEDULES entries, or extra entries


# Schedules are raw sigma nodes. Edit starts from pure noise (1.0); img2img enters mid-way from the noised frame.
SCHEDULES = {
    "edit-2": [1.0, 0.5],  # the 2-step LoRA's training nodes (default)
    "edit-3": [1.0, 0.75, 0.25],
    "edit-4": [1.0, 0.75, 0.5, 0.25],
    "edit-6": [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25],  # Viggle's 6-step nodes
    "img2img-subtle": [0.5, 0.25],
    "img2img-medium": [0.75, 0.5, 0.25],
    "img2img-strong": [0.875, 0.75, 0.5, 0.25],
}
# Few-step LoRAs that can be fused into the transformer.
PRUNA, PRUNA_SIGMAS = "PrunaAI/P-Q-Image-2.1", [1.0, 0.925, 0.85, 0.625, 0.375, 0.125]  # v0.2, the card's schedule
LORAS = {
    "dmd2-2step": LoRA(("linoyts/Qwen-Image-2.1-DMD2-2steps", "pytorch_lora_weights_v1_5.safetensors"), shift=False),
    "pruna-6": LoRA((PRUNA, "p_q_image_2.1_6_steps_v0.2.safetensors"), shift=True, presets={"edit-6": PRUNA_SIGMAS}),
    "pruna-6-upsampled": LoRA((PRUNA, "p_q_image_2.1_6_steps_upsampled_v0.2.safetensors"), shift=True,
                              presets={"edit-6": PRUNA_SIGMAS}),
    "viggle": LoRA(("Viggle/Qwen-Image-2.1-viggle-turbo", "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"), shift=True),
    "lightx2v": LoRA(("lightx2v/Qwen-Image-2.1-Lightning", "pytorch_lora_weights.safetensors"), shift=True),
}  # fmt: skip


class StreamQwen:
    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        schedule: str = "edit-2",
        seed: int = 0,
        device: str = "cuda",
        text_encoder_device: str = "cpu",
        fp8: str | None = None,
        compile: bool = False,
        attention: str = "cudnn",
        lora: str = "dmd2-2step",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device, self.text_encoder_device, self.dtype = device, text_encoder_device, dtype
        self.latent_hw = (height // 16, width // 16)
        num_tokens = self.latent_hw[0] * self.latent_hw[1]

        # The full VAE is never loaded: the tiny VAE does both ends.
        self.pipe = QwenImage21Pipeline.from_pretrained(BASE_MODEL, vae=None, torch_dtype=dtype)
        # Only the DiT has to live on the GPU; the 8B text encoder runs once per prompt wherever it is put.
        self.pipe.text_encoder.to(text_encoder_device)
        self.pipe.ref_token_list = ["Picture "]  # the edit template otherwise picks its image label at random
        self.transformer = self.pipe.transformer.to(device)
        # The LoRA is fused into the weights: a little precision traded for no per-layer LoRA matmuls.
        weights, self.shift, presets = LORAS[lora]
        self.presets = {**SCHEDULES, **presets}
        apply_lora(self.transformer, hf_hub_download(*weights))
        fuse_projections(self.transformer)  # after the LoRA (it targets to_q/to_k/...), before FP8
        if fp8:  # FP8 weights and activations (Ada, Hopper, Blackwell), scaled per "row" or per "tensor"
            from torchao.quantization import PerRow, PerTensor, float8_dynamic_activation_float8_weight, quantize_

            granularity = {"row": PerRow(), "tensor": PerTensor()}[fp8]
            quantize_(self.transformer.transformer_blocks, float8_dynamic_activation_float8_weight(granularity=granularity))
        self.vae = TinyVAE.from_pretrained().to(device, dtype)

        # Shifted LoRAs get the pipeline's resolution-dependent time shift, computed by its own scheduler.
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config, shift_terminal=None)
        if not self.shift:
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config, use_dynamic_shifting=False, shift=1.0, shift_terminal=None
            )
        cfg = self.scheduler.config
        self.mu = calculate_shift(num_tokens, cfg.base_image_seq_len, cfg.max_image_seq_len, cfg.base_shift, cfg.max_shift)
        self.set_schedule(schedule)

        # One fixed noise sample for the whole stream, so a still scene stays still.
        generator = torch.Generator(device).manual_seed(seed)
        self.noise = torch.randn(1, num_tokens, 64, generator=generator, device=device, dtype=torch.float32)

        # Lean prefill and target-only steps, compiled once for the stream's lifetime.
        self.denoiser = CachedDenoiser(self.transformer, attention)
        compiled = lambda fn: torch.compile(fn, fullgraph=True, mode="reduce-overhead") if compile else fn
        self.step, prefill = compiled(self.denoiser.__call__), compiled(self.denoiser.prefill)
        # CUDA graphs reuse their output buffers, and the prefix K/V must outlive later graph runs: copy them out.
        self.prefill = lambda *cond: [(k.clone(), v.clone()) for k, v in prefill(*cond)]

    def set_schedule(self, name: str):
        """Pick a schedule preset (`SCHEDULES`, or the LoRA's own). Its first node is where denoising starts; 1.0 means pure noise."""
        self.scheduler.set_timesteps(sigmas=self.presets[name], mu=self.mu if self.shift else None, device=self.device)
        self.sigmas = self.scheduler.sigmas.float()  # shifted nodes plus the terminal 0

    @torch.no_grad()
    def set_prompt(self, prompt: str, reference: torch.Tensor | None = None):
        """Encode the prompt once per prompt change.

        Without `reference` (img2img): the prefix is text only, so its K/V are prefilled here, once.
        With a reference frame (edit): the frame is the condition image. The VLM sees `reference` once, and every
        frame then re-prefills the prefix with its own latents in the condition-image slots (see `denoise`).
        """
        self.edit = reference is not None
        image = None if reference is None else [Image.fromarray(reference.cpu().numpy()).convert("RGBA")]
        embeds, _, image_pad_mask = self.pipe.encode_prompt(prompt, image=image, device=self.text_encoder_device)
        num_tokens = self.noise.shape[1]
        self.prompt_embeds = embeds.to(self.device)
        # Each image slot of the text sequence stands for 2x2 latent tokens; append the target's slots.
        self.img_mask = torch.cat([image_pad_mask.to(self.device), image_pad_mask.new_ones(1, num_tokens // 4).to(self.device)], dim=1)
        self.img_shapes = [[(1, *self.latent_hw)] * (2 if self.edit else 1)]  # condition image first, target last
        assert int(self.img_mask.sum()) * 4 == num_tokens * (2 if self.edit else 1), "condition image size mismatch"
        self.denoiser.set_layout(self.prompt_embeds, self.img_mask, self.img_shapes)
        if not self.edit:
            self.denoiser.set_kv(self.prefill())

    @torch.no_grad()
    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        """(H, W, 3) uint8 on device -> packed latents (1, h*w, 64)."""
        rgb = frame.permute(2, 0, 1)[None].to(self.dtype) / 255
        rgba = torch.cat([rgb, torch.ones_like(rgb[:, :1])], dim=1)  # the VAE is RGBA; camera frames are opaque
        latents = self.vae.encode(rgba)
        return latents.flatten(2).transpose(1, 2).float()

    @torch.no_grad()
    def denoise(self, latents: torch.Tensor) -> torch.Tensor:
        """Noise the frame's latents to the start node, then Euler-integrate the velocity to 0.

        In edit mode the clean frame latents are also the condition image, so the prefix is prefilled per frame.
        """
        if self.edit:
            self.denoiser.set_kv(self.prefill(latents.to(self.dtype)))
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
    """StreamDiffusion's Stochastic Similarity Filter.

    Skips a frame with probability max(0, (sim - threshold) / (1 - threshold)), where `sim` is the cosine similarity
    to the last frame that was actually processed. A static camera then costs almost nothing.
    """

    def __init__(self, threshold: float = 0.98, size: int = 64):
        self.threshold, self.size, self.reference = threshold, size, None

    def should_skip(self, frame: torch.Tensor) -> bool:
        small = F.interpolate(frame.permute(2, 0, 1)[None].float(), size=self.size, mode="area").flatten()
        if self.reference is not None:
            sim = F.cosine_similarity(small, self.reference, dim=0).item()
            if not small.any() and not self.reference.any():  # two black frames (cosine of zero vectors is 0)
                sim = 1.0
            if torch.rand(1).item() < (sim - self.threshold) / (1 - self.threshold):
                return True
        self.reference = small
        return False
