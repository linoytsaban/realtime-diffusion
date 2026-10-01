"""StreamDiffusion-style img2img streaming for Qwen-Image 2.1. Decision numbers (D1...) refer to DESIGN.md."""

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
TURBO_LORA = ("Viggle/Qwen-Image-2.1-viggle-turbo", "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors")
TURBO_SIGMAS = [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]  # raw nodes of the turbo student (D2)
# Raw-node schedules (D2). img2img enters mid-way (D4); edit always starts from pure noise (D13).
SCHEDULES = {
    "img2img-subtle": [0.5, 0.25],
    "img2img-medium": [0.75, 0.5, 0.25],
    "img2img-strong": [0.875, 0.75, 0.5, 0.25],
    "edit-3": [1.0, 0.75, 0.25],  # skips the .5 node; close to edit-4 in the step sweep (D13)
    "edit-4": [1.0, 0.75, 0.5, 0.25],  # the student's 4-step training nodes
    "edit-5": [1.0, 0.875, 0.75, 0.5, 0.25],
    "edit-6": TURBO_SIGMAS,  # the card's recommended schedule
}


class StreamQwen:
    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        schedule: str = "img2img-medium",
        seed: int = 0,
        device: str = "cuda",
        text_encoder_device: str = "cpu",
        fp8: str | None = None,
        compile: bool = False,
        attention: str = "cudnn",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device, self.text_encoder_device, self.dtype = device, text_encoder_device, dtype
        self.latent_hw = (height // 16, width // 16)
        num_tokens = self.latent_hw[0] * self.latent_hw[1]

        # D1: the full VAE is never loaded, TAEQI2.1 does both ends.
        self.pipe = QwenImage21Pipeline.from_pretrained(BASE_MODEL, vae=None, torch_dtype=dtype)
        # D9: only the DiT lives on the GPU by default; the 8B text encoder runs once per prompt wherever it is put.
        self.pipe.text_encoder.to(text_encoder_device)
        self.pipe.ref_token_list = ["Picture "]  # D13: the edit template otherwise picks its image label at random
        self.transformer = self.pipe.transformer.to(device)
        # D3: the turbo LoRA is fused into the weights, trading a little precision for no per-layer LoRA matmuls.
        apply_lora(self.transformer, hf_hub_download(*TURBO_LORA))
        fuse_projections(self.transformer)  # D17; after the LoRA (it targets to_q/to_k/...), before FP8
        if fp8:  # D10: FP8 weights and activations (Ada, Hopper, Blackwell), scaled per "row" or per "tensor"
            from torchao.quantization import PerRow, PerTensor, float8_dynamic_activation_float8_weight, quantize_

            granularity = {"row": PerRow(), "tensor": PerTensor()}[fp8]
            quantize_(self.transformer.transformer_blocks, float8_dynamic_activation_float8_weight(granularity=granularity))
        self.vae = TinyVAE.from_pretrained().to(device, dtype)

        # D2: turbo nodes get the pipeline's resolution-dependent time shift, computed by its own scheduler.
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config, shift_terminal=None)
        cfg = self.scheduler.config
        self.mu = calculate_shift(num_tokens, cfg.base_image_seq_len, cfg.max_image_seq_len, cfg.base_shift, cfg.max_shift)
        self.set_schedule(schedule)

        # D5: one fixed noise sample for the whole stream.
        generator = torch.Generator(device).manual_seed(seed)
        self.noise = torch.randn(1, num_tokens, 64, generator=generator, device=device, dtype=torch.float32)

        # D11/D14: lean prefill and target-only steps, compiled once for the stream's lifetime.
        self.denoiser = CachedDenoiser(self.transformer, attention)
        compiled = lambda fn: torch.compile(fn, fullgraph=True, mode="reduce-overhead") if compile else fn
        self.step, prefill = compiled(self.denoiser.__call__), compiled(self.denoiser.prefill)
        # CUDA graphs reuse their output buffers, and the prefix K/V must outlive later graph runs: copy them out.
        self.prefill = lambda *cond: [(k.clone(), v.clone()) for k, v in prefill(*cond)]

    def set_schedule(self, name: str):
        """Pick a `SCHEDULES` preset. Its first node is where denoising starts; 1.0 means pure noise."""
        self.scheduler.set_timesteps(sigmas=SCHEDULES[name], mu=self.mu, device=self.device)
        self.sigmas = self.scheduler.sigmas.float()  # shifted nodes plus the terminal 0

    @torch.no_grad()
    def set_prompt(self, prompt: str, reference: torch.Tensor | None = None):
        """Encode the prompt once per prompt change.

        Without `reference` (img2img, D6): the prefix is text only, so its K/V are prefilled here, once.
        With a reference frame (edit, D13): the frame is the condition image. The VLM sees `reference` once, and every
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
        """D2/D4: noise the frame's latents to the start node, then Euler-integrate the velocity to 0.

        In edit mode (D13) the clean frame latents are also the condition image, so the prefix is prefilled per frame.
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
