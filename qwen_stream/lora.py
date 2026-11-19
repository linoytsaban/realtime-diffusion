"""Minimal LoRA application. Avoids peft, which is not needed to add B @ A to a handful of Linear layers."""

import json

import torch
import torch.nn.functional as F
from safetensors import safe_open


def apply_lora(model: torch.nn.Module, path: str, fuse: bool = True, prefix: str = "transformer."):
    """Apply a LoRA to `model`, in either diffusers/peft (`.lora_A/.lora_B.weight`) or kohya-style
    (`.lora.down/.lora.up.weight`) naming. Keys are `<prefix><module>.<down|A>...`.

    fuse=True:  W <- W + scale * B @ A, computed in fp32 and rounded to the weight dtype once. Zero runtime cost.
    fuse=False: a forward hook adds scale * B(A(x)). Exact, but costs two extra skinny matmuls per layer.
    """
    with safe_open(path, "pt") as f:
        metadata = json.loads(f.metadata().get("lora_adapter_metadata", "{}"))
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    rank, alpha = metadata.get(f"{prefix}r"), metadata.get(f"{prefix}lora_alpha")
    scale = alpha / rank if rank and alpha else 1.0

    # lightx2v ships `.lora.down/.lora.up`; diffusers/Pruna/Viggle ship `.lora_A/.lora_B`.
    down, up = (".lora_A.weight", ".lora_B.weight")
    if not any(k.endswith(down) for k in state_dict):
        down, up = ".lora.down.weight", ".lora.up.weight"

    for key in state_dict:
        if not key.endswith(down):
            continue
        name = key[len(prefix) : -len(down)]
        linear = model.get_submodule(name)
        A = state_dict[key].to(linear.weight.device)
        B = state_dict[key[: -len(down)] + up].to(linear.weight.device)
        if fuse:
            delta = scale * (B.float() @ A.float())
            linear.weight.data = (linear.weight.float() + delta).to(linear.weight.dtype)
        else:
            A, B = A.to(linear.weight.dtype), B.to(linear.weight.dtype)
            linear.register_forward_hook(lambda m, args, out, A=A, B=B: out + scale * F.linear(F.linear(args[0], A), B))
