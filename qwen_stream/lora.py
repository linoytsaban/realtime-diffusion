"""Minimal LoRA application (D3). Avoids peft, which is not needed to add B @ A to a handful of Linear layers."""

import json

import torch
import torch.nn.functional as F
from safetensors import safe_open


def apply_lora(model: torch.nn.Module, path: str, fuse: bool = True, prefix: str = "transformer."):
    """Apply a diffusers-format LoRA (`<prefix><module>.lora_{A,B}.weight`) to `model`.

    fuse=True:  W <- W + scale * B @ A, computed in fp32 and rounded to the weight dtype once. Zero runtime cost.
    fuse=False: a forward hook adds scale * B(A(x)). Exact, but costs two extra skinny matmuls per layer.
    """
    with safe_open(path, "pt") as f:
        metadata = json.loads(f.metadata().get("lora_adapter_metadata", "{}"))
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    rank, alpha = metadata.get(f"{prefix}r"), metadata.get(f"{prefix}lora_alpha")
    scale = alpha / rank if rank and alpha else 1.0

    for key in state_dict:
        if not key.endswith(".lora_A.weight"):
            continue
        name = key[len(prefix) : -len(".lora_A.weight")]
        linear = model.get_submodule(name)
        A = state_dict[key].to(linear.weight.device)
        B = state_dict[key.replace("lora_A", "lora_B")].to(linear.weight.device)
        if fuse:
            delta = scale * (B.float() @ A.float())
            linear.weight.data = (linear.weight.float() + delta).to(linear.weight.dtype)
        else:
            A, B = A.to(linear.weight.dtype), B.to(linear.weight.dtype)
            linear.register_forward_hook(lambda m, args, out, A=A, B=B: out + scale * F.linear(F.linear(args[0], A), B))
