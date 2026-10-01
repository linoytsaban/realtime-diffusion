"""CPU-only checks of the gaze focus (D18). Run: CUDA_VISIBLE_DEVICES= python tests/test_focus.py"""

import torch

from qwen_stream.focus import focus_mask
from qwen_stream.stream import StreamQwen

HW = (32, 32)


def test_focus_mask():
    m = focus_mask(0.5, 0.5, radius=0.25, latent_hw=HW).reshape(HW)
    assert m.shape == HW
    assert m[16, 16] == 1 and m[0, 0] == 0, "1 at the gaze point, 0 far away"
    ramp = m[16, :]
    assert ((ramp > 0) & (ramp < 1)).any(), "the edge is feathered"
    corner = focus_mask(0.0, 0.0, radius=0.25, latent_hw=HW).reshape(HW)
    assert corner[0, 0] == 1 and corner[31, 31] == 0, "u, v are image fractions from the top-left"


def stub_stream(focus):
    """A StreamQwen whose model is a fixed random velocity field; enough to test the sampling loop."""
    s = StreamQwen.__new__(StreamQwen)
    s.edit, s.dtype, s.device = False, torch.float32, "cpu"
    s.sigmas = torch.tensor([1.0, 0.84, 0.36, 0.0])
    s.noise = torch.randn(1, HW[0] * HW[1], 64, generator=torch.Generator().manual_seed(0))
    velocity = torch.randn(1, HW[0] * HW[1], 64, generator=torch.Generator().manual_seed(1))
    s.step = lambda x, sigma: velocity + 0.1 * x
    s.set_focus(focus)
    return s


def test_restore_outside():
    latents = torch.randn(1, HW[0] * HW[1], 64, generator=torch.Generator().manual_seed(2))
    free = stub_stream(None).denoise(latents)
    assert torch.equal(stub_stream(torch.zeros(1, HW[0] * HW[1], 1)).denoise(latents), latents), "mask 0 = the frame"
    assert torch.allclose(stub_stream(torch.ones(1, HW[0] * HW[1], 1)).denoise(latents), free), "mask 1 = no focus"
    mask = focus_mask(0.3, 0.6, radius=0.2, latent_hw=HW)
    out = stub_stream(mask).denoise(latents)
    outside, inside = mask[0, :, 0] == 0, mask[0, :, 0] == 1
    assert torch.equal(out[:, outside], latents[:, outside]), "outside the focus the frame is untouched"
    assert not torch.allclose(out[:, inside], latents[:, inside]), "inside the focus it is edited"


if __name__ == "__main__":
    test_focus_mask()
    test_restore_outside()
    print("focus tests passed")
