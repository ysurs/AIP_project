"""
super_resolve.py  –  EF3 super-resolution module

Uses Real-ESRGAN x4plus exclusively.
Install: pip install realesrgan basicsr facexlib gfpgan
"""

import numpy as np
from PIL import Image

# Cached model – loaded once per process
_realesrgan_upsampler = None


# ---------------------------------------------------------------------------
# Real-ESRGAN
# ---------------------------------------------------------------------------

def _load_realesrgan():
    global _realesrgan_upsampler
    if _realesrgan_upsampler is not None:
        return _realesrgan_upsampler

    # Compatibility shim: torchvision >= 0.16 removed functional_tensor,
    # but basicsr still imports it. Patch sys.modules before importing basicsr.
    import sys, types
    if 'torchvision.transforms.functional_tensor' not in sys.modules:
        import torchvision.transforms.functional as _F
        _shim = types.ModuleType('torchvision.transforms.functional_tensor')
        for _attr in dir(_F):
            setattr(_shim, _attr, getattr(_F, _attr))
        sys.modules['torchvision.transforms.functional_tensor'] = _shim

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32,
        scale=4
    )
    _realesrgan_upsampler = RealESRGANer(
        scale=4,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        model=model,
        tile=256,       # tile-based inference to avoid OOM on large images
        tile_pad=10,
        pre_pad=0,
        half=False,     # float32 for MPS/CPU compatibility
    )
    print("[SR] Real-ESRGAN x4plus model loaded.")
    return _realesrgan_upsampler


def _sr_realesrgan(img_bgr, target_h, target_w):
    upsampler = _load_realesrgan()
    # enhance() expects BGR uint8, returns BGR uint8
    output, _ = upsampler.enhance(img_bgr, outscale=4)
    # Resize to exact target if needed
    if output.shape[1] != target_w or output.shape[0] != target_h:
        pil = Image.fromarray(output[:, :, ::-1])
        pil = pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
        output = np.array(pil)[:, :, ::-1].copy()
    return output


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def super_resolve_image(img_bgr, target_h, target_w):
    """
    Upsample img_bgr using Real-ESRGAN x4plus.

    Parameters
    ----------
    img_bgr  : numpy uint8 array (H, W, 3) – BGR channel order
    target_h : int – desired output height
    target_w : int – desired output width

    Returns
    -------
    numpy uint8 array (target_h, target_w, 3) – BGR channel order
    """
    return _sr_realesrgan(img_bgr, target_h, target_w)
