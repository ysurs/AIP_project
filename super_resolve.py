"""
super_resolve.py  –  EF3 super-resolution module

Primary:  Real-ESRGAN x4plus  (pip install realesrgan basicsr)
Fallback: EDSR-base x4        (pip install super-image)
Fallback: PIL LANCZOS          (always available)

Install Real-ESRGAN:
    pip install realesrgan basicsr facexlib gfpgan
"""

import numpy as np
from PIL import Image

# Cached models – loaded once per process
_realesrgan_upsampler = None
_edsr_model = None


# ---------------------------------------------------------------------------
# Real-ESRGAN (primary)
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
    # Resize to exact target if needed (aspect-ratio-preserving call from pipeline
    # means this is usually a no-op)
    if output.shape[1] != target_w or output.shape[0] != target_h:
        pil = Image.fromarray(output[:, :, ::-1])
        pil = pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
        output = np.array(pil)[:, :, ::-1].copy()
    return output


# ---------------------------------------------------------------------------
# EDSR-base (first fallback)
# ---------------------------------------------------------------------------

def _load_edsr_model(scale=4):
    global _edsr_model
    if _edsr_model is not None:
        return _edsr_model

    from super_image import EdsrModel
    print(f"[SR] Loading EDSR-base x{scale} model (downloads on first use)...")
    _edsr_model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=scale)
    _edsr_model.eval()
    print("[SR] EDSR-base model loaded.")
    return _edsr_model


def _sr_edsr(img_bgr, target_h, target_w):
    import torch
    from super_image import EdsrModel, ImageLoader

    model = _load_edsr_model(scale=4)
    img_rgb = img_bgr[:, :, ::-1].astype(np.uint8)
    pil_img = Image.fromarray(img_rgb).convert('RGB')
    inputs = ImageLoader.load_image(pil_img)

    with torch.no_grad():
        preds = model(inputs)

    sr_np = (
        preds.detach().cpu()
             .squeeze(0)
             .permute(1, 2, 0)
             .clamp(0.0, 1.0)
             .numpy()
    )
    sr_uint8 = (sr_np * 255.0).round().astype(np.uint8)
    sr_pil = Image.fromarray(sr_uint8).convert('RGB')
    if sr_pil.width != target_w or sr_pil.height != target_h:
        sr_pil = sr_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return np.array(sr_pil)[:, :, ::-1].copy()


# ---------------------------------------------------------------------------
# PIL LANCZOS (final fallback)
# ---------------------------------------------------------------------------

def _lanczos_resize(img_bgr, target_h, target_w):
    img_rgb = img_bgr[:, :, ::-1].astype(np.uint8)
    pil_img = Image.fromarray(img_rgb).convert('RGB')
    resized = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return np.array(resized)[:, :, ::-1].copy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def super_resolve_image(img_bgr, target_h, target_w):
    """
    Upsample img_bgr using the best available SR model.

    Priority: Real-ESRGAN x4plus → EDSR-base x4 → PIL LANCZOS

    Parameters
    ----------
    img_bgr  : numpy uint8 array (H, W, 3) – BGR channel order
    target_h : int – desired output height
    target_w : int – desired output width

    Returns
    -------
    numpy uint8 array (target_h, target_w, 3) – BGR channel order
    """
    # Try Real-ESRGAN first
    try:
        return _sr_realesrgan(img_bgr, target_h, target_w)
    except ImportError as e:
        print(f"[SR] 'realesrgan' import error: {e}. Trying EDSR-base...")
    except Exception as e:
        print(f"[SR] Real-ESRGAN failed ({e}). Trying EDSR-base...")

    # Try EDSR
    try:
        return _sr_edsr(img_bgr, target_h, target_w)
    except ImportError:
        print("[SR] 'super-image' not installed. Falling back to PIL LANCZOS.")
    except Exception as e:
        print(f"[SR] EDSR failed ({e}). Falling back to PIL LANCZOS.")

    # Final fallback
    print("[SR] WARNING: Using PIL LANCZOS (not true SR).")
    return _lanczos_resize(img_bgr, target_h, target_w)
