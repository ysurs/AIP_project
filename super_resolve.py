"""
super_resolve.py  –  EF3 super-resolution module

Third-party library (super-image) is ONLY used here for the SR step.
All surrounding logic (resize, format conversion) uses pure numpy/PIL.

Install:
    pip install super-image
    (PyTorch is required – super-image will prompt if missing)

The EDSR-base x4 model is auto-downloaded from HuggingFace Hub on first use.
If super-image is unavailable, falls back to high-quality PIL LANCZOS resize.
"""

import numpy as np
from PIL import Image


# Cached model so we don't reload on every call within one run
_edsr_model = None


def _load_edsr_model(scale=4):
    """Load and cache the EDSR-base super-resolution model."""
    global _edsr_model
    if _edsr_model is not None:
        return _edsr_model

    from super_image import EdsrModel
    print(f"[SR] Loading EDSR-base x{scale} model (downloads on first use)...")
    _edsr_model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=scale)
    _edsr_model.eval()
    print("[SR] Model loaded.")
    return _edsr_model


def super_resolve_image(img_bgr, target_h, target_w):
    """
    Upsample img_bgr using EDSR x4 super-resolution, then resize to (target_h, target_w).

    Parameters
    ----------
    img_bgr   : numpy uint8 array (H, W, 3)  –  BGR channel order (OpenCV convention)
    target_h  : int  –  desired output height in pixels
    target_w  : int  –  desired output width  in pixels

    Returns
    -------
    numpy uint8 array (target_h, target_w, 3)  –  BGR channel order
    """
    try:
        import torch
        from super_image import EdsrModel, ImageLoader

        model = _load_edsr_model(scale=4)

        # BGR → RGB PIL Image
        img_rgb = img_bgr[:, :, ::-1].astype(np.uint8)
        pil_img = Image.fromarray(img_rgb).convert('RGB')

        # Build input tensor via ImageLoader (handles normalisation)
        inputs = ImageLoader.load_image(pil_img)

        # Run super-resolution (no gradient needed)
        with torch.no_grad():
            preds = model(inputs)

        # Tensor (1, 3, H*4, W*4) → numpy uint8 (H*4, W*4, 3) RGB
        sr_np = (
            preds.detach().cpu()
                 .squeeze(0)           # (3, H*4, W*4)
                 .permute(1, 2, 0)     # (H*4, W*4, 3)
                 .clamp(0.0, 1.0)
                 .numpy()
        )
        sr_uint8 = (sr_np * 255.0).round().astype(np.uint8)

        # Exact resize to target dimensions (pure PIL – not another SR pass)
        sr_pil = Image.fromarray(sr_uint8).convert('RGB')
        if sr_pil.width != target_w or sr_pil.height != target_h:
            sr_pil = sr_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)

        # RGB → BGR for OpenCV-compatible output
        return np.array(sr_pil)[:, :, ::-1].copy()

    except ImportError:
        print("[SR] WARNING: 'super-image' or 'torch' not installed.")
        print("[SR] Falling back to high-quality PIL LANCZOS resize (not true SR).")
        print("[SR] Install with:  pip install super-image")
        return _lanczos_resize(img_bgr, target_h, target_w)


def _lanczos_resize(img_bgr, target_h, target_w):
    """
    Fallback: high-quality PIL LANCZOS upsampling (interpolation, not SR).
    Uses only PIL – no third-party dependency.
    """
    img_rgb = img_bgr[:, :, ::-1].astype(np.uint8)
    pil_img = Image.fromarray(img_rgb).convert('RGB')
    resized = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return np.array(resized)[:, :, ::-1].copy()
