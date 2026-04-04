"""
ef2_segmentation.py  –  EF2 smart-mask segmentation module

Only the SAM inference call uses a third-party library (segment-anything).
All surrounding logic – mask merging, mask removal, overlay rendering –
is implemented in pure Python / pure numpy.

Install:
    pip install segment-anything
    # Download checkpoint (375 MB):
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

Place sam_vit_b_01ec64.pth in the project root (same folder as this file),
or set the SAM_CHECKPOINT environment variable to its absolute path.
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# Checkpoint path – can be overridden via env var
# ---------------------------------------------------------------------------
_DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "sam_vit_b_01ec64.pth")
SAM_CHECKPOINT = os.environ.get("SAM_CHECKPOINT", _DEFAULT_CHECKPOINT)
SAM_MODEL_TYPE = "vit_b"


# ---------------------------------------------------------------------------
# SAM wrapper  (only function that touches a third-party library)
# ---------------------------------------------------------------------------

def load_sam_predictor(image_bgr):
    """
    Load the SAM model and initialise the predictor on image_bgr.

    Parameters
    ----------
    image_bgr : numpy uint8 array (H, W, 3)  –  BGR channel order

    Returns
    -------
    SamPredictor ready for predict() calls.

    Raises
    ------
    FileNotFoundError  – if the SAM checkpoint is not present.
    ImportError        – if segment-anything is not installed.
    """
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError:
        raise ImportError(
            "segment-anything is not installed.\n"
            "Install with:  pip install segment-anything"
        )

    if not os.path.exists(SAM_CHECKPOINT):
        raise FileNotFoundError(
            f"SAM checkpoint not found at:\n  {SAM_CHECKPOINT}\n\n"
            "Download it with:\n"
            "  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth\n"
            "Or set the SAM_CHECKPOINT environment variable to its path."
        )

    print(f"[EF2] Loading SAM ({SAM_MODEL_TYPE}) from {SAM_CHECKPOINT} ...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    predictor = SamPredictor(sam)

    # SAM expects RGB – convert from OpenCV BGR
    image_rgb = image_bgr[:, :, ::-1].copy()
    predictor.set_image(image_rgb)
    print("[EF2] SAM ready. Click on the image to segment regions.")
    return predictor


def predict_mask_at_point(predictor, x, y, image_h, image_w):
    """
    Run a SAM foreground-point prompt at pixel (x, y).

    Parameters
    ----------
    predictor : SamPredictor  (from load_sam_predictor)
    x, y      : int  – pixel column and row of the click
    image_h   : int  – image height  (for bounds-checking only)
    image_w   : int  – image width   (for bounds-checking only)

    Returns
    -------
    numpy bool array (image_h, image_w) – True where the segment lies.
    """
    # Clamp click to image bounds (pure arithmetic)
    x = max(0, min(x, image_w - 1))
    y = max(0, min(y, image_h - 1))

    point_coords = np.array([[x, y]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)   # 1 = foreground

    # Third-party call: SAM inference
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,   # returns 3 candidate masks
    )

    # Pure Python: pick the mask with the highest confidence score
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return masks[best_idx]   # bool (H, W)


# ---------------------------------------------------------------------------
# Pure-Python / pure-numpy helpers (NO third-party libraries below)
# ---------------------------------------------------------------------------

def merge_masks_pure(mask_list, image_h, image_w):
    """
    OR-combine all bool masks in mask_list into a single uint8 mask.
    Values: 255 = masked region, 0 = background.

    Pure numpy – no third-party dependency.
    """
    if not mask_list:
        return np.array([[0] * image_w for _ in range(image_h)], dtype=np.uint8)

    combined = mask_list[0].copy()
    for m in mask_list[1:]:
        combined = combined | m   # element-wise OR

    return (combined * 255).astype(np.uint8)


def remove_mask_at_point_pure(mask_list, x, y):
    """
    Remove every mask in mask_list that covers pixel (x, y).

    Pure Python list comprehension – no third-party dependency.

    Parameters
    ----------
    mask_list : list of numpy bool arrays (H, W)
    x, y      : int – column and row of the right-click point

    Returns
    -------
    Filtered list (masks that do NOT cover (x, y) are kept).
    """
    return [m for m in mask_list if not m[y, x]]


def render_overlay_pure(bg_array_rgb, mask_uint8, alpha=0.55):
    """
    Blend a red overlay onto bg_array_rgb wherever mask_uint8 > 127.
    All arithmetic is pure numpy – no third-party dependency.

    Parameters
    ----------
    bg_array_rgb : numpy uint8 (H, W, 3)  –  RGB background image
    mask_uint8   : numpy uint8 (H, W)     –  0 or 255
    alpha        : float  – strength of red overlay (0 = invisible, 1 = solid red)

    Returns
    -------
    numpy uint8 (H, W, 3)  –  composited RGB image
    """
    overlay = bg_array_rgb.astype(np.float32).copy()
    where = mask_uint8 > 127

    # Red channel: blend toward 255
    overlay[where, 0] = overlay[where, 0] * (1.0 - alpha) + 255.0 * alpha
    # Green channel: blend toward 0
    overlay[where, 1] = overlay[where, 1] * (1.0 - alpha)
    # Blue channel:  blend toward 0
    overlay[where, 2] = overlay[where, 2] * (1.0 - alpha)

    # Pure-math clip to [0, 255] (no np.clip)
    overlay = overlay * (overlay >= 0) + 0.0 * (overlay < 0)
    overlay = overlay * (overlay <= 255) + 255.0 * (overlay > 255)

    return overlay.astype(np.uint8)
