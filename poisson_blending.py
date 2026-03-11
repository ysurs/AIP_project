import cv2
import numpy as np
from local_context_matching import get_local_context_mask

def blend_match(query_img, mask_bool, match_img, placement):
    """
    Blends the best matched patch into the query image using Poisson blending.
    
    query_img: Original image (BGR)
    mask_bool: Boolean or 0/1 array where hole is 1
    match_img: The matched scene image (BGR)
    placement: Tuple (scale, ty, tx) from local context matching
    """
    scale, ty, tx = placement
    
    # Process mask to find proper coordinates
    mask_u8 = (mask_bool * 255).astype(np.uint8)
    
    # 1. Recompute the context box to know where the matched patch places exactly
    context_mask = get_local_context_mask(mask_bool, radius=80)
    coords = np.argwhere(context_mask)
    if len(coords) == 0:
        return query_img
        
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1
    context_h = y_max - y_min
    context_w = x_max - x_min
    
    # 2. Scale the matched image
    if scale != 1.0:
        h, w = match_img.shape[:2]
        scaled_match = cv2.resize(match_img, (int(w * scale), int(h * scale)))
    else:
        scaled_match = match_img
        
    # 3. Extract the aligned patch from the matched image
    patch = scaled_match[ty:ty+context_h, tx:tx+context_w]
    
    # Create an empty source image of the same size as the query target
    src_full = np.zeros_like(query_img)
    
    # Safety check for boundaries and sizes
    ph, pw = patch.shape[:2]
    src_full[y_min:y_min+ph, x_min:x_min+pw] = patch
    
    # 4. Find the center of the original hole to guide the seamless clone
    hole_coords = np.argwhere(mask_bool > 0)
    if len(hole_coords) == 0:
        return query_img
        
    center_y = int(np.mean(hole_coords[:, 0]))
    center_x = int(np.mean(hole_coords[:, 1]))
    center = (center_x, center_y) # OpenCV requires (x, y) order
    
    # 5. Perform Poisson Blending
    # cv2.NORMAL_CLONE preserves textures but adapts colors/lighting
    blended = cv2.seamlessClone(src_full, query_img, mask_u8, center, cv2.NORMAL_CLONE)
    
    return blended