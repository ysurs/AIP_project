import numpy as np
import cv2
from scipy.ndimage import median_filter
import math

def get_local_context_mask(mask, radius=80):
    """
    Find all pixels within an 80-pixel radius of the hole's boundary.
    mask: boolean or 0/1 array where hole is 1/True.
    Returns a boolean mask of the local context.
    """
    # Find boundary of the hole
    # Dilation minus original gives the boundary, or we can just dilate the hole and subtract the hole
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    
    # Ensure mask is uint8
    mask_u8 = mask.astype(np.uint8)
    
    # Dilate the hole by the radius
    dilated_hole = cv2.dilate(mask_u8, kernel)
    
    # The local context is the dilated area minus the original hole
    context_mask = (dilated_hole > 0) & (mask_u8 == 0)
    
    return context_mask

def compute_gradient_magnitude(img_gray):
    """
    Computes the gradient magnitude of a grayscale image.
    """
    # Typical Sobel gradients
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return mag

def texture_descriptor(img_gray):
    """
    A simple texture descriptor is computed as a 5x5 median filter 
    of image gradient magnitude at each pixel.
    """
    mag = compute_gradient_magnitude(img_gray)
    median_mag = median_filter(mag, size=5)
    return median_mag

def calculate_translational_offset(source_center, target_center):
    """
    Magnitude of translational offset.
    """
    dy = target_center[0] - source_center[0]
    dx = target_center[1] - source_center[1]
    return math.sqrt(dx**2 + dy**2)

def local_context_matching(query_img_bgr, mask, match_img_bgr_list):
    """
    Implements local context matching from Section 4 of the scene paper.
    
    query_img_bgr: The original query image (with the hole).
    mask: The mask where hole is 1 and valid pixels are 0.
    match_img_bgr_list: List of top matching scene images.
    """
    
    context_mask = get_local_context_mask(mask, radius=80)
    context_pixels_count = np.sum(context_mask)
    if context_pixels_count == 0:
        return []
    
    coords = np.argwhere(context_mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1
    
    context_h = y_max - y_min
    context_w = x_max - x_min
    
    # FIX: Use true float representation for accurate L*a*b* scales
    query_img_float = query_img_bgr.astype(np.float32) / 255.0
    query_lab = cv2.cvtColor(query_img_float, cv2.COLOR_BGR2Lab)
    
    # Extract query context crop
    query_context_lab = query_lab[y_min:y_max, x_min:x_max]
    query_context_mask = context_mask[y_min:y_max, x_min:x_max]
    
    # Query structure/texture
    query_gray = cv2.cvtColor(query_img_bgr, cv2.COLOR_BGR2GRAY)
    query_tex = texture_descriptor(query_gray)
    query_context_tex = query_tex[y_min:y_max, x_min:x_max]
    
    center_y_q = y_min + context_h / 2.0
    center_x_q = x_min + context_w / 2.0
    
    scales = [0.81, 0.90, 1.0]
    
    best_results = []
    
    for match_idx, match_img in enumerate(match_img_bgr_list):
        # FIX: standard L*a*b* range
        match_float = match_img.astype(np.float32) / 255.0
        match_lab = cv2.cvtColor(match_float, cv2.COLOR_BGR2Lab)
        match_gray = cv2.cvtColor(match_img, cv2.COLOR_BGR2GRAY)
        
        best_score = float('inf')
        best_placement = None #(scale, ty, tx)
        best_texture_ssd = float('inf')
        
        for scale in scales:
            if scale != 1.0:
                h, w = match_img.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                if new_h < context_h or new_w < context_w:
                    continue
                scaled_lab = cv2.resize(match_lab, (new_w, new_h))
                scaled_gray = cv2.resize(match_gray, (new_w, new_h))
            else:
                scaled_lab = match_lab
                scaled_gray = match_gray
                
            sh, sw = scaled_lab.shape[:2]
            
            if sh < context_h or sw < context_w:
                continue
                
            scaled_tex = texture_descriptor(scaled_gray)
            
            # Increase step size to 4 (or 8) to drastically speed up the search
            step_size = 4
            for ty in range(0, sh - context_h + 1, step_size):
                for tx in range(0, sw - context_w + 1, step_size):
                    # Region in scaled match
                    match_region = scaled_lab[ty:ty+context_h, tx:tx+context_w]
                    
                    diff_l = match_region[:,:,0] - query_context_lab[:,:,0]
                    diff_a = match_region[:,:,1] - query_context_lab[:,:,1]
                    diff_b = match_region[:,:,2] - query_context_lab[:,:,2]
                    
                    ssd = np.sum((diff_l**2 + diff_a**2 + diff_b**2)[query_context_mask])
                    
                    # FIX: Correct scale-adjusted translational offset based on bounding box centers
                    center_y_m = ty + context_h / 2.0
                    center_x_m = tx + context_w / 2.0
                    orig_center_y_m = center_y_m / scale
                    orig_center_x_m = center_x_m / scale
                    
                    offset = calculate_translational_offset(
                        (center_y_q, center_x_q), 
                        (orig_center_y_m, orig_center_x_m)
                    )
                    
                    if offset < 1.0:
                        offset = 1.0
                        
                    weighted_ssd = ssd * offset
                    
                    if weighted_ssd < best_score:
                        best_score = weighted_ssd
                        best_placement = (scale, ty, tx)
                        
                        # Texture SSD
                        match_tex_region = scaled_tex[ty:ty+context_h, tx:tx+context_w]
                        tex_diff = match_tex_region - query_context_tex
                        tex_ssd = np.sum((tex_diff ** 2)[query_context_mask])
                        best_texture_ssd = tex_ssd
                        
        best_results.append({
            'match_idx': match_idx,
            'best_score': best_score,
            'best_placement': best_placement,
            'texture_ssd': best_texture_ssd
        })
        
    return best_results
