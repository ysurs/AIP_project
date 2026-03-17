import numpy as np
import cv2
from scipy.ndimage import median_filter
import math

# def get_local_context_mask(mask, radius=80):
#     """
#     Find all pixels within an 80-pixel radius of the hole's boundary.
#     mask: boolean or 0/1 array where hole is 1/True.
#     Returns a boolean mask of the local context.
#     """
#     # Find boundary of the hole
#     # Dilation minus original gives the boundary, or we can just dilate the hole and subtract the hole
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    
#     # Ensure mask is uint8
#     mask_u8 = mask.astype(np.uint8)
    
#     # Dilate the hole by the radius
#     dilated_hole = cv2.dilate(mask_u8, kernel)
    
#     # The local context is the dilated area minus the original hole
#     context_mask = (dilated_hole > 0) & (mask_u8 == 0)
    
#     return context_mask

# def compute_gradient_magnitude(img_gray):
#     """
#     Computes the gradient magnitude of a grayscale image.
#     """
#     # Typical Sobel gradients
#     gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
#     gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
#     mag = cv2.magnitude(gx, gy)
#     return mag

# def texture_descriptor(img_gray):
#     """
#     A simple texture descriptor is computed as a 5x5 median filter 
#     of image gradient magnitude at each pixel.
#     """
#     mag = compute_gradient_magnitude(img_gray)
#     median_mag = median_filter(mag, size=5)
#     return median_mag

# def calculate_translational_offset(source_center, target_center):
#     """
#     Magnitude of translational offset.
#     """
#     dy = target_center[0] - source_center[0]
#     dx = target_center[1] - source_center[1]
#     return math.sqrt(dx**2 + dy**2)

# def local_context_matching(query_img_path, mask_path, match_img_bgr_list):
#     """
#     Implements local context matching from Section 4 of the scene paper.
    
#     query_img_bgr: The original query image (with the hole).
#     mask: The mask where hole is 1 and valid pixels are 0.
#     match_img_bgr_list: List of top matching scene images.
#     """
#     # Load images using cv2
#     query_img_bgr = cv2.imread(query_img_path)
#     if query_img_bgr is None:
#         raise ValueError(f"Unable to load query image from {query_img_path}")
        
#     mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     if mask_img is None:
#         raise ValueError(f"Unable to load mask from {mask_path}")
    
#     # Assuming hole is white (>127) and valid area is black
#     mask = mask_img > 127
        
#     context_mask = get_local_context_mask(mask, radius=80)
    
#     context_pixels_count = np.sum(context_mask)
#     if context_pixels_count == 0:
#         return []
    
#     coords = np.argwhere(context_mask)
#     y_min, x_min = coords.min(axis=0)
#     y_max, x_max = coords.max(axis=0) + 1
    
#     context_h = y_max - y_min
#     context_w = x_max - x_min
    
#     # FIX: Use true float representation for accurate L*a*b* scales
#     query_img_float = query_img_bgr.astype(np.float32) / 255.0
#     query_lab = cv2.cvtColor(query_img_float, cv2.COLOR_BGR2Lab)
    
#     # Extract query context crop
#     query_context_lab = query_lab[y_min:y_max, x_min:x_max]
#     query_context_mask = context_mask[y_min:y_max, x_min:x_max]
    
#     # Query structure/texture
#     query_gray = cv2.cvtColor(query_img_bgr, cv2.COLOR_BGR2GRAY)
#     query_tex = texture_descriptor(query_gray)
#     query_context_tex = query_tex[y_min:y_max, x_min:x_max]
    
#     center_y_q = y_min + context_h / 2.0
#     center_x_q = x_min + context_w / 2.0
    
#     scales = [0.81, 0.90, 1.0]
    
#     best_results = []
    
#     for match_idx, match_img in enumerate(match_img_bgr_list):
#         # FIX: standard L*a*b* range
#         # print(match_img)
#         # exit()
#         match_float = match_img.astype(np.float32) / 255.0
#         match_lab = cv2.cvtColor(match_float, cv2.COLOR_BGR2Lab)
#         match_gray = cv2.cvtColor(match_img, cv2.COLOR_BGR2GRAY)
        
#         best_score = float('inf')
#         best_placement = None #(scale, ty, tx)
#         best_texture_ssd = float('inf')
        
#         for scale in scales:
#             if scale != 1.0:
#                 h, w = match_img.shape[:2]
#                 new_h, new_w = int(h * scale), int(w * scale)
#                 if new_h < context_h or new_w < context_w:
#                     continue
#                 scaled_lab = cv2.resize(match_lab, (new_w, new_h))
#                 scaled_gray = cv2.resize(match_gray, (new_w, new_h))
#             else:
#                 scaled_lab = match_lab
#                 scaled_gray = match_gray
                
#             sh, sw = scaled_lab.shape[:2]
            
#             if sh < context_h or sw < context_w:
#                 continue
                
#             scaled_tex = texture_descriptor(scaled_gray)
            
#             # Increase step size to 4 (or 8) to drastically speed up the search
#             step_size = 4
#             for ty in range(0, sh - context_h + 1, step_size):
#                 for tx in range(0, sw - context_w + 1, step_size):
#                     # Region in scaled match
#                     match_region = scaled_lab[ty:ty+context_h, tx:tx+context_w]
                    
#                     diff_l = match_region[:,:,0] - query_context_lab[:,:,0]
#                     diff_a = match_region[:,:,1] - query_context_lab[:,:,1]
#                     diff_b = match_region[:,:,2] - query_context_lab[:,:,2]
                    
#                     ssd = np.sum((diff_l**2 + diff_a**2 + diff_b**2)[query_context_mask])
                    
#                     # FIX: Correct scale-adjusted translational offset based on bounding box centers
#                     center_y_m = ty + context_h / 2.0
#                     center_x_m = tx + context_w / 2.0
#                     orig_center_y_m = center_y_m / scale
#                     orig_center_x_m = center_x_m / scale
                    
#                     offset = calculate_translational_offset(
#                         (center_y_q, center_x_q), 
#                         (orig_center_y_m, orig_center_x_m)
#                     )
                    
#                     if offset < 1.0:
#                         offset = 1.0
                        
#                     weighted_ssd = ssd * offset
                    
#                     if weighted_ssd < best_score:
#                         best_score = weighted_ssd
#                         best_placement = (scale, ty, tx)
                        
#                         # Texture SSD
#                         match_tex_region = scaled_tex[ty:ty+context_h, tx:tx+context_w]
#                         tex_diff = match_tex_region - query_context_tex
#                         tex_ssd = np.sum((tex_diff ** 2)[query_context_mask])
#                         best_texture_ssd = tex_ssd
                        
#         best_results.append({
#             'match_idx': match_idx,
#             'best_score': best_score,
#             'best_placement': best_placement,
#             'texture_ssd': best_texture_ssd
#         })
        
#     return best_results

# import cv2

# import numpy as np

# import math

# from scipy.ndimage import median_filter



# def get_local_context_mask(mask, radius=80):

#     """Generates the 'donut' region around the hole for matching."""

#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

#     mask_u8 = mask.astype(np.uint8)

#     dilated_hole = cv2.dilate(mask_u8, kernel)

#     # Context is the area outside the hole but inside the dilated boundary

#     return (dilated_hole > 0) & (mask_u8 == 0)



# def texture_descriptor(img_gray):

#     """Computes texture magnitude via Sobel gradients and median filtering."""

#     gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)

#     gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

#     mag = cv2.magnitude(gx, gy)

#     # Median filter captures 'textural vibe' rather than specific edges

#     return median_filter(mag, size=5)



# def local_context_matching(query_img_path, mask_path, match_img_list, weight_tex=0.5):

#     """

#     Finds best local alignments for a list of candidate images.

#     Aligns with Hays & Efros (Section 4).

#     """
#     #Load images using cv2
#     query_img_bgr = cv2.imread(query_img_path)
#     if query_img_bgr is None:
#         raise ValueError(f"Unable to load query image from {query_img_path}")
        
#     mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     if mask_img is None:
#         raise ValueError(f"Unable to load mask from {mask_path}")
    
#     # Assuming hole is white (>127) and valid area is black
#     mask = mask_img > 127

#     # 1. Pre-calculate Query Context

#     context_mask = get_local_context_mask(mask, radius=80)

#     num_pixels = np.sum(context_mask)

#     if num_pixels == 0: return []



#     # Bounding box of the context donut for efficient cropping

#     coords = np.argwhere(context_mask)

#     y1, x1 = coords.min(axis=0)

#     y2, x2 = coords.max(axis=0) + 1

#     ch, cw = y2 - y1, x2 - x1

    

#     # Extract Query Descriptors

#     q_lab = cv2.cvtColor(query_img_bgr.astype(np.float32)/255.0, cv2.COLOR_BGR2Lab)

#     q_gray = cv2.cvtColor(query_img_bgr, cv2.COLOR_BGR2GRAY)

#     q_tex = texture_descriptor(q_gray)

    

#     q_ctx_lab = q_lab[y1:y2, x1:x2]

#     q_ctx_tex = q_tex[y1:y2, x1:x2]

#     q_ctx_mask = context_mask[y1:y2, x1:x2]

    

#     center_q = (y1 + ch/2.0, x1 + cw/2.0)

#     scales = [0.81, 0.90, 1.0]

#     results = []



#     for idx, match_img in enumerate(match_img_list):

#         m_lab_full = cv2.cvtColor(match_img.astype(np.float32)/255.0, cv2.COLOR_BGR2Lab)

#         m_gray_full = cv2.cvtColor(match_img, cv2.COLOR_BGR2GRAY)

        

#         best_match_for_img = {'score': float('inf'), 'placement': None}



#         for s in scales:

#             sh, sw = int(match_img.shape[0] * s), int(match_img.shape[1] * s)

#             if sh < ch or sw < cw: continue

            

#             s_lab = cv2.resize(m_lab_full, (sw, sh))

#             s_gray = cv2.resize(m_gray_full, (sw, sh))

#             s_tex = texture_descriptor(s_gray)



#             # Search donor image

#             for ty in range(0, sh - ch + 1, 4): # Step 4 for performance

#                 for tx in range(0, sw - cw + 1, 4):

#                     # Color SSD (L*a*b*)

#                     reg_lab = s_lab[ty:ty+ch, tx:tx+cw]

#                     color_ssd = np.sum(np.square(reg_lab - q_ctx_lab)[q_ctx_mask])

                    

#                     # Texture SSD

#                     reg_tex = s_tex[ty:ty+ch, tx:tx+cw]

#                     tex_ssd = np.sum(np.square(reg_tex - q_ctx_tex)[q_ctx_mask])

                    

#                     # Spatial Penalty (Distance from original coordinates)

#                     center_m = ((ty + ch/2.0)/s, (tx + cw/2.0)/s)

#                     dist = math.sqrt((center_q[0]-center_m[0])**2 + (center_q[1]-center_m[1])**2)

#                     penalty = 1.0 + (dist / 100.0) # Scaled linear penalty



#                     # Normalized Combined Cost

#                     score = ((color_ssd + weight_tex * tex_ssd) / num_pixels) * penalty

                    

#                     if score < best_match_for_img['score']:

#                         best_match_for_img = {

#                             'score': score,

#                             'placement': (s, ty, tx), # Needed for Graph Cut

#                             'color_ssd': color_ssd / num_pixels,

#                             'texture_ssd': tex_ssd / num_pixels

#                         }



#         results.append({

#             'match_idx': idx,

#             'best_score': best_match_for_img['score'],

#             'placement': best_match_for_img['placement'],

#             'color_error': best_match_for_img.get('color_ssd'),

#             'texture_error': best_match_for_img.get('texture_ssd')

#         })

        

#     return sorted(results, key=lambda x: x['best_score'])

import cv2
import numpy as np

import cv2
import numpy as np

def get_texture_map(img_gray):
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_blurred = cv2.medianBlur(mag, 5)
    
    # Fix 1: Normalize to [0, 255] to align with the scale of Lab space features
    cv2.normalize(mag_blurred, mag_blurred, 0, 255, cv2.NORM_MINMAX)
    return mag_blurred

def match_context_optimized(query_img_path, mask_path, match_img_list, weight_tex=10):
    q_bgr = cv2.imread(query_img_path)
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_bool = mask_img > 127
    
    # Get query dimensions for relative spatial penalty
    q_h, q_w = q_bgr.shape[:2]
    
    # 1. Prepare Query Features
    q_lab = cv2.cvtColor(q_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    q_gray = cv2.cvtColor(q_bgr, cv2.COLOR_BGR2GRAY)
    q_tex = get_texture_map(q_gray)
    
    # 2. Generate Context 'Donut' Mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (161, 161)) 
    dilated_hole = cv2.dilate(mask_img, kernel)
    context_mask = ((dilated_hole > 0) & (~mask_bool)).astype(np.float32)
    num_pixels = np.sum(context_mask)
    
    # 3. Crop Templates to Bounding Box
    coords = np.argwhere(context_mask > 0)
    y1, x1, y2, x2 = coords[:,0].min(), coords[:,1].min(), coords[:,0].max()+1, coords[:,1].max()+1
    
    t_lab = q_lab[y1:y2, x1:x2]
    t_tex = q_tex[y1:y2, x1:x2]
    t_mask_1ch = context_mask[y1:y2, x1:x2]
    
    # Optional performance tweak: matchTemplate can broadcast a 1ch mask over a 3ch image
    t_mask_3ch = np.repeat(t_mask_1ch[:, :, np.newaxis], 3, axis=2)
    
    center_q = (y1 + t_lab.shape[0]/2.0, x1 + t_lab.shape[1]/2.0)
    
    # Fix 2a: Normalize query center to [0.0, 1.0]
    rel_center_q_y = center_q[0] / q_h
    rel_center_q_x = center_q[1] / q_w
    
    scales = [0.81, 0.90, 1.0]
    results = []

    for idx, match_img in enumerate(match_img_list):
        m_h, m_w = match_img.shape[:2]
        
        m_lab_full = cv2.cvtColor(match_img, cv2.COLOR_BGR2Lab).astype(np.float32)
        m_gray_full = cv2.cvtColor(match_img, cv2.COLOR_BGR2GRAY)
        m_tex_full = get_texture_map(m_gray_full)
        
        best_for_candidate = {'score': float('inf')}

        for s in scales:
            sh, sw = int(match_img.shape[0] * s), int(match_img.shape[1] * s)
            if sh < t_lab.shape[0] or sw < t_lab.shape[1]: continue
            
            s_lab = cv2.resize(m_lab_full, (sw, sh))
            s_tex = cv2.resize(m_tex_full, (sw, sh))
            
            # Pass 1: Match Lab (3-channel)
            ssd_lab = cv2.matchTemplate(s_lab, t_lab, cv2.TM_SQDIFF, mask=t_mask_3ch)
            
            # Pass 2: Match Texture (1-channel)
            ssd_tex = cv2.matchTemplate(s_tex, t_tex, cv2.TM_SQDIFF, mask=t_mask_1ch)
            
            # Combine scores with weight
            ssd_total = ssd_lab + (weight_tex * ssd_tex)
            
            # 4. Vectorized Spatial Penalty (Using Relative Coordinates)
            res_h, res_w = ssd_total.shape
            yy, xx = np.mgrid[0:res_h, 0:res_w]
            
            # Get unscaled candidate pixel centers
            m_centers_y = (yy + t_lab.shape[0]/2.0) / s
            m_centers_x = (xx + t_lab.shape[1]/2.0) / s
            
            # Fix 2b: Normalize candidate centers to [0.0, 1.0]
            rel_m_centers_y = m_centers_y / m_h
            rel_m_centers_x = m_centers_x / m_w
            
            # Calculate Euclidean distance in relative space
            dist = np.sqrt((rel_center_q_y - rel_m_centers_y)**2 + (rel_center_q_x - rel_m_centers_x)**2)
            
            # Apply penalty factor (0.1 relative distance = 10% displacement of image dimensions)
            penalty_map = 1.0 + (dist / 0.1) 
            
            score_map = (ssd_total / num_pixels) * penalty_map
            
            min_val, _, min_loc, _ = cv2.minMaxLoc(score_map)
            
            if min_val < best_for_candidate['score']:
                best_for_candidate = {
                    'score': min_val,
                    'placement': (s, min_loc[0], min_loc[1])
                }

        results.append({
            'match_idx': idx,
            'score': best_for_candidate['score'],
            'placement': best_for_candidate.get('placement')
        })

    return sorted(results, key=lambda x: x['score'])