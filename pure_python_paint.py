import tkinter as tk
from tkinter import filedialog
import os, sys, subprocess
from PIL import Image
from local_context_matching import match_context_optimized
import cv2
from graph_cut import find_optimal_seam
import numpy as np
from skimage.exposure import match_histograms
import argparse


import numpy as np


def bgr_to_lab_pure(img_bgr):
    # 1. Normalize and split
    b = img_bgr[:, :, 0] / 255.0
    g = img_bgr[:, :, 1] / 255.0
    r = img_bgr[:, :, 2] / 255.0

    # 2. Inverse sRGB Gamma (Arithmetic masking instead of np.where)
    def gamma_inv(c):
        mask_high = (c > 0.04045)
        mask_low = (c <= 0.04045)
        return mask_high * (((c + 0.055) / 1.055) ** 2.4) + mask_low * (c / 12.92)

    r, g, b = gamma_inv(r), gamma_inv(g), gamma_inv(b)

    # 3. RGB to XYZ Matrix Multiplication
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # 4. XYZ to LAB Conversion (Normalized by D65 Illuminant)
    x /= 0.95047
    y /= 1.00000
    z /= 1.08883

    def f_xyz(t):
        mask_high = (t > 0.008856)
        mask_low = (t <= 0.008856)
        return mask_high * (t ** (1/3)) + mask_low * ((7.787 * t) + (16 / 116))

    fx, fy, fz = f_xyz(x), f_xyz(y), f_xyz(z)

    l = (116 * fy) - 16
    a = 500 * (fx - fy)
    b_channel = 200 * (fy - fz)

    # 5. Stack into LAB array
    lab = np.zeros_like(img_bgr, dtype=np.float32)
    lab[:, :, 0] = l
    lab[:, :, 1] = a
    lab[:, :, 2] = b_channel
    return lab

def lab_to_bgr_pure(img_lab):
    l = img_lab[:, :, 0]
    a = img_lab[:, :, 1]
    b_channel = img_lab[:, :, 2]

    # 1. LAB to XYZ
    fy = (l + 16) / 116
    fx = (a / 500) + fy
    fz = fy - (b_channel / 200)

    def f_inv(t):
        t3 = t ** 3
        mask_high = (t3 > 0.008856)
        mask_low = (t3 <= 0.008856)
        return mask_high * t3 + mask_low * ((t - 16/116) / 7.787)

    x = f_inv(fx) * 0.95047
    y = f_inv(fy) * 1.00000
    z = f_inv(fz) * 1.08883

    # 2. XYZ to RGB Matrix Multiplication
    r = x * 3.2404542 - y * 1.5371385 - z * 0.4985314
    g = -x * 0.9692660 + y * 1.8760108 + z * 0.0415560
    b = x * 0.0556434 - y * 0.2040259 + z * 1.0572252

    # 3. sRGB Gamma Correction
    def gamma_fwd(c):
        mask_high = (c > 0.0031308)
        mask_low = (c <= 0.0031308)
        return mask_high * (1.055 * (c ** (1/2.4)) - 0.055) + mask_low * (12.92 * c)

    r, g, b = gamma_fwd(r), gamma_fwd(g), gamma_fwd(b)

    # 4. Scale and pure-math clip (No np.clip allowed)
    bgr = np.zeros_like(img_lab, dtype=np.float32)
    bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2] = b * 255, g * 255, r * 255

    # Pure arithmetic clipping to 0-255 bounds
    bgr = bgr * (bgr >= 0) + 0 * (bgr < 0)
    bgr = bgr * (bgr <= 255) + 255 * (bgr > 255)

    return bgr.astype(np.uint8)

def get_mean_std_pure(channel):
    n = channel.shape[0] * channel.shape[1]
    mean_val = np.sum(channel) / n
    # Variance = Sum of squared differences / n
    std_val = (np.sum((channel - mean_val) ** 2) / n) ** 0.5
    return mean_val, std_val

def color_transfer(source, target):
    # 1. Convert to LAB
    s_lab = bgr_to_lab_pure(source.astype(np.float32))
    t_lab = bgr_to_lab_pure(target.astype(np.float32))

    # 2. Compute Mean and Std for each channel
    s_means, s_stds, t_means, t_stds = [], [], [], []
    for i in range(3):
        sm, ss = get_mean_std_pure(s_lab[:, :, i])
        tm, ts = get_mean_std_pure(t_lab[:, :, i])
        s_means.append(sm); s_stds.append(ss)
        t_means.append(tm); t_stds.append(ts)

    # 3. Shift source colors
    for i in range(3):
        s_lab[:, :, i] -= s_means[i]
        s_lab[:, :, i] = (s_lab[:, :, i] * (t_stds[i] / (s_stds[i] + 1e-5))) + t_means[i]

    # 4. Convert back to BGR
    return lab_to_bgr_pure(s_lab)

def resize_image_pure(img, new_width, new_height):
    """
    Pure Numpy/Math implementation of cv2.resize using Bilinear Interpolation.
    """
    old_height, old_width, channels = img.shape

    # Create an empty array for the new image
    resized = np.zeros((new_height, new_width, channels), dtype=np.float32)

    # Calculate the scaling factors
    x_ratio = float(old_width - 1) / (new_width - 1) if new_width > 1 else 0
    y_ratio = float(old_height - 1) / (new_height - 1) if new_height > 1 else 0

    # Create grid of coordinates for the new image
    y_coords = np.arange(new_height)
    x_coords = np.arange(new_width)

    # Map new coordinates to old coordinates
    y_old = y_coords * y_ratio
    x_old = x_coords * x_ratio

    # Get the integer parts (top-left pixel coordinates)
    y_low = np.floor(y_old).astype(np.int32)
    x_low = np.floor(x_old).astype(np.int32)

    # Get the bottom-right pixel coordinates (bound by max width/height)
    y_high = np.clip(y_low + 1, 0, old_height - 1)
    x_high = np.clip(x_low + 1, 0, old_width - 1)

    # Get the decimal parts (weights for interpolation)
    y_weight = y_old - y_low
    x_weight = x_old - x_low

    # Expand weights to match channel dimensions for broadcasting
    y_weight = y_weight[:, np.newaxis, np.newaxis]
    x_weight = x_weight[np.newaxis, :, np.newaxis]

    # Perform the Bilinear interpolation for all channels at once
    for c in range(channels):
        # Extract the 4 surrounding pixels for all points
        top_left = img[y_low[:, None], x_low, c]
        top_right = img[y_low[:, None], x_high, c]
        bottom_left = img[y_high[:, None], x_low, c]
        bottom_right = img[y_high[:, None], x_high, c]

        # Calculate horizontal interpolations
        top = top_left * (1 - x_weight[:, :, 0]) + top_right * x_weight[:, :, 0]
        bottom = bottom_left * (1 - x_weight[:, :, 0]) + bottom_right * x_weight[:, :, 0]

        # Calculate vertical interpolation
        resized[:, :, c] = top * (1 - y_weight[:, :, 0]) + bottom * y_weight[:, :, 0]

    # Arithmetic clip and convert back to uint8
    resized = resized * (resized >= 0) + 0 * (resized < 0)
    resized = resized * (resized <= 255) + 255 * (resized > 255)

    return resized.astype(np.uint8)

def dilate_pure(mask, kernel_size=161):
    """
    Pure math replacement for cv2.dilate with a circular/elliptical kernel.
    Instead of a slow sliding window, we draw the radius around active pixels.
    """
    h, w = mask.shape
    radius = kernel_size // 2
    out = np.zeros((h, w), dtype=np.uint8)

    # Find coordinates of all white pixels
    ys, xs = np.nonzero(mask)

    # Apply the mathematical circle equation to active regions
    for y, x in zip(ys, xs):
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)

        y_grid = np.arange(y0, y1)[:, None]
        x_grid = np.arange(x0, x1)[None, :]

        # (x - h)^2 + (y - k)^2 <= r^2
        circle_mask = ((y_grid - y)**2 + (x_grid - x)**2) <= radius**2
        out[y0:y1, x0:x1] += circle_mask.astype(np.uint8)

    # Cap values at 255 using pure arithmetic
    out = (out > 0) * 255
    return out.astype(np.uint8)

def bounding_rect_pure(mask):
    """
    Pure math replacement for cv2.boundingRect.
    Finds the extreme x and y coordinates of non-zero pixels.
    """
    # Collapse the 2D array to 1D arrays to find where pixels exist
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    # Find the first and last True values
    ymin, ymax = np.argmax(rows), mask.shape[0] - 1 - np.argmax(rows[::-1])
    xmin, xmax = np.argmax(cols), mask.shape[1] - 1 - np.argmax(cols[::-1])

    width = xmax - xmin + 1
    height = ymax - ymin + 1

    return xmin, ymin, width, height

def pad_reflect_pure(img, pad):
    """
    Pure math replacement for cv2.copyMakeBorder(..., cv2.BORDER_REFLECT).
    It mathematically mirrors the arrays across the edges.
    """
    H, W, C = img.shape
    padded = np.zeros((H + 2*pad, W + 2*pad, C), dtype=img.dtype)

    # Insert the original image in the center
    padded[pad:pad+H, pad:pad+W] = img

    # Reflect Top and Bottom
    padded[:pad, pad:pad+W] = img[0:pad][::-1]
    padded[pad+H:, pad:pad+W] = img[H-pad:H][::-1]

    # Reflect Left and Right (including the corners we just made)
            # Left
    padded[:, :pad] = padded[:, pad:2*pad][:, ::-1]

        # Right
    padded[:, pad+W:] = padded[:, W:pad+W][:, ::-1]

    return padded

import numpy as np

def seamless_clone_pure(src, dst, mask, center):
    """
    Pure math implementation of cv2.seamlessClone (NORMAL_CLONE).
    Includes edge-collision detection to handle patches that exceed image boundaries.
    """
    src_h, src_w, channels = src.shape
    dst_h, dst_w, _ = dst.shape
    center_x, center_y = center

    # 1. Calculate ideal boundary coordinates
    top = center_y - src_h // 2
    left = center_x - src_w // 2
    bottom = top + src_h
    right = left + src_w

    # 2. Find how much to crop from src if it spills outside dst bounds
    src_top = max(0, -top)
    src_left = max(0, -left)
    src_bottom = src_h - max(0, bottom - dst_h)
    src_right = src_w - max(0, right - dst_w)

    # 3. Calculate safe bounds for dst (prevents negative indexing / truncation)
    dst_top = max(0, top)
    dst_left = max(0, left)
    dst_bottom = min(dst_h, bottom)
    dst_right = min(dst_w, right)

    # 4. Slice EVERYTHING safely so shapes perfectly align
    roi = dst[dst_top:dst_bottom, dst_left:dst_right].astype(np.float32)
    src_f = src[src_top:src_bottom, src_left:src_right].astype(np.float32)
    mask_crop = mask[src_top:src_bottom, src_left:src_right]

    # Normalize mask to boolean and expand to 3D for RGB broadcasting
    mask_bool = mask_crop > 127
    mask_3d = mask_bool[:, :, np.newaxis]

    # Helper function to safely pad boundaries by 1 pixel (Pure slicing)
    def pad_1px(img_array):
        h, w, c = img_array.shape
        padded = np.zeros((h + 2, w + 2, c), dtype=np.float32)
        padded[1:-1, 1:-1] = img_array
        padded[0, 1:-1] = img_array[0, :]   # Top edge
        padded[-1, 1:-1] = img_array[-1, :] # Bottom edge
        padded[1:-1, 0] = img_array[:, 0]   # Left edge
        padded[1:-1, -1] = img_array[:, -1] # Right edge
        return padded

    # Calculate the Laplacian (gradients) of the source image
    src_pad = pad_1px(src_f)
    laplacian = (4.0 * src_f) - src_pad[:-2, 1:-1] - src_pad[2:, 1:-1] - src_pad[1:-1, :-2] - src_pad[1:-1, 2:]

    # Initialize our working blend area
    blend = roi.copy()

    # Jacobi Iteration (The Solver)
    # NOTE: Jacobi needs O(n^2) iterations for an n-pixel-wide patch.
    # 900 = ~1% convergence for a 300px patch. Increase for better quality.
    iterations = 90000
    for _ in range(iterations):
        b_pad = pad_1px(blend)
        b_next = (b_pad[:-2, 1:-1] + b_pad[2:, 1:-1] + b_pad[1:-1, :-2] + b_pad[1:-1, 2:] + laplacian) / 4.0
        blend = mask_3d * b_next + (~mask_3d) * roi

    # Arithmetic clipping
    blend = blend * (blend >= 0) + 0 * (blend < 0)
    blend = blend * (blend <= 255) + 255 * (blend > 255)

    # Paste the perfectly aligned patch back in
    out = dst.copy()
    out[dst_top:dst_bottom, dst_left:dst_right] = blend.astype(np.uint8)

    return out


# ---------------------------------------------------------------------------
# EF1 helper metrics  (pure Python inside; numpy only for array indexing)
# ---------------------------------------------------------------------------

def boundary_gradient_coherence(q_crop, m_crop, seam_mask):
    """
    Mean absolute difference of gradient magnitudes at the seam boundary ring.
    Lower = smoother boundary.  Pure Python loops over boundary pixels only.
    """
    H, W = seam_mask.shape

    # Collect boundary pixels: seam pixels that have at least one 0-neighbour
    boundary = []
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if seam_mask[y, x] > 0:
                if (seam_mask[y-1, x] == 0 or seam_mask[y+1, x] == 0 or
                        seam_mask[y, x-1] == 0 or seam_mask[y, x+1] == 0):
                    boundary.append((y, x))

    if not boundary:
        return 0.0

    def lum(img, py, px):
        return (0.114 * float(img[py, px, 0]) +
                0.587 * float(img[py, px, 1]) +
                0.299 * float(img[py, px, 2]))

    total_diff = 0.0
    for (y, x) in boundary:
        gx_q = (lum(q_crop, y, x+1) - lum(q_crop, y, x-1)) * 0.5
        gy_q = (lum(q_crop, y+1, x) - lum(q_crop, y-1, x)) * 0.5
        gq   = (gx_q*gx_q + gy_q*gy_q) ** 0.5

        gx_m = (lum(m_crop, y, x+1) - lum(m_crop, y, x-1)) * 0.5
        gy_m = (lum(m_crop, y+1, x) - lum(m_crop, y-1, x)) * 0.5
        gm   = (gx_m*gx_m + gy_m*gy_m) ** 0.5

        diff = gq - gm
        total_diff += diff if diff >= 0 else -diff

    return total_diff / len(boundary)


def context_ncc(q_crop, m_crop, context_mask_crop):
    """
    Normalised Cross-Correlation between query and patch in the context ring.
    Returns (1 - NCC) so that lower = better.  Pure Python loops.
    """
    H, W = context_mask_crop.shape[:2]

    q_vals = []
    m_vals = []
    for y in range(H):
        for x in range(W):
            if context_mask_crop[y, x] > 0:
                q_v = (float(q_crop[y, x, 0]) + float(q_crop[y, x, 1]) + float(q_crop[y, x, 2])) / 3.0
                m_v = (float(m_crop[y, x, 0]) + float(m_crop[y, x, 1]) + float(m_crop[y, x, 2])) / 3.0
                q_vals.append(q_v)
                m_vals.append(m_v)

    n = len(q_vals)
    if n < 2:
        return 0.0

    q_mean = sum(q_vals) / n
    m_mean = sum(m_vals) / n

    numerator = 0.0
    denom_q   = 0.0
    denom_m   = 0.0
    for i in range(n):
        qc = q_vals[i] - q_mean
        mc = m_vals[i] - m_mean
        numerator += qc * mc
        denom_q   += qc * qc
        denom_m   += mc * mc

    denom = (denom_q ** 0.5) * (denom_m ** 0.5)
    if denom < 1e-9:
        return 0.0

    ncc = numerator / denom
    ncc = min(1.0, max(-1.0, ncc))
    return 1.0 - ncc   # lower = better


# ---------------------------------------------------------------------------
# Pipeline (extracted from finish_drawing so both EF2 and base UI can call it)
# ---------------------------------------------------------------------------

def run_completion_pipeline(image_1024_path, mask_1024_path, original_image_path, args):
    """
    Full scene-completion pipeline: matching → LCM → graph-cut → blending.
    Called after the mask has been saved to disk.

    EF3 behaviour (--use_ef3):
      - Uses skyline_tiny/ as the matching DB instead of skyline_1024/
      - Super-resolves each tiny matched image up to query resolution
      - Saves an additional *_EF3_original_size.png at the original input resolution
    """
    from match_scenes import find_k_best_matches

    # ------------------------------------------------------------------
    # EF3: choose DB folder
    # ------------------------------------------------------------------
    if args.use_ef3:
        db_dir = "skyline_tiny"
        if not os.path.isdir(db_dir):
            print(f"\n[EF3] ERROR: '{db_dir}/' not found.")
            print("[EF3] Create it first with:  python create_tiny_db.py")
            print("[EF3] Falling back to skyline_1024/ ...")
            db_dir = "skyline_1024"
            args.use_ef3 = False   # disable SR step if no tiny DB
        else:
            print(f"\n[EF3] Using tiny database '{db_dir}/' with super-resolution upsampling.")
    else:
        db_dir = "skyline_1024"
        
    print(f"\n[Step 3] Finding k={20} best matches in '{db_dir}/' for the query image and mask...")

    matches = find_k_best_matches(image_1024_path, mask_1024_path, db_dir, k=20)

    match_img_bgr_list = []
    valid_matches = []
    scene_scores = []

    # Step 4: Load k matched images (tiny if EF3, full-res otherwise) — NO SR yet
    for rank, (score, path, fname) in enumerate(matches, 1):
        print(f"{rank}. {fname} (Score: {score:.4f})")
        img = cv2.imread(path)
        if img is not None:
            match_img_bgr_list.append(img)
            valid_matches.append(fname)
            scene_scores.append(score)

    q_bgr = cv2.imread(image_1024_path)
    mask_gray = cv2.imread(mask_1024_path, cv2.IMREAD_GRAYSCALE)

    if args.use_ef3:
        # EF3: SR all k matched tiny images up to query resolution,
        # then let the base pipeline run as usual.
        from super_resolve import super_resolve_image
        target_h, target_w = q_bgr.shape[:2]
        print(f"\n[EF3] Super-resolving all {len(match_img_bgr_list)} matched tiny images to {target_w}x{target_h}...")
        sr_img_list = []
        for i, tiny in enumerate(match_img_bgr_list):
            # Scale by width ratio to preserve each image's aspect ratio.
            scale = target_w / tiny.shape[1]
            sr_h = int(tiny.shape[0] * scale)
            sr_w = target_w
            print(f"   [EF3-SR] rank {i+1}: {tiny.shape[1]}x{tiny.shape[0]} → {sr_w}x{sr_h}")
            sr_img_list.append(super_resolve_image(tiny, sr_h, sr_w))
        match_img_bgr_list = sr_img_list

    # Step 5: LCM on (SR'd if EF3, or full-res) DB images
    print("\n[Step 5] Running LCM on matched images...")
    local_results = match_context_optimized(q_bgr, mask_gray, match_img_bgr_list)

    print("\nLocal Context Matching Results:")
    print(local_results)

    # Track output filenames for EF3 final-resize step
    output_files = []

    if args.use_ef1:
        print("\n--- EF1: LCM + Seam Energy Ranking (ranks 1-20, skip rank 0 = same image) ---")
        evaluated_candidates = []

        seen_img_indices = set()
        for i in range(1, min(20, len(local_results))):
            best_match = local_results[i]
            if best_match['placement'] is None:
                print(f"  Skipping LCM rank {i}: no valid placement found")
                continue
            best_img_idx = best_match['match_idx']
            if best_img_idx in seen_img_indices:
                print(f"  Skipping LCM rank {i}: duplicate image (match_idx={best_img_idx})")
                continue
            seen_img_indices.add(best_img_idx)
            best_scale, min_x, min_y = best_match['placement']

            q_bgr = cv2.imread(image_1024_path)
            mask_img = cv2.imread(mask_1024_path, cv2.IMREAD_GRAYSCALE)

            mask_bool = mask_img > 127
            dilated_hole = dilate_pure(mask_img, kernel_size=161)
            context_mask = ((dilated_hole > 0) & (~mask_bool)).astype(np.uint8) * 255

            coords = np.argwhere(context_mask > 0)
            orig_y1, orig_x1 = coords[:,0].min(), coords[:,1].min()
            orig_y2, orig_x2 = coords[:,0].max()+1, coords[:,1].max()+1

            pad = 100
            y1, x1 = max(0, orig_y1 - pad), max(0, orig_x1 - pad)
            y2, x2 = min(q_bgr.shape[0], orig_y2 + pad), min(q_bgr.shape[1], orig_x2 + pad)
            pad_top, pad_left = orig_y1 - y1, orig_x1 - x1
            box_h, box_w = y2 - y1, x2 - x1

            q_crop = q_bgr[y1:y2, x1:x2]
            hole_mask_crop = mask_img[y1:y2, x1:x2]
            context_mask_crop = context_mask[y1:y2, x1:x2]

            best_img = match_img_bgr_list[best_img_idx]
            sh, sw = int(best_img.shape[0] * best_scale), int(best_img.shape[1] * best_scale)
            best_img_scaled = resize_image_pure(best_img, sw, sh)
            best_img_padded = pad_reflect_pure(best_img_scaled, pad)

            start_y = int(min_y) + pad - pad_top
            start_x = int(min_x) + pad - pad_left
            m_crop = best_img_padded[start_y : start_y + box_h, start_x : start_x + box_w]

            m_crop = color_transfer(m_crop, q_crop)

            seam_mask, seam_energy = find_optimal_seam(q_crop, m_crop, hole_mask_crop, context_mask_crop, first_component=True)

            coherence   = boundary_gradient_coherence(q_crop, m_crop, seam_mask)
            ncc_penalty = context_ncc(q_crop, m_crop, context_mask_crop)

            evaluated_candidates.append({
                'index': i,
                'm_crop': m_crop,
                'seam_mask': seam_mask,
                'scene_score': scene_scores[best_img_idx],
                'lcm_score': best_match['score'],
                'energy': seam_energy,
                'coherence': coherence,
                'ncc_penalty': ncc_penalty,
                'x1': x1,
                'y1': y1
            })

        def safe_mean(vals):
            m = sum(vals) / len(vals)
            return m if m != 0.0 else 1.0

        mean_lcm        = safe_mean([c['lcm_score']    for c in evaluated_candidates])
        mean_energy     = safe_mean([c['energy']       for c in evaluated_candidates])
        mean_coherence  = safe_mean([c['coherence']    for c in evaluated_candidates])
        mean_ncc        = safe_mean([c['ncc_penalty']  for c in evaluated_candidates])

        for c in evaluated_candidates:
            c['ef1_score'] = (c['lcm_score']   / mean_lcm       +
                              c['energy']       / mean_energy    +
                              c['coherence']    / mean_coherence +
                              c['ncc_penalty']  / mean_ncc)

        evaluated_candidates.sort(key=lambda x: x['ef1_score'])
        winner = evaluated_candidates[0]
        print(f"\nEF1 selected LCM rank {winner['index']} | "
              f"lcm={winner['lcm_score']:.4f}  energy={winner['energy']:.4f}  "
              f"coherence={winner['coherence']:.4f}  ncc_penalty={winner['ncc_penalty']:.4f}  "
              f"ef1_score={winner['ef1_score']:.4f}")

        h_crop, w_crop = winner['m_crop'].shape[:2]
        center_x = int(winner['x1'] + (w_crop / 2))
        center_y = int(winner['y1'] + (h_crop / 2))

        final_result = seamless_clone_pure(
            src=winner['m_crop'],
            dst=q_bgr,
            mask=winner['seam_mask'],
            center=(center_x, center_y)
        )

        out_name = "final_completed_image_EF1_BEST.png"
        cv2.imwrite(out_name, final_result)
        output_files.append(out_name)
        print(f"EF1 Pipeline complete! Saved as {out_name}")

    else:
        print("\n--- Base Pipeline: 4-Component Composite Ranking (top 20 LCM) ---")
        base_candidates = []

        q_bgr = cv2.imread(image_1024_path)
        mask_img = cv2.imread(mask_1024_path, cv2.IMREAD_GRAYSCALE)
        mask_bool = mask_img > 127
        dilated_hole = dilate_pure(mask_img, kernel_size=161)
        context_mask = ((dilated_hole > 0) & (~mask_bool)).astype(np.uint8) * 255

        coords = np.argwhere(context_mask > 0)
        orig_y1, orig_x1 = coords[:,0].min(), coords[:,1].min()
        orig_y2, orig_x2 = coords[:,0].max()+1, coords[:,1].max()+1
        pad = 100
        y1, x1 = max(0, orig_y1 - pad), max(0, orig_x1 - pad)
        y2, x2 = min(q_bgr.shape[0], orig_y2 + pad), min(q_bgr.shape[1], orig_x2 + pad)
        pad_top, pad_left = orig_y1 - y1, orig_x1 - x1
        box_h, box_w = y2 - y1, x2 - x1
        q_crop = q_bgr[y1:y2, x1:x2]
        hole_mask_crop = mask_img[y1:y2, x1:x2]
        context_mask_crop = context_mask[y1:y2, x1:x2]

        seen_img_indices = set()
        for i in range(min(20, len(local_results))):
            best_match = local_results[i]
            if best_match['placement'] is None:
                print(f"  Skipping LCM rank {i}: no valid placement found")
                continue
            best_img_idx = best_match['match_idx']
            if best_img_idx in seen_img_indices:
                print(f"  Skipping LCM rank {i}: duplicate image (match_idx={best_img_idx})")
                continue
            seen_img_indices.add(best_img_idx)
            best_scale, min_x, min_y = best_match['placement']

            best_img = match_img_bgr_list[best_img_idx]
            sh, sw = int(best_img.shape[0] * best_scale), int(best_img.shape[1] * best_scale)
            best_img_scaled = resize_image_pure(best_img, sw, sh)
            best_img_padded = pad_reflect_pure(best_img_scaled, pad)

            start_y = int(min_y) + pad - pad_top
            start_x = int(min_x) + pad - pad_left
            m_crop = best_img_padded[start_y : start_y + box_h, start_x : start_x + box_w]

            m_crop = color_transfer(m_crop, q_crop)

            seam_mask, seam_energy = find_optimal_seam(q_crop, m_crop, hole_mask_crop, context_mask_crop, first_component=True)

            base_candidates.append({
                'index': i,
                'm_crop': m_crop,
                'seam_mask': seam_mask,
                'scene_score': scene_scores[best_img_idx],
                'lcm_score': best_match['score'],
                'energy': seam_energy,
                'x1': x1,
                'y1': y1
            })
            print(f"  Evaluated LCM rank {i}: scene={scene_scores[best_img_idx]:.4f} "
                  f"lcm={best_match['score']:.4f} energy={seam_energy:.4f}")

        def safe_mean(vals):
            m = sum(vals) / len(vals)
            return m if m != 0.0 else 1.0

        mean_scene  = safe_mean([c['scene_score'] for c in base_candidates])
        mean_lcm    = safe_mean([c['lcm_score']   for c in base_candidates])
        mean_energy = safe_mean([c['energy']       for c in base_candidates])

        for c in base_candidates:
            c['composite'] = (c['scene_score'] / mean_scene +
                              c['lcm_score']   / mean_lcm   +
                              c['energy']       / mean_energy)

        base_candidates.sort(key=lambda x: x['composite'])

        for rank, cand in enumerate(base_candidates[:10]):
            # cv2.imwrite(f"debug_match_{rank}.png", cand['m_crop'])
            # cv2.imwrite(f"debug_seam_{rank}.png", cand['seam_mask'] * 255)

            h_crop, w_crop = cand['m_crop'].shape[:2]
            center_x = int(cand['x1'] + (w_crop / 2))
            center_y = int(cand['y1'] + (h_crop / 2))

            final_result = seamless_clone_pure(
                src=cand['m_crop'],
                dst=q_bgr,
                mask=cand['seam_mask'],
                center=(center_x, center_y)
            )
            out_name = f"final_completed_image_{rank}.png"
            cv2.imwrite(out_name, final_result)
            output_files.append(out_name)
            print(f"  Saved rank {rank}: lcm_rank={cand['index']} composite={cand['composite']:.4f}")

        print("Base Pipeline complete! Saved top 4 by composite score.")

    # ------------------------------------------------------------------
    # EF3 final step: resize completed outputs to original input resolution.
    # The input image is NOT super-resolved – we just go back to its original
    # dimensions using high-quality PIL LANCZOS (plain resize, not SR).
    # ------------------------------------------------------------------
    # if args.use_ef3 and output_files:
    #     orig_pil = Image.open(original_image_path)
    #     orig_w, orig_h = orig_pil.size
    #     print(f"\n[EF3] Resizing output(s) to original input size: {orig_w}x{orig_h}")

    #     for fname in output_files:
    #         completed_bgr = cv2.imread(fname)
    #         if completed_bgr is None:
    #             continue
    #         # Pure PIL resize – input is NOT super-resolved
    #         completed_rgb = completed_bgr[:, :, ::-1]
    #         pil_out = Image.fromarray(completed_rgb.astype(np.uint8))
    #         pil_out = pil_out.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
    #         out_bgr = np.array(pil_out)[:, :, ::-1]
    #         ef3_name = fname.replace('.png', '_EF3_original_size.png')
    #         cv2.imwrite(ef3_name, out_bgr)
    #         print(f"  [EF3] Saved at original size: {ef3_name}")


# ---------------------------------------------------------------------------
# Main: UI
# ---------------------------------------------------------------------------

def main(args):
    global root, canvas, WIDTH, HEIGHT, bg_img, image_path

    max_dim = 1024
    temp_png = "temp_ui_background.png"

    # ------------------------------------------------------------------
    # Fast path: both --image and --mask supplied → no UI needed at all
    # ------------------------------------------------------------------
    if args.image and args.mask:
        image_path = args.image
        subprocess.run(["sips", "-Z", str(max_dim), "-s", "format", "png",
                        image_path, "--out", "image_1024.png"], capture_output=True)
        img_1024 = Image.open("image_1024.png")
        w, h = img_1024.size
        mask_pil = Image.open(args.mask).convert("L")
        if mask_pil.size != (w, h):
            print(f"Resizing mask from {mask_pil.size} → ({w}, {h})...")
            mask_pil = mask_pil.resize((w, h), Image.NEAREST)
        mask_pil.save("mask_1024.png")
        print(f"Using pre-existing mask: {args.mask}")
        run_completion_pipeline("image_1024.png", "mask_1024.png", image_path, args)
        return

    root = tk.Tk()
    root.withdraw()

    if args.image:
        image_path = args.image
    else:
        image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )

    if not image_path:
        sys.exit(0)

    # 1. Resize to 1024px for UI / feature matching
    subprocess.run(["sips", "-Z", str(max_dim), "-s", "format", "png",
                    image_path, "--out", temp_png], capture_output=True)

    # ------------------------------------------------------------------
    # If --mask is provided, skip drawing UI and go straight to pipeline
    # ------------------------------------------------------------------
    if args.mask:
        img_1024 = Image.open(temp_png)
        w, h = img_1024.size
        mask_pil = Image.open(args.mask).convert("L")
        if mask_pil.size != (w, h):
            print(f"Resizing mask from {mask_pil.size} → ({w}, {h})...")
            mask_pil = mask_pil.resize((w, h), Image.NEAREST)
        mask_pil.save("mask_1024.png")
        os.rename(temp_png, "image_1024.png")
        root.destroy()
        print(f"Using pre-existing mask: {args.mask}")
        run_completion_pipeline("image_1024.png", "mask_1024.png", image_path, args)
        return

    root.deiconify()
    bg_img = tk.PhotoImage(file=temp_png)
    WIDTH, HEIGHT = bg_img.width(), bg_img.height()
    root.title(f"Draw Mask (1024px Standard) - {os.path.basename(image_path)}")

    canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, cursor="crosshair", bg="black")
    canvas.pack()
    canvas.create_image(0, 0, image=bg_img, anchor="nw")

    # ------------------------------------------------------------------
    # EF2: click-to-segment UI
    # ------------------------------------------------------------------
    if args.use_ef2:
        from PIL import ImageTk as PILImageTk

        from ef2_segmentation import (
            load_sam_predictor,
            predict_mask_at_point,
            merge_masks_pure,
            remove_mask_at_point_pure,
            render_overlay_pure,
        )

        # Load image as numpy array for overlay compositing (pure numpy)
        bg_pil = Image.open(temp_png).convert('RGB')
        bg_array_rgb = np.array(bg_pil)   # (H, W, 3) uint8, RGB

        # Load SAM predictor (third-party call lives inside ef2_segmentation.py)
        img_bgr_for_sam = bg_array_rgb[:, :, ::-1].copy()
        try:
            predictor = load_sam_predictor(img_bgr_for_sam)
        except (FileNotFoundError, ImportError) as e:
            print(f"\n[EF2] {e}\n")
            sys.exit(1)

        # State: list of bool (H, W) masks, one per click
        mask_list = []
        # Mutable container to prevent tkinter GC of PhotoImage
        tk_img_holder = [None]

        def render_overlay():
            """Recomposite and redisplay canvas with current mask overlay. Pure numpy."""
            combined = merge_masks_pure(mask_list, HEIGHT, WIDTH)
            overlay_rgb = render_overlay_pure(bg_array_rgb, combined, alpha=0.55)
            photo = PILImageTk.PhotoImage(Image.fromarray(overlay_rgb))
            tk_img_holder[0] = photo          # keep reference alive
            canvas.create_image(0, 0, image=photo, anchor="nw")

        def on_left_click(event):
            """Left-click: run SAM at click point and add the segment to the mask."""
            print(f"[EF2] Segmenting at ({event.x}, {event.y}) ...")
            mask = predict_mask_at_point(predictor, event.x, event.y, HEIGHT, WIDTH)
            mask_list.append(mask)
            render_overlay()

        def on_right_click(event):
            """Right-click: remove any segment that covers the clicked pixel. Pure Python."""
            mask_list[:] = remove_mask_at_point_pure(mask_list, event.x, event.y)
            render_overlay()

        def on_clear(event):
            """'c' key: clear all segments. Pure Python."""
            mask_list.clear()
            render_overlay()

        def finish_drawing_ef2(event):
            """'q' key: save combined SAM mask and launch the pipeline."""
            if not mask_list:
                print("[EF2] No regions selected. Click on objects first.")
                return

            print("[EF2] Saving segmentation mask ...")
            # merge_masks_pure is pure numpy – no third-party
            combined = merge_masks_pure(mask_list, HEIGHT, WIDTH)
            pil_mask = Image.fromarray(combined, mode='L')
            pil_mask.save("mask_1024.png")
            print(f"[EF2] Mask saved: mask_1024.png ({WIDTH}x{HEIGHT})")

            os.rename(temp_png, "image_1024.png")
            root.quit()
            root.destroy()
            run_completion_pipeline("image_1024.png", "mask_1024.png", image_path, args)

        canvas.bind("<Button-1>", on_left_click)
        canvas.bind("<Button-3>", on_right_click)
        root.bind("c", on_clear)
        root.bind("q", finish_drawing_ef2)

        print("\nEF2 Instructions:")
        print("  Left-click  : segment the clicked object/region (adds to mask)")
        print("  Right-click : remove segment at that point")
        print("  'c'         : clear all segments")
        print("  'q'         : confirm mask and run completion pipeline")

    # ------------------------------------------------------------------
    # Base UI: free-hand brush painting
    # ------------------------------------------------------------------
    else:
        # 2. Initialize mask at the current (1024px) resolution
        ui_mask = [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]

        def draw(event):
            r = 20  # Brush radius
            canvas.create_oval(event.x-r, event.y-r, event.x+r, event.y+r,
                               fill="white", outline="white")
            for i in range(max(0, event.y-r), min(HEIGHT, event.y+r)):
                for j in range(max(0, event.x-r), min(WIDTH, event.x+r)):
                    if (i-event.y)**2 + (j-event.x)**2 <= r*r:
                        ui_mask[i][j] = 255

        def finish_drawing(event):
            print("Saving mask at 1024px resolution...")
            final_mask = Image.new("L", (WIDTH, HEIGHT), 0)
            pixels = final_mask.load()

            for y in range(HEIGHT):
                for x in range(WIDTH):
                    if ui_mask[y][x] == 255:
                        pixels[x, y] = 255

            output_name = "mask_1024.png"
            final_mask.save(output_name)
            print(f"Success: Mask saved as {output_name} ({WIDTH}x{HEIGHT})")

            os.rename(temp_png, "image_1024.png")
            root.quit()
            root.destroy()
            run_completion_pipeline("image_1024.png", "mask_1024.png", image_path, args)

        canvas.bind("<B1-Motion>", draw)
        root.bind("q", finish_drawing)

        print("Instructions:")
        print("1. Drag to paint the area you want to remove.")
        print("2. Press 'q' to save the 1024px mask and exit.")

    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scene Completion Pipeline")
    parser.add_argument('--use_ef1', action='store_true',
                        help='EF1: Enable LCM + seam-energy automatic ranking')
    parser.add_argument('--use_ef2', action='store_true',
                        help='EF2: Use SAM click-to-segment mask painting interface')
    parser.add_argument('--use_ef3', action='store_true',
                        help='EF3: Match against tiny DB and super-resolve matches')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image (skips the file-selection dialog)')
    parser.add_argument('--mask', type=str, default=None,
                        help='Path to an existing mask PNG (skips the drawing UI)')

    args = parser.parse_args()
    main(args)