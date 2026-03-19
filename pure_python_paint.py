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
    iterations = 900 
    # for _ in range(iterations):
    #     b_pad = pad_1px(blend)
        
    #     # Calculate next state
    #     b_next = (b_pad[:-2, 1:-1] + b_pad[2:, 1:-1] + b_pad[1:-1, :-2] + b_pad[1:-1, 2:] - laplacian) / 4.0
        
    #     # Anchor the outside pixels, update the inside pixels
    #     blend = mask_3d * b_next + (~mask_3d) * roi
    h, w, _ = blend.shape

    for _ in range(iterations):
        for y in range(1, h-1):
            for x in range(1, w-1):
                if mask_bool[y, x]:
                    for c in range(3):
                        blend[y, x, c] = (
                            blend[y-1, x, c] +
                            blend[y+1, x, c] +
                            blend[y, x-1, c] +
                            blend[y, x+1, c] +
                            laplacian[y, x, c]
                        ) / 4.0
        
    # Arithmetic clipping
    blend = blend * (blend >= 0) + 0 * (blend < 0)
    blend = blend * (blend <= 255) + 255 * (blend > 255)
    
    # Paste the perfectly aligned patch back in
    out = dst.copy()
    out[dst_top:dst_bottom, dst_left:dst_right] = blend.astype(np.uint8)
    
    return out

def main(args):
    global root, canvas, WIDTH, HEIGHT, ui_mask, bg_img, image_path
    
    root = tk.Tk()
    root.withdraw()
    
    image_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    
    if not image_path:
        sys.exit(0)

    # 1. Resize to 1024px (The Paper's Database Standard)
    # We use this resolution for the UI and the final output mask
    max_dim = 1024
    temp_png = "temp_ui_background.png"
    
    # -Z 1024 ensures the longest edge is 1024, matching the paper's scene matching specs
    subprocess.run(["sips", "-Z", str(max_dim), "-s", "format", "png", image_path, "--out", temp_png], capture_output=True)
    
    root.deiconify()
    bg_img = tk.PhotoImage(file=temp_png)
    WIDTH, HEIGHT = bg_img.width(), bg_img.height()
    root.title(f"Draw Mask (1024px Standard) - {os.path.basename(image_path)}")

    # 2. Initialize mask at the current (1024px) resolution
    ui_mask = [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]
    
    canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, cursor="crosshair", bg="black")
    canvas.pack()
    canvas.create_image(0, 0, image=bg_img, anchor="nw")

    def draw(event):
        r = 20 # Brush radius
        # Draw on the screen
        canvas.create_oval(event.x-r, event.y-r, event.x+r, event.y+r, fill="white", outline="white")
        # Record in the mask array
        # for i in range(max(0, event.y-r), min(HEIGHT, event.y+r)):
        #     for j in range(max(0, event.x-r), min(WIDTH, event.x+r)):
        #         ui_mask[i][j] = 255
        for i in range(max(0, event.y-r), min(HEIGHT, event.y+r)):
            for j in range(max(0, event.x-r), min(WIDTH, event.x+r)):
                if (i-event.y)**2 + (j-event.x)**2 <= r*r:
                    ui_mask[i][j] = 255

    def finish_drawing(event):
        print("Saving mask at 1024px resolution...")
        # Create image directly from the UI mask array
        final_mask = Image.new("L", (WIDTH, HEIGHT), 0)
        pixels = final_mask.load()
        
        for y in range(HEIGHT):
            for x in range(WIDTH):
                if ui_mask[y][x] == 255:
                    pixels[x, y] = 255
        
        # Save the 1024px mask
        output_name = "mask_1024.png"
        final_mask.save(output_name)
        print(f"Success: Mask saved as {output_name} ({WIDTH}x{HEIGHT})")
        
        # Optional: Save a 1024px version of the original image for easy matching
        # This ensures the 'Input Image' and 'Mask' are identical sizes for the algorithm
        os.rename(temp_png, "image_1024.png")
        
        #root.destroy()
        root.quit()
        root.destroy()
        
        # Import your matching function
        from match_scenes import find_k_best_matches
        
        # Pass the 1024px image and mask, along with your database folder (e.g., 'beaches')
        matches = find_k_best_matches("image_1024.png", "mask_1024.png", "skyline_1024", k=12)
        
        match_img_bgr_list = []
        valid_matches = []
        #print(matches)
        for rank, (score, path, fname) in enumerate(matches, 1):
            print(f"{rank}. {fname} (Score: {score:.4f})")
            # exit()
            img = cv2.imread(path)
            if img is not None:
                match_img_bgr_list.append(img)
                
                valid_matches.append(fname)
        
        print("the images")
        print(match_img_bgr_list[0].shape,match_img_bgr_list[1].shape)
        # exit()
        local_results = match_context_optimized("image_1024.png", "mask_1024.png", match_img_bgr_list)
        
        print("\nLocal Context Matching Results:")
        print(local_results)
                
        if args.use_ef1:
            print("\n--- EF1 Enabled: Finding the mathematically best seam ---")
            evaluated_candidates = []
            
            # Phase 1: Evaluate all top matches
            for i in range(1,min(4, len(local_results))):
                best_match = local_results[i]
                best_img_idx = best_match['match_idx']
                best_scale, min_x, min_y = best_match['placement']

                # Reload Query & Masks to get the Bounding Box 
                q_bgr = cv2.imread("image_1024.png")
                mask_img = cv2.imread("mask_1024.png", cv2.IMREAD_GRAYSCALE)

                # mask_bool = mask_img > 127
                # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (161, 161)) 
                # dilated_hole = cv2.dilate(mask_img, kernel)
                mask_bool = mask_img > 127
                dilated_hole = dilate_pure(mask_img, kernel_size=161)
                # context_mask = ((dilated_hole > 0) & (~mask_bool)).astype(np.uint8) * 255
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
                # best_img_scaled = cv2.resize(best_img, (sw, sh))
                # best_img_padded = cv2.copyMakeBorder(best_img_scaled, pad, pad, pad, pad, cv2.BORDER_REFLECT)
                best_img_scaled = resize_image_pure(best_img, sw, sh) # From our previous step
                best_img_padded = pad_reflect_pure(best_img_scaled, pad)

                start_y = int(min_y) + pad - pad_top
                start_x = int(min_x) + pad - pad_left
                m_crop = best_img_padded[start_y : start_y + box_h, start_x : start_x + box_w]
                
                #m_crop = color_transfer(m_crop, q_crop) # Uncomment if using color_transfer
                
                # Calculate seam and energy
                seam_mask, seam_energy = find_optimal_seam(q_crop, m_crop, hole_mask_crop, context_mask_crop,first_component=True)
                
                evaluated_candidates.append({
                    'index': i,
                    'm_crop': m_crop,
                    'seam_mask': seam_mask,
                    'energy': seam_energy,
                    'x1': x1,
                    'y1': y1
                })
            
            # Phase 2: Sort by lowest energy
            evaluated_candidates.sort(key=lambda x: x['energy'])
            winner = evaluated_candidates[0]
            print(f"\nEF1 automatically selected match index {winner['index']} with lowest energy: {winner['energy']}")
            
            # Phase 3: Blend ONLY the winner
            # x, y, w_mask, h_mask = cv2.boundingRect(winner['seam_mask'])
            # x, y, w_mask, h_mask = bounding_rect_pure(winner['seam_mask'])
            # center_x = int(winner['x1'] + x + (w_mask / 2))
            # center_y = int(winner['y1'] + y + (h_mask / 2))
            h_crop, w_crop = winner['m_crop'].shape[:2]
            center_x = int(winner['x1'] + (w_crop / 2))
            center_y = int(winner['y1'] + (h_crop / 2))
            
            # final_result = cv2.seamlessClone(
            #     src=winner['m_crop'], dst=q_bgr, mask=winner['seam_mask'], 
            #     p=(center_x, center_y), flags=cv2.NORMAL_CLONE
            # )
            
            final_result = seamless_clone_pure(
                src=winner['m_crop'], 
                dst=q_bgr, 
                mask=winner['seam_mask'], 
                center=(center_x, center_y)
            )
            
            cv2.imwrite("final_completed_image_EF1_BEST.png", final_result)
            print("EF1 Pipeline complete! Saved as final_completed_image_EF1_BEST.png")
        
        else:
            print("\n--- Base Pipeline: Saving all top matches ---")
            for i in range(min(4, len(local_results))):
                best_match = local_results[i]
                best_img_idx = best_match['match_idx']
                best_scale, min_x, min_y = best_match['placement']

                # Reload Query & Masks to get the Bounding Box 
                q_bgr = cv2.imread("image_1024.png")
                mask_img = cv2.imread("mask_1024.png", cv2.IMREAD_GRAYSCALE)

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
                # best_img_scaled = cv2.resize(best_img, (sw, sh))
                # best_img_padded = cv2.copyMakeBorder(best_img_scaled, pad, pad, pad, pad, cv2.BORDER_REFLECT)
                best_img_scaled = resize_image_pure(best_img, sw, sh) # From our previous step
                best_img_padded = pad_reflect_pure(best_img_scaled, pad)

                start_y = int(min_y) + pad - pad_top
                start_x = int(min_x) + pad - pad_left
                m_crop = best_img_padded[start_y : start_y + box_h, start_x : start_x + box_w]
                
                #m_crop = color_transfer(m_crop, q_crop)
                
                cv2.imwrite(f"debug_match_{i}.png", m_crop)
                
                # Unpack both values, but we only care about the mask in the base pipeline
                seam_mask= find_optimal_seam(q_crop, m_crop, hole_mask_crop, context_mask_crop,first_component=False)

                # x, y, w_mask, h_mask = cv2.boundingRect(seam_mask)
                # x, y, w_mask, h_mask = bounding_rect_pure(seam_mask)
                # center_x = int(x1 + x + (w_mask / 2))
                # center_y = int(y1 + y + (h_mask / 2))
                
                h_crop, w_crop = m_crop.shape[:2]
                center_x = int(x1 + (w_crop / 2))
                center_y = int(y1 + (h_crop / 2))

                # final_result = cv2.seamlessClone(
                #     src=m_crop, dst=q_bgr, mask=seam_mask, p=(center_x, center_y), flags=cv2.NORMAL_CLONE
                # )
                cv2.imwrite(f"debug_seam_{i}.png", seam_mask * 255)
                
                final_result = seamless_clone_pure(
                src=m_crop, 
                dst=q_bgr, 
                mask=seam_mask, 
                center=(center_x, center_y)
                )
                
                cv2.imwrite(f"final_completed_image_{i}.png", final_result)
            print("Base Pipeline complete! Saved all top candidates.")
        
        # root.quit()
        # root.destroy()
        #exit()
        

        
        
    canvas.bind("<B1-Motion>", draw)
    root.bind("q", finish_drawing)
    
    print("Instructions:")
    print("1. Drag to paint the area you want to remove.")
    print("2. Press 'q' to save the 1024px mask and exit.")
    
    root.mainloop()

if __name__ == "__main__":
    # main()
    parser = argparse.ArgumentParser(description="Scene Completion Pipeline")
    parser.add_argument('--use_ef1', action='store_true', help='Enable Automatic Ranking (EF1)')
    
    args = parser.parse_args()
    
    main(args)