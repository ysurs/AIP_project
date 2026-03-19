import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image
import os
from scipy.ndimage import gaussian_filter
import math


# def get_gabor_kernel(sigma, theta, Lambda, gamma):
#     """Generates a zero-mean Gabor kernel (returns numpy ndarray)."""
    
#     half_size = int(3 * sigma)
#     size = 2 * half_size + 1
    
#     kernel = np.zeros((size, size), dtype=float)
    
#     for i, y in enumerate(range(-half_size, half_size + 1)):
#         for j, x in enumerate(range(-half_size, half_size + 1)):
            
#             # Rotate coordinates
#             x_theta = x * math.cos(theta) + y * math.sin(theta)
#             y_theta = -x * math.sin(theta) + y * math.cos(theta)
            
#             # Gaussian envelope
#             gaussian = math.exp(
#                 -0.5 * (
#                     (x_theta ** 2) / (sigma ** 2) +
#                     (gamma ** 2 * y_theta ** 2) / (sigma ** 2)
#                 )
#             )
            
#             # Cosine carrier
#             sinusoid = math.cos(2 * math.pi * x_theta / Lambda)
            
#             kernel[i, j] = gaussian * sinusoid
    
#     # Subtract mean (zero-mean property)
#     kernel -= kernel.mean()
    
#     return np.array(kernel)

# def pure_python_gaussian_kernel(sigma):
#     """Generates a 1D Gaussian kernel using standard python math."""
#     radius = int(math.ceil(3 * sigma))
#     kernel = []
    
#     for x in range(-radius, radius + 1):
#         kernel.append(math.exp(-(x**2) / (2 * sigma**2)))
        
#     total = sum(kernel)
#     return [k / total for k in kernel]

# def pure_python_gaussian_blur(img_list, sigma):
#     """Applies a 2D Gaussian blur using pure Python list comprehensions and separable 1D passes."""
#     kernel = pure_python_gaussian_kernel(sigma)
#     k_radius = len(kernel) // 2
    
#     rows = len(img_list)
#     cols = len(img_list[0])
    
#     # 1. Horizontal Pass
#     temp = [[0.0] * cols for _ in range(rows)]
#     for r in range(rows):
#         for c in range(cols):
#             val = 0.0
#             for i, k_val in enumerate(kernel):
#                 idx = c + i - k_radius
#                 # Reflective padding for boundaries
#                 if idx < 0:
#                     idx = -idx
#                 elif idx >= cols:
#                     idx = 2 * cols - 1 - idx
#                     # Safety clamp
#                     if idx < 0: idx = 0
#                     elif idx >= cols: idx = cols - 1
#                 val += img_list[r][idx] * k_val
#             temp[r][c] = val
            
#     # 2. Vertical Pass
#     blurred = [[0.0] * cols for _ in range(rows)]
#     for c in range(cols):
#         for r in range(rows):
#             val = 0.0
#             for i, k_val in enumerate(kernel):
#                 idx = r + i - k_radius
#                 if idx < 0:
#                     idx = -idx
#                 elif idx >= rows:
#                     idx = 2 * rows - 1 - idx
#                     if idx < 0: idx = 0
#                     elif idx >= rows: idx = rows - 1
#                 val += temp[idx][c] * k_val
#             blurred[r][c] = val
            
#     return blurred

# def local_contrast_normalization(img, sigma=10):
#     """Normalizes lighting and enhances local textures purely in standard Python."""
#     # Convert input to a standard Python list of lists
#     try:
#         img_list = img.tolist()
#     except AttributeError:
#         img_list = img

#     rows = len(img_list)
#     cols = len(img_list[0])
    
#     # Calculate local mean
#     local_mean = pure_python_gaussian_blur(img_list, sigma)
    
#     # Center the image and square the differences
#     centered = [[0.0] * cols for _ in range(rows)]
#     centered_sq = [[0.0] * cols for _ in range(rows)]
    
#     for r in range(rows):
#         for c in range(cols):
#             diff = img_list[r][c] - local_mean[r][c]
#             centered[r][c] = diff
#             centered_sq[r][c] = diff ** 2
            
#     # Calculate local variance (blur of the squared differences)
#     local_var_sq = pure_python_gaussian_blur(centered_sq, sigma)
    
#     threshold=0.02
#     # Final normalization
#     result = [[0.0] * cols for _ in range(rows)]
#     for r in range(rows):
#         for c in range(cols):
#             var = local_var_sq[r][c]
#             if var < 0: var = 0.0 # Prevent domain errors from float inaccuracies
#             local_var_sqrt = math.sqrt(var)
            
#             if local_var_sqrt < threshold:
#                 result[r][c] = 0.0
#             else:
#                 # Use a slightly larger epsilon (0.1) for better stability
#                 result[r][c] = centered[r][c] / (local_var_sqrt + 0.1)
            
#     return np.array(result)

# def compute_color_feature(image_path, blocks=4):
#     """
#     Computes the color feature by resizing the image to `blocks x blocks` 
#     and converting to L*a*b* color space.
#     """
#     if isinstance(image_path, str):
#         img = Image.open(image_path).convert('RGB')
#     else:
#         # Assume it's a NumPy array
#         if len(image_path.shape) == 2:
#             # Duplicate grayscale into 3 channels
#             image_path = np.stack((image_path,)*3, axis=-1)
        
#         # Ensure it's in uint8 format for PIL
#         if image_path.dtype != np.uint8:
#             if np.max(image_path) <= 1.0:
#                 img_uint8 = (image_path * 255).astype(np.uint8)
#             else:
#                 img_uint8 = image_path.astype(np.uint8)
#         else:
#             img_uint8 = image_path
            
#         img = Image.fromarray(img_uint8)
        
#     #print(f"Original image size: {img.size}")
#     # Resize to exactly 4x4
#     img_small = img.resize((blocks, blocks), Image.Resampling.LANCZOS)
#     #print(f"Resized image size for color feature: {img_small.size}")
#     img_small_array = np.array(img_small) / 255.0
    
#     # NumPy RGB to LAB conversion (Standard D65 Illuminant)
#     def rgb_to_xyz(rgb_input):
#         rgb = rgb_input.copy()
#         mask = rgb > 0.04045
#         rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
#         rgb[~mask] = rgb[~mask] / 12.92
#         rgb *= 100
        
#         x = rgb[:,:,0]*0.4124 + rgb[:,:,1]*0.3576 + rgb[:,:,2]*0.1805
#         y = rgb[:,:,0]*0.2126 + rgb[:,:,1]*0.7152 + rgb[:,:,2]*0.0722
#         z = rgb[:,:,0]*0.0193 + rgb[:,:,1]*0.1192 + rgb[:,:,2]*0.9505
#         return np.stack([x, y, z], axis=-1)
        
#     def xyz_to_lab(xyz):
#         xyz_ref_white = np.array([95.047, 100.0, 108.883])
#         xyz_normalized = xyz / xyz_ref_white
        
#         mask = xyz_normalized > 0.008856
#         xyz_normalized[mask] = np.cbrt(xyz_normalized[mask])
#         xyz_normalized[~mask] = (7.787 * xyz_normalized[~mask]) + (16 / 116)
        
#         L = (116 * xyz_normalized[:,:,1]) - 16
#         a = 500 * (xyz_normalized[:,:,0] - xyz_normalized[:,:,1])
#         b = 200 * (xyz_normalized[:,:,1] - xyz_normalized[:,:,2])
#         return np.stack([L, a, b], axis=-1)
        
#     xyz = rgb_to_xyz(img_small_array)
#     lab = xyz_to_lab(xyz)
    
#     return lab


# def compute_gist(image_path, mask_path=None, scales=5, orientations=6, blocks=4):
#     """
#     Computes Gist descriptor and spatial weights based on valid pixels.
#     Returns:
#         gist_data: (scales, orientations, blocks, blocks) descriptor
#         gist_weights: (blocks, blocks) weights for matching
#     """
#     # 1. Load Image
#     if isinstance(image_path, str):
#         img = Image.open(image_path).convert('L')
#         img = img.resize((256, 256), Image.Resampling.LANCZOS)
#         img_array = np.array(img) / 255.0
#     else:
#         img_array = image_path 

#     # 2. Initialize Weights & Handle Mask
#     # Default weights = 1.0 (fully valid)
#     gist_weights = np.ones((blocks, blocks), dtype=float)

#     if mask_path and os.path.exists(mask_path):
#         mask_img = Image.open(mask_path).convert('L')
#         mask_img = mask_img.resize((256, 256), Image.Resampling.NEAREST)
#         mask_array = np.array(mask_img) / 255.0
        
#         # Assume > 0.5 is hole/invalid (standard inpainting mask)
#         hole_mask = mask_array > 0.5
#         valid_mask = mask_array <= 0.5
        
#         # A. Fill hole smoothly to avoid hard edge artifacts during Gabor convolution
#         if np.any(valid_mask):
#             mean_val = np.mean(img_array[valid_mask])
#         else:
#             mean_val = 0.5

#         # Create a working copy
#         img_filled = img_array.copy()

#         # Initially fill with mean
#         img_filled[hole_mask] = mean_val

#         # Diffuse the valid pixels into the hole iteratively
#         # Iterations with a small sigma create a smooth inward bleed
#         for _ in range(20):
#             blurred_img = gaussian_filter(img_filled, sigma=2)
#             img_filled[hole_mask] = blurred_img[hole_mask]

#         # Assign back to original array
#         img_array = img_filled
        
#         # B. Compute Weights for each 4x4 spatial block
#         # This matches the paper's "weights each spatial bin"
#         h_img, w_img = mask_array.shape
#         block_h = h_img // blocks
#         block_w = w_img // blocks
        
#         for bh in range(blocks):
#             for bw in range(blocks):
#                 # Extract mask block
#                 mask_block = valid_mask[bh*block_h:(bh+1)*block_h, 
#                                         bw*block_w:(bw+1)*block_w]
                
#                 # Weight = percentage of valid pixels
#                 gist_weights[bh, bw] = np.sum(mask_block) / mask_block.size

#     # 3. Normalization and Gabor Filtering
#     img_array = local_contrast_normalization(img_array)
#     #print("Image array after local contrast normalization:")
#     #print(img_array.shape)
    
#     # def plot_histogram(arr):
#     #     plt.figure(figsize=(8, 4))
#     #     plt.hist(arr.flatten(), bins=100, color='crimson', alpha=0.7)
#     #     plt.title("Value Distribution (Histogram)")
#     #     plt.xlabel("Value")
#     #     plt.ylabel("Pixel Count")
#     #     plt.yscale('log') # Log scale helps find tiny amounts of noise
#     #     plt.grid(True, alpha=0.3)
#     #     plt.show()
#     # plot_histogram(img_array)
#     # exit()    
#     gist_data = np.zeros((scales, orientations, blocks, blocks))
#     sigmas = np.linspace(4.0, 1.0, scales) 
#     lambdas = sigmas * 2.5 
    
#     for s in range(scales):
#         for o in range(orientations):
#             theta = o * np.pi / orientations
#             kernel = get_gabor_kernel(sigmas[s], theta, lambdas[s], 0.5)
            
#             # 'symm' boundary reduces border artifacts
#             filtered = convolve2d(img_array, kernel, mode='same', boundary='symm')
#             energy = np.abs(filtered)
            
#             # 4. Spatial Pooling
#             h, w = energy.shape
#             for bh in range(blocks):
#                 for bw in range(blocks):
#                     block = energy[bh*h//blocks:(bh+1)*h//blocks, 
#                                    bw*w//blocks:(bw+1)*w//blocks]
#                     if mask_path and os.path.exists(mask_path):
#                         valid_block = valid_mask[bh*h//blocks:(bh+1)*h//blocks, 
#                                                  bw*w//blocks:(bw+1)*w//blocks]
#                         if np.sum(valid_block) > 0:
#                             gist_data[s, o, bh, bw] = np.sum(block[valid_block]) / np.sum(valid_block)
#                         else:
#                             gist_data[s, o, bh, bw] = 0.0
#                     else:
#                         gist_data[s, o, bh, bw] = np.mean(block)
    
#     # Returns BOTH the descriptor and the weights for proper matching
#     return gist_data, gist_weights

import math
import os
from PIL import Image

# ---------------------------------------------------------
# 1. CORE MATH & KERNEL HELPERS
# ---------------------------------------------------------

def get_gabor_kernel(sigma, theta, Lambda, gamma):
    """Generates a zero-mean Gabor kernel using standard math."""
    half_size = int(3 * sigma)
    size = 2 * half_size + 1
    
    kernel = [[0.0] * size for _ in range(size)]
    total = 0.0
    
    for i, y in enumerate(range(-half_size, half_size + 1)):
        for j, x in enumerate(range(-half_size, half_size + 1)):
            
            x_theta = x * math.cos(theta) + y * math.sin(theta)
            y_theta = -x * math.sin(theta) + y * math.cos(theta)
            
            gaussian = math.exp(
                -0.5 * (
                    (x_theta ** 2) / (sigma ** 2) +
                    (gamma ** 2 * y_theta ** 2) / (sigma ** 2)
                )
            )
            sinusoid = math.cos(2 * math.pi * x_theta / Lambda)
            
            val = gaussian * sinusoid
            kernel[i][j] = val
            total += val
            
    # Subtract mean (zero-mean property)
    mean_val = total / (size * size)
    for i in range(size):
        for j in range(size):
            kernel[i][j] -= mean_val
            
    return kernel

def pure_python_gaussian_kernel(sigma):
    radius = int(math.ceil(3 * sigma))
    kernel = []
    for x in range(-radius, radius + 1):
        kernel.append(math.exp(-(x**2) / (2 * sigma**2)))
    total = sum(kernel)
    return [k / total for k in kernel]

def pure_python_gaussian_blur(img_list, sigma):
    kernel = pure_python_gaussian_kernel(sigma)
    k_radius = len(kernel) // 2
    rows = len(img_list)
    cols = len(img_list[0])
    
    temp = [[0.0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            val = 0.0
            for i, k_val in enumerate(kernel):
                idx = c + i - k_radius
                if idx < 0: idx = -idx
                elif idx >= cols: idx = 2 * cols - 1 - idx
                if idx < 0: idx = 0
                elif idx >= cols: idx = cols - 1
                val += img_list[r][idx] * k_val
            temp[r][c] = val
            
    blurred = [[0.0] * cols for _ in range(rows)]
    for c in range(cols):
        for r in range(rows):
            val = 0.0
            for i, k_val in enumerate(kernel):
                idx = r + i - k_radius
                if idx < 0: idx = -idx
                elif idx >= rows: idx = 2 * rows - 1 - idx
                if idx < 0: idx = 0
                elif idx >= rows: idx = rows - 1
                val += temp[idx][c] * k_val
            blurred[r][c] = val
            
    return blurred

# def pure_convolve2d(image, kernel):
#     """Pure Python 2D convolution with symmetric boundaries."""
#     r_img, c_img = len(image), len(image[0])
#     r_k, c_k = len(kernel), len(kernel[0])
#     pad_r, pad_c = r_k // 2, c_k // 2

#     out = [[0.0] * c_img for _ in range(r_img)]
#     for i in range(r_img):
#         for j in range(c_img):
#             val = 0.0
#             for ki in range(r_k):
#                 for kj in range(c_k):
#                     img_i = i + ki - pad_r
#                     img_j = j + kj - pad_c

#                     # Symmetric boundary
#                     if img_i < 0: img_i = -img_i
#                     elif img_i >= r_img: img_i = 2 * r_img - 1 - img_i
#                     if img_j < 0: img_j = -img_j
#                     elif img_j >= c_img: img_j = 2 * c_img - 1 - img_j
                    
#                     # Safety clamp
#                     img_i = max(0, min(r_img - 1, img_i))
#                     img_j = max(0, min(c_img - 1, img_j))

#                     val += image[img_i][img_j] * kernel[ki][kj]
#             out[i][j] = val
#     return out

def pure_convolve2d_optimized(image_2d, kernel_2d):
    """
    Highly optimized pure-Python 2D convolution using 1D flattening,
    pre-computed index offsets, and C-optimized sum generators.
    """
    r_img, c_img = len(image_2d), len(image_2d[0])
    r_k, c_k = len(kernel_2d), len(kernel_2d[0])
    pad_r, pad_c = r_k // 2, c_k // 2
    
    # Width of our padded image
    w_pad = c_img + 2 * pad_c

    # 1. PRE-PAD AND FLATTEN THE IMAGE
    # Doing this once upfront removes boundary checks from the inner loops
    padded_1d = [0.0] * ((r_img + 2 * pad_r) * w_pad)
    idx = 0
    for i in range(-pad_r, r_img + pad_r):
        # Calculate symmetric row index
        mi = -i if i < 0 else (2 * r_img - 1 - i if i >= r_img else i)
        mi = max(0, min(r_img - 1, mi)) # Safety clamp
        
        for j in range(-pad_c, c_img + pad_c):
            # Calculate symmetric column index
            mj = -j if j < 0 else (2 * c_img - 1 - j if j >= c_img else j)
            mj = max(0, min(c_img - 1, mj)) # Safety clamp
            
            padded_1d[idx] = image_2d[mi][mj]
            idx += 1

    # 2. PRE-COMPUTE KERNEL OFFSETS
    # Instead of iterating x,y inside the kernel, we calculate the exact 
    # 1D index offsets relative to the top-left of the receptive field.
    kernel_1d = []
    offsets = []
    for ki in range(r_k):
        for kj in range(c_k):
            kernel_1d.append(kernel_2d[ki][kj])
            offsets.append(ki * w_pad + kj)

    # Bundle them into tuples for the fastest possible iteration unpacking
    k_data = list(zip(offsets, kernel_1d))

    # 3. THE OPTIMIZED CONVOLUTION LOOP
    out_2d = []
    for i in range(r_img):
        out_row = []
        row_start = i * w_pad # Base 1D index for this row
        
        for j in range(c_img):
            start_idx = row_start + j
            
            # This is the magic line. It delegates the iteration and addition 
            # down to Python's C-compiled `sum()` function, bypassing standard for-loops.
            val = sum(padded_1d[start_idx + off] * k_val for off, k_val in k_data)
            
            out_row.append(val)
        out_2d.append(out_row)
        
    return out_2d

def local_contrast_normalization(img_list, sigma=10):
    rows = len(img_list)
    cols = len(img_list[0])
    
    local_mean = pure_python_gaussian_blur(img_list, sigma)
    
    centered = [[0.0] * cols for _ in range(rows)]
    centered_sq = [[0.0] * cols for _ in range(rows)]
    
    for r in range(rows):
        for c in range(cols):
            diff = img_list[r][c] - local_mean[r][c]
            centered[r][c] = diff
            centered_sq[r][c] = diff ** 2
            
    local_var_sq = pure_python_gaussian_blur(centered_sq, sigma)
    
    threshold = 0.02
    result = [[0.0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            var = max(0.0, local_var_sq[r][c]) 
            local_var_sqrt = math.sqrt(var)
            
            if local_var_sqrt < threshold:
                result[r][c] = 0.0
            else:
                result[r][c] = centered[r][c] / (local_var_sqrt + 0.1)
            
    return result

# ---------------------------------------------------------
# 2. FEATURE EXTRACTION
# ---------------------------------------------------------

def compute_color_feature(image_path, blocks=4):
    """Computes color features manually mapping RGB to LAB using basic math."""
    # PIL strictly used for I/O
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    pixels = list(img.getdata())
    
    # Restructure into 3D list [row][col][rgb] scaled 0-1
    img_3d = [[[0.0]*3 for _ in range(width)] for _ in range(height)]
    idx = 0
    for r in range(height):
        for c in range(width):
            img_3d[r][c][0] = pixels[idx][0] / 255.0
            img_3d[r][c][1] = pixels[idx][1] / 255.0
            img_3d[r][c][2] = pixels[idx][2] / 255.0
            idx += 1

    # Manual Block Averaging (Resizing to 4x4)
    r_step = height // blocks
    c_step = width // blocks
    small_img = [[[0.0]*3 for _ in range(blocks)] for _ in range(blocks)]
    
    for r in range(blocks):
        for c in range(blocks):
            for k in range(3):
                total = 0.0
                count = 0
                for br in range(r * r_step, (r + 1) * r_step):
                    for bc in range(c * c_step, (c + 1) * c_step):
                        if br < height and bc < width:
                            total += img_3d[br][bc][k]
                            count += 1
                small_img[r][c][k] = total / count if count > 0 else 0.0

    # Manual RGB to LAB Conversion
    lab_out = [[[0.0]*3 for _ in range(blocks)] for _ in range(blocks)]
    for r in range(blocks):
        for c in range(blocks):
            rgb = small_img[r][c]
            
            # RGB to XYZ
            xyz_input = [0.0]*3
            for i in range(3):
                val = rgb[i]
                if val > 0.04045:
                    xyz_input[i] = ((val + 0.055) / 1.055) ** 2.4
                else:
                    xyz_input[i] = val / 12.92
                xyz_input[i] *= 100.0
                
            x = xyz_input[0]*0.4124 + xyz_input[1]*0.3576 + xyz_input[2]*0.1805
            y = xyz_input[0]*0.2126 + xyz_input[1]*0.7152 + xyz_input[2]*0.0722
            z = xyz_input[0]*0.0193 + xyz_input[1]*0.1192 + xyz_input[2]*0.9505
            
            # XYZ to LAB
            xyz_norm = [x / 95.047, y / 100.0, z / 108.883]
            f_xyz = [0.0]*3
            for i in range(3):
                if xyz_norm[i] > 0.008856:
                    f_xyz[i] = xyz_norm[i] ** (1.0 / 3.0) # Pure math cbrt
                else:
                    f_xyz[i] = (7.787 * xyz_norm[i]) + (16.0 / 116.0)
                    
            L = (116 * f_xyz[1]) - 16
            a = 500 * (f_xyz[0] - f_xyz[1])
            b = 200 * (f_xyz[1] - f_xyz[2])
            
            lab_out[r][c] = [L, a, b]
            
    return np.array(lab_out)

def compute_gist(image_path, mask_path=None, scales=5, orientations=6, blocks=4):
    """Computes Gist descriptor purely with Python lists and basic math."""
    # 1. Load Image (PIL for I/O, resizing to 256x256 to save math time)
    img = Image.open(image_path).convert('L').resize((256, 256), Image.Resampling.LANCZOS)
    pixels = list(img.getdata())
    width, height = 256, 256
    
    img_list = [[0.0] * width for _ in range(height)]
    for r in range(height):
        for c in range(width):
            img_list[r][c] = pixels[r * width + c] / 255.0

    gist_weights = [[1.0] * blocks for _ in range(blocks)]
    valid_mask = [[True] * width for _ in range(height)]
    hole_mask = [[False] * width for _ in range(height)]

    # 2. Handle Mask
    if mask_path and os.path.exists(mask_path):
        mask_img = Image.open(mask_path).convert('L').resize((256, 256), Image.Resampling.NEAREST)
        mask_pixels = list(mask_img.getdata())
        
        valid_sum = 0.0
        valid_count = 0
        
        for r in range(height):
            for c in range(width):
                val = mask_pixels[r * width + c] / 255.0
                if val > 0.5:
                    hole_mask[r][c] = True
                    valid_mask[r][c] = False
                else:
                    valid_sum += img_list[r][c]
                    valid_count += 1
                    
        mean_val = valid_sum / valid_count if valid_count > 0 else 0.5
        
        # Fill holes with mean initially
        for r in range(height):
            for c in range(width):
                if hole_mask[r][c]:
                    img_list[r][c] = mean_val

        # Smooth diffusion (replacing scipy gaussian_filter)
        for _ in range(10): # Reduced iterations from 20 for pure-python speed
            blurred_img = pure_python_gaussian_blur(img_list, sigma=2)
            for r in range(height):
                for c in range(width):
                    if hole_mask[r][c]:
                        img_list[r][c] = blurred_img[r][c]

        # Compute block weights
        block_h = height // blocks
        block_w = width // blocks
        for bh in range(blocks):
            for bw in range(blocks):
                valid_pixels = 0
                total_pixels = 0
                for r in range(bh * block_h, (bh + 1) * block_h):
                    for c in range(bw * block_w, (bw + 1) * block_w):
                        if valid_mask[r][c]:
                            valid_pixels += 1
                        total_pixels += 1
                gist_weights[bh][bw] = valid_pixels / total_pixels if total_pixels > 0 else 0.0

    # 3. Normalization and Filtering
    img_list = local_contrast_normalization(img_list)
    
    # Descriptor dimensions: [scales][orientations][blocks][blocks]
    gist_data = [[[[0.0]*blocks for _ in range(blocks)] for _ in range(orientations)] for _ in range(scales)]
    
    # Manual linspace
    sigmas = [4.0 - (3.0 * i / (scales - 1)) if scales > 1 else 4.0 for i in range(scales)]
    lambdas = [s * 2.5 for s in sigmas]
    
    for s in range(scales):
        for o in range(orientations):
            theta = o * math.pi / orientations
            kernel = get_gabor_kernel(sigmas[s], theta, lambdas[s], 0.5)
            
            # Pure Python 2D Convolution
            filtered = pure_convolve2d_optimized(img_list, kernel)
            
            # Energy calculation (absolute value)
            for r in range(height):
                for c in range(width):
                    filtered[r][c] = abs(filtered[r][c])
            
            # 4. Spatial Pooling
            block_h = height // blocks
            block_w = width // blocks
            
            for bh in range(blocks):
                for bw in range(blocks):
                    block_sum = 0.0
                    block_valid_count = 0
                    
                    for r in range(bh * block_h, (bh + 1) * block_h):
                        for c in range(bw * block_w, (bw + 1) * block_w):
                            if mask_path and os.path.exists(mask_path):
                                if valid_mask[r][c]:
                                    block_sum += filtered[r][c]
                                    block_valid_count += 1
                            else:
                                block_sum += filtered[r][c]
                                block_valid_count += 1
                                
                    if block_valid_count > 0:
                        gist_data[s][o][bh][bw] = block_sum / block_valid_count
                    else:
                        gist_data[s][o][bh][bw] = 0.0
                        
    return np.array(gist_data), np.array(gist_weights)

def visualize_gist(gist_data, scales, orientations):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 10), facecolor='#001a33')
    
    gs = fig.add_gridspec(scales + 1, orientations + 1, 
                          width_ratios=[1]*orientations + [0.3],
                          height_ratios=[1]*scales + [0.5])
    
    v_max = np.max(gist_data) if np.max(gist_data) > 0 else 1
    
    for s in range(scales):
        for o in range(orientations):
            ax = fig.add_subplot(gs[s, o])
            # Use 'jet' or 'magma' for high-contrast visibility
            im = ax.imshow(gist_data[s, o], cmap='jet', vmin=0, vmax=v_max)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor('#444444')

    # Add Orientation Icons
    for o in range(orientations):
        ax_icon = fig.add_subplot(gs[scales, o])
        theta = o * np.pi / orientations
        icon = get_gabor_kernel(3, theta, 6, 0.5)
        ax_icon.imshow(icon, cmap='gray')
        ax_icon.axis('off')

    # Labels
    fig.text(0.5, 0.05, 'Edge Orientation', ha='center', color='white', fontsize=18)
    fig.text(0.07, 0.5, 'Frequency (Low to High)', va='center', rotation='vertical', color='white', fontsize=18)
    
    cax = fig.add_subplot(gs[0:scales, -1])
    plt.colorbar(im, cax=cax, label='Activation (Energy)')
    plt.tight_layout(rect=[0.08, 0.08, 0.95, 0.95])
    plt.savefig("gist_visualization.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    #plt.show()
    
def generate_vertical_split(size=128):
    """
    Generates a synthetic image that is half black (0) and half white (1), 
    split vertically in the center.
    """
    # 1. Start with an array of zeros (all black)
    image_array = np.ones((size, size), dtype=np.float32)
    
    # 2. Set the right half to 1 (all white)
    # This selects rows :, and columns from size//2 (64) to the end.
    image_array[:, size // 2:] = 0.0
    
    return image_array

def generate_high_freq_checkerboard(size=128, block_size=2):
    """
    Generates a dense checkerboard pattern. 
    Smaller block_size = higher frequency = more activation in the bottom rows.
    """
    # Create a small 2x2 base pattern
    base = np.array([[0, 1], [1, 0]])
    # Repeat it to fill the image size
    image_array = np.tile(base, (size // (2 * block_size), size // (2 * block_size)))
    # Repeat elements to create blocks of 'block_size'
    image_array = np.repeat(np.repeat(image_array, block_size, axis=0), block_size, axis=1)
    
    return image_array.astype(np.float32)

def test_color_feature():
    print("Running test case: Texture vs Color.")
    size = 128
    # 1. Create two identical structural images (vertical split)
    img_blue = np.zeros((size, size, 3), dtype=np.float32)
    img_red = np.zeros((size, size, 3), dtype=np.float32)
    
    # Blue half / Red half
    img_blue[:, :size//2, 2] = 1.0  
    img_red[:, :size//2, 0] = 1.0   
    
    # White halves
    img_blue[:, size//2:] = 1.0
    img_red[:, size//2:] = 1.0
    
    # Convert to Grayscale for GIST
    img_blue_gray = np.mean(img_blue, axis=-1)
    img_red_gray = np.mean(img_red, axis=-1)

    # 2. Compute GIST
    gist_blue, _ = compute_gist(img_blue_gray)
    gist_red, _ = compute_gist(img_red_gray)

    # 3. Compute Color Features
    color_blue = compute_color_feature(img_blue)
    color_red = compute_color_feature(img_red)

    # Assertions to prove the concept
    gist_diff = np.sum(np.abs(gist_blue - gist_red))
    color_diff = np.sum(np.abs(color_blue - color_red))

    print(f"GIST Descriptor Difference (should be ~0.0): {gist_diff:.4f}")
    print(f"L*a*b* Color Feature Difference (should be large): {color_diff:.4f}")
    
    if gist_diff < 0.1 and color_diff > 10.0:
        print("SUCCESS! The system successfully distinguishes identical textures by color.\n")
    else:
        print("FAILED.\n")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"Processing image: {image_path}...")
        data, weights = compute_gist(image_path)
    else:
        test_color_feature()
        
        # TEST: Generate a pure white image
        print("No image provided. Processing white image...")
        test_img = np.ones((128, 128))
        #test_img = generate_vertical_split(128)
        #test_img = generate_high_freq_checkerboard(128, 2)
        #test_img = np.zeros((128, 128))
        #test_img[64:, :] = 1.0
        # Create the PIL image object first, then call .save() on it
        Image.fromarray((test_img * 255).astype(np.uint8)).save("horiz_split.png")
        data, weights = compute_gist(test_img)
    
    #print(data.shape)
    visualize_gist(data, 5, 6)
