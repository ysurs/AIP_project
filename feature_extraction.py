import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import math

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
    """Returns a 1D normalized Gaussian kernel for separable blur."""
    radius = int(math.ceil(3 * sigma))
    kernel = []
    for x in range(-radius, radius + 1):
        kernel.append(math.exp(-(x**2) / (2 * sigma**2)))
    total = sum(kernel)
    return [k / total for k in kernel]

def pure_python_gaussian_blur(img_list, sigma):
    """Applies separable Gaussian blur (horizontal then vertical) with mirror padding."""
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

def pure_convolve2d_optimized(image_2d, kernel_2d):
    """2D convolution using pre-padded flat array and precomputed kernel offsets to minimize per-pixel overhead."""
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
            
            val = sum(padded_1d[start_idx + off] * k_val for off, k_val in k_data)
            
            out_row.append(val)
        out_2d.append(out_row)
        
    return out_2d

def local_contrast_normalization(img_list, sigma=10):
    """Subtracts local mean and divides by local std dev; zeros out near-uniform regions below threshold."""
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
    if isinstance(image_path, str):
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
    else:
        # Input is a list or numpy array [rows][cols][3]
        try:
            raw = image_path.tolist()
        except AttributeError:
            raw = image_path
        height = len(raw)
        width = len(raw[0])
        img_3d = [[[float(raw[r][c][k]) for k in range(3)] for c in range(width)] for r in range(height)]

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
    # 1. Load Image
    if isinstance(image_path, str):
        img = Image.open(image_path).convert('L').resize((256, 256), Image.Resampling.LANCZOS)
        pixels = list(img.getdata())
        width, height = 256, 256
        img_list = [[0.0] * width for _ in range(height)]
        for r in range(height):
            for c in range(width):
                img_list[r][c] = pixels[r * width + c] / 255.0
    else:
        # Input is a 2D list or numpy array
        try:
            raw = image_path.tolist()
        except AttributeError:
            raw = image_path
        height = len(raw)
        width = len(raw[0])
        img_list = [[float(raw[r][c]) for c in range(width)] for r in range(height)]

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

        # Smooth diffusion
        for _ in range(20): # Reduced iterations from 20 for pure-python speed
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
    """Renders a grid of GIST block-response maps (scale x orientation) and saves to gist_visualization.png."""
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 10), facecolor='#001a33')
    
    gs = fig.add_gridspec(scales + 1, orientations + 1, 
                          width_ratios=[1]*orientations + [0.3],
                          height_ratios=[1]*scales + [0.5])
    
    v_max = max(gist_data.flat)
    if v_max <= 0:
        v_max = 1
    
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
        theta = o * math.pi / orientations
        icon = get_gabor_kernel(3, theta, 6, 0.5)
        ax_icon.imshow(icon, cmap='gray')
        ax_icon.axis('off')

    # Labels
    fig.text(0.5, 0.05, 'Edge Orientation', ha='center', color='white', fontsize=18)
    fig.text(0.07, 0.5, 'Frequency (High to Low)', va='center', rotation='vertical', color='white', fontsize=18)
    
    cax = fig.add_subplot(gs[0:scales, -1])
    plt.colorbar(im, cax=cax, label='Activation (Energy)')
    plt.tight_layout(rect=[0.08, 0.08, 0.95, 0.95])
    plt.savefig("gist_visualization.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    
def generate_vertical_split(size=128):
    """
    Generates a synthetic image that is half black (0) and half white (1),
    split vertically in the center.
    """
    half = size // 2
    image_list = [[1.0 if c < half else 0.0 for c in range(size)] for r in range(size)]
    return np.array(image_list, dtype='float32')

def generate_high_freq_checkerboard(size=128, block_size=2):
    """
    Generates a dense checkerboard pattern.
    Smaller block_size = higher frequency = more activation in the bottom rows.
    """
    image_list = [
        [float((r // block_size + c // block_size) % 2) for c in range(size)]
        for r in range(size)
    ]
    return np.array(image_list, dtype='float32')

def test_color_feature():
    """Verifies that GIST is texture-only (color-blind) while compute_color_feature captures color differences."""
    print("Running test case: Texture vs Color.")
    size = 128
    half = size // 2

    # 1. Create two identical structural images (vertical split) as 3D lists [row][col][rgb]
    img_blue = [[[0.0, 0.0, 0.0] for _ in range(size)] for _ in range(size)]
    img_red  = [[[0.0, 0.0, 0.0] for _ in range(size)] for _ in range(size)]

    for r in range(size):
        for c in range(half):
            img_blue[r][c][2] = 1.0   # Blue channel
            img_red[r][c][0] = 1.0    # Red channel
        for c in range(half, size):
            img_blue[r][c] = [1.0, 1.0, 1.0]
            img_red[r][c]  = [1.0, 1.0, 1.0]

    # Convert to Grayscale for GIST (average of 3 channels)
    img_blue_gray = [[(img_blue[r][c][0] + img_blue[r][c][1] + img_blue[r][c][2]) / 3.0
                      for c in range(size)] for r in range(size)]
    img_red_gray  = [[(img_red[r][c][0]  + img_red[r][c][1]  + img_red[r][c][2])  / 3.0
                      for c in range(size)] for r in range(size)]

    # 2. Compute GIST
    gist_blue, _ = compute_gist(img_blue_gray)
    gist_red, _  = compute_gist(img_red_gray)

    # 3. Compute Color Features
    color_blue = compute_color_feature(img_blue)
    color_red  = compute_color_feature(img_red)

    # Assertions to prove the concept
    gist_diff  = sum(abs(a - b) for a, b in zip(gist_blue.flat,  gist_red.flat))
    color_diff = sum(abs(a - b) for a, b in zip(color_blue.flat, color_red.flat))

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
        test_img = [[1.0] * 128 for _ in range(128)]
        #test_img = generate_vertical_split(128)
        #test_img = generate_high_freq_checkerboard(128, 2)
        #test_img = np.zeros((128, 128))
        #test_img[64:, :] = 1.0
        # Create the PIL image object first, then call .save() on it
        img_bytes = bytes(int(test_img[r][c] * 255) for r in range(128) for c in range(128))
        Image.frombytes('L', (128, 128), img_bytes).save("horiz_split.png")
        data, weights = compute_gist(test_img)
    
    #print(data.shape)
    visualize_gist(data, 5, 6)
