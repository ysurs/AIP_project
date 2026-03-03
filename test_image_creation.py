import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image
import os
from scipy.ndimage import gaussian_filter
import math


def get_gabor_kernel(sigma, theta, Lambda, gamma):
    """Generates a zero-mean Gabor kernel (returns numpy ndarray)."""
    
    half_size = int(3 * sigma)
    size = 2 * half_size + 1
    
    kernel = np.zeros((size, size), dtype=float)
    
    for i, y in enumerate(range(-half_size, half_size + 1)):
        for j, x in enumerate(range(-half_size, half_size + 1)):
            
            # Rotate coordinates
            x_theta = x * math.cos(theta) + y * math.sin(theta)
            y_theta = -x * math.sin(theta) + y * math.cos(theta)
            
            # Gaussian envelope
            gaussian = math.exp(
                -0.5 * (
                    (x_theta ** 2) / (sigma ** 2) +
                    (gamma ** 2 * y_theta ** 2) / (sigma ** 2)
                )
            )
            
            # Cosine carrier
            sinusoid = math.cos(2 * math.pi * x_theta / Lambda)
            
            kernel[i, j] = gaussian * sinusoid
    
    # Subtract mean (zero-mean property)
    kernel -= kernel.mean()
    
    return np.array(kernel)

def local_contrast_normalization(img, sigma=10):
    """Normalizes lighting and enhances local textures."""
    local_mean = gaussian_filter(img, sigma)
    centered = img - local_mean
    # Epsilon (0.01) prevents division by zero in perfectly flat areas
    local_var = np.sqrt(gaussian_filter(centered**2, sigma))
    return centered / (local_var + 0.01)

def compute_gist(image_path, mask_path=None, scales=5, orientations=6, blocks=4):
    # 1. Load and Preprocess
    # If path is a string, open it; if it's already an array, use it directly
    if isinstance(image_path, str):
        img = Image.open(image_path).convert('L')
        img = img.resize((128, 128), Image.Resampling.LANCZOS)
        img_array = np.array(img) / 255.0
    else:
        img_array = image_path 

    # Apply mask if provided
    if mask_path and os.path.exists(mask_path):
        mask_img = Image.open(mask_path).convert('L')
        mask_img = mask_img.resize((128, 128), Image.Resampling.NEAREST)
        mask_array = np.array(mask_img) / 255.0
        
        # Calculate mean of the unmasked area
        unmasked_pixels = img_array[mask_array < 0.5]
        mean_val = np.mean(unmasked_pixels) if len(unmasked_pixels) > 0 else 0.5
        
        # Fill the hole with the mean value to prevent false edge responses
        img_array[mask_array > 0.5] = mean_val

    img_array = local_contrast_normalization(img_array)
    
    gist_data = np.zeros((scales, orientations, blocks, blocks))
    
    # Define frequency range
    sigmas = np.linspace(4.0, 1.0, scales) 
    lambdas = sigmas * 2.5 
    
    for s in range(scales):
        for o in range(orientations):
            theta = o * np.pi / orientations
            kernel = get_gabor_kernel(sigmas[s], theta, lambdas[s], 0.5)
            
            # FIX: boundary='symm' prevents the "frame" artifact on white images
            filtered = convolve2d(img_array, kernel, mode='same', boundary='symm')
            energy = np.abs(filtered)
            
            # 3. Spatial Pooling
            h, w = energy.shape
            for bh in range(blocks):
                for bw in range(blocks):
                    block = energy[bh*h//blocks:(bh+1)*h//blocks, 
                                   bw*w//blocks:(bw+1)*w//blocks]
                    gist_data[s, o, bh, bw] = np.mean(block)
                    
    return gist_data

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
    plt.show()
    
def generate_vertical_split(size=128):
    """
    Generates a synthetic image that is half black (0) and half white (1), 
    split vertically in the center.
    """
    # 1. Start with an array of zeros (all black)
    image_array = np.zeros((size, size), dtype=np.float32)
    
    # 2. Set the right half to 1 (all white)
    # This selects rows :, and columns from size//2 (64) to the end.
    image_array[:, size // 2:] = 1.0
    
    return image_array

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"Processing image: {image_path}...")
        data = compute_gist(image_path)
    else:
        # TEST: Generate a pure white image
        print("No image provided. Processing white image...")
        #white_img = np.ones((128, 128))
        test_img = generate_vertical_split(128)
        #test_img = np.zeros((128, 128))
        #test_img[64:, :] = 1.0
        # Create the PIL image object first, then call .save() on it
        Image.fromarray((test_img * 255).astype(np.uint8)).save("horiz_split.png")
        data = compute_gist(test_img)
    
    visualize_gist(data, 5, 6)
