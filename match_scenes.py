import os
import argparse
import numpy as np
import pickle
from PIL import Image
import cv2
from feature_extraction import compute_gist, compute_color_feature,visualize_gist


def resize_longest_side(img, max_dim=1024):
    """
    Resizes an image using pure standard basic arithmetic (Nearest-Neighbor).
    Assumes `img` is accessible via indexing (e.g., list of lists, or numpy array).
    """
    h = len(img)
    w = len(img[0]) if h > 0 else 0
    longest = max(h, w)

    if longest <= max_dim:
        return img

    scale = max_dim / longest
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Detect if image is colored (3 channels) or grayscale
    is_color = False
    if h > 0 and w > 0:
        first_pixel = img[0][0]
        if isinstance(first_pixel, (list, tuple)) or (hasattr(first_pixel, '__len__') and not isinstance(first_pixel, str)):
            is_color = True

    new_img = []
    for y in range(new_h):
        row = []
        orig_y = min(int(y / scale), h - 1)
        for x in range(new_w):
            orig_x = min(int(x / scale), w - 1)
            
            if is_color:
                # Reconstruct the pixel channels via list comprehension
                pixel = img[orig_y][orig_x]
                row.append([p for p in pixel])
            else:
                row.append(img[orig_y][orig_x])
        new_img.append(row)

    return new_img

def load_or_compute_db_features(db_dir, cache_file="db_features.pkl"):
    if os.path.exists(cache_file):
        print(f"Loading database features from {cache_file}...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("Computing features for database images...")
    db_features = []

    for filename in os.listdir(db_dir):
        if not filename.lower().endswith(('.png','.jpg','.jpeg','.bmp')):
            continue

        filepath = os.path.join(db_dir, filename)

        try:
            # The database images have no mask, so weights are all 1
            gist_data, _ = compute_gist(filepath)
            color_data = compute_color_feature(filepath)
            
            db_features.append({
                'filename': filename,
                'filepath': filepath,
                'gist': gist_data,
                'color': color_data
            })

            print(f"Processed {filename}")

        except Exception as e:
            print(f"Failed to process {filename}: {e}")
            
    with open(cache_file, "wb") as f:
        pickle.dump(db_features, f)
        
    return db_features


# --- Pure math helper functions for distance calculation ---

def calculate_gist_ssd(db_gist, q_gist, q_weights):
    """Calculates weighted Sum of Squared Differences for 4D GIST features."""
    ssd = 0.0
    scales = len(q_gist)
    
    for s in range(scales):
        oris = len(q_gist[s])
        for o in range(oris):
            by = len(q_gist[s][o])
            for y in range(by):
                bx = len(q_gist[s][o][y])
                for x in range(bx):
                    diff = db_gist[s][o][y][x] - q_gist[s][o][y][x]
                    
                    # Manual broadcasting: Check if weights are 4D or 2D
                    try:
                        w = q_weights[s][o][y][x]
                    except (TypeError, IndexError):
                        try:
                            w = q_weights[y][x]
                        except (TypeError, IndexError):
                            w = 1.0  # Fallback
                            
                    ssd += w * (diff * diff)
    return ssd

def calculate_color_ssd(db_color, q_color, q_weights):
    """Calculates weighted Sum of Squared Differences for 3D Color features."""
    ssd = 0.0
    by = len(q_color)
    
    for y in range(by):
        bx = len(q_color[y])
        for x in range(bx):
            channels = len(q_color[y][x])
            
            # Manual broadcasting: Spatial weights applied across all channels
            try:
                w = q_weights[y][x]
            except (TypeError, IndexError):
                try:
                    w = q_weights[0][0][y][x] # Extract from 4D if necessary
                except (TypeError, IndexError):
                    w = 1.0 # Fallback
            
            for c in range(channels):
                diff = db_color[y][x][c] - q_color[y][x][c]
                ssd += w * (diff * diff)
    return ssd


def find_k_best_matches(query_image, mask_image, db_dir, k=10):
    # 1. Compute query features
    print("Computing query features...")
    q_gist, q_weights = compute_gist(query_image, mask_image)
    
    s_dim = len(q_gist)
    o_dim = len(q_gist[0]) if s_dim > 0 else 0
    y_dim = len(q_gist[0][0]) if o_dim > 0 else 0
    x_dim = len(q_gist[0][0][0]) if y_dim > 0 else 0
    print(f"Query GIST shape: ({s_dim}, {o_dim}, {y_dim}, {x_dim})")
    
    q_color = compute_color_feature(query_image)
    
    visualize_gist(q_gist, 5, 6)
    
    # 2. Load ALL database features
    # Derive a cache filename from the DB folder so skyline_1024 and skyline_tiny
    # each get their own cache file and never overwrite each other.
    cache_file = f"db_features_{os.path.basename(db_dir.rstrip('/'))}.pkl"
    db_features = load_or_compute_db_features(db_dir, cache_file=cache_file)
    n_images = len(db_features)
    
    if n_images == 0:
        return []

    # 3. Calculate distances iteratively (No Numpy Arrays)
    gist_ssds = []
    color_ssds = []
    
    for i in range(n_images):
        g_ssd = calculate_gist_ssd(db_features[i]['gist'], q_gist, q_weights)
        c_ssd = calculate_color_ssd(db_features[i]['color'], q_color, q_weights)
        
        gist_ssds.append(g_ssd)
        color_ssds.append(c_ssd)
        
    # 4. Apply the "Twice as Much" Weighting Rule
    # Calculate Mean manually
    sum_gist = 0.0
    sum_color = 0.0
    for i in range(n_images):
        sum_gist += gist_ssds[i]
        sum_color += color_ssds[i]
        
    mean_gist_dist = sum_gist / n_images
    mean_color_dist = sum_color / n_images
    
    # Prevent divide by zero
    if mean_color_dist == 0: 
        mean_color_dist = 1.0
    
    # Calculate lambda such that: mean(gist) = 2 * (lambda * mean(color))
    scaling_factor = mean_gist_dist / (2 * mean_color_dist)
    
    # Calculate total scores
    total_scores = []
    for i in range(n_images):
        score = gist_ssds[i] + (scaling_factor * color_ssds[i])
        total_scores.append(score)

    # 5. Find Top K
    # Create a list of tuples: (score, original_index)
    indexed_scores = [(total_scores[i], i) for i in range(n_images)]
    
    # Sort the list based on the score (the first element of the tuple)
    indexed_scores.sort(key=lambda item: item[0])
    
    k_actual = min(k, n_images)
    
    results = []
    for i in range(k_actual):
        score = indexed_scores[i][0]
        idx = indexed_scores[i][1]
        results.append((score, db_features[idx]['filepath'], db_features[idx]['filename']))
        
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find k best matching scenes.")
    parser.add_argument("--query", required=True, help="Path to query image")
    parser.add_argument("--mask", required=True, help="Path to mask image. (Holes are typically white > 0.5)")
    parser.add_argument("--db", default="beaches", help="Path to image database folder")
    parser.add_argument("--k", type=int, default=10, help="Number of best matches to return")
    
    args = parser.parse_args()
    
    matches = find_k_best_matches(args.query, args.mask, args.db, k=args.k)
    
    print(f"\nTop {args.k} matches:")
    for rank, (score, path, fname) in enumerate(matches, 1):
        print(f"{rank}. {fname} (Score: {score:.4f})")
