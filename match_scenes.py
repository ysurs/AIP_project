import os
import argparse
import numpy as np
import pickle
from PIL import Image

from test_image_creation import compute_gist, compute_color_feature,visualize_gist

def load_or_compute_db_features(db_dir, cache_file="db_features.pkl"):
    """Loads precomputed features if possible, else computes them."""
    if os.path.exists(cache_file):
        print(f"Loading database features from {cache_file}...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("Computing features for database images...")
    db_features = []
    
    for filename in os.listdir(db_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
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
            
    # Save to cache
    with open(cache_file, "wb") as f:
        pickle.dump(db_features, f)
        
    return db_features


def find_k_best_matches(query_image, mask_image, db_dir, k=10):
    # 1. Compute query features
    print("Computing query features...")
    q_gist, q_weights = compute_gist(query_image, mask_image)
    q_color = compute_color_feature(query_image)
    
    visualize_gist(q_gist,5,6)
    
    # 2. Load ALL database features into memory as giant arrays
    db_features = load_or_compute_db_features(db_dir)
    
    # Convert list of dicts to Arrays for Vectorization
    # Shape: (N_images, scales, orientations, blocks, blocks)
    db_gist_arr = np.array([item['gist'] for item in db_features]) 
    # Shape: (N_images, blocks, blocks, 3)
    db_color_arr = np.array([item['color'] for item in db_features]) 
    
    # 3. VECTORIZED Distance Calculation
    
    # GIST Distance
    # (N, ...) - (query_shape) -> Broadcasting works automatically
    gist_diffs = db_gist_arr - q_gist 
    # Weight and Sum: Result is shape (N_images,)
    weighted_gist_ssd = np.sum(q_weights * (gist_diffs ** 2), axis=(1,2,3,4)) 

    # Color Distance
    color_diffs = db_color_arr - q_color
    # Weight needs expansion to match color channels (blocks, blocks, 1) or (blocks, blocks, 3)
    weights_expanded = np.expand_dims(q_weights, axis=-1)
    weighted_color_ssd = np.sum(weights_expanded * (color_diffs ** 2), axis=(1,2,3))

    # 4. Apply the "Twice as Much" Weighting Rule
    # We calculate the mean distance of the batch to normalize
    mean_gist_dist = np.mean(weighted_gist_ssd)
    mean_color_dist = np.mean(weighted_color_ssd)
    
    # Prevent divide by zero
    if mean_color_dist == 0: mean_color_dist = 1.0
    
    # Calculate lambda such that: mean(gist) = 2 * (lambda * mean(color))
    scaling_factor = mean_gist_dist / (2 * mean_color_dist)
    
    total_scores = weighted_gist_ssd + (scaling_factor * weighted_color_ssd)

    # 5. Find Top K (using argpartition for speed)
    # Handle the case where k > number of elements
    k_actual = min(k, len(db_features) - 1)
    if k_actual <= 0:
        best_indices = np.argsort(total_scores)
    else:
        best_indices = np.argpartition(total_scores, k_actual)[:k]
        # Sort the top k
        best_indices = best_indices[np.argsort(total_scores[best_indices])]
    
    results = []
    for idx in best_indices:
        results.append((total_scores[idx], db_features[idx]['filepath'], db_features[idx]['filename']))
        
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
