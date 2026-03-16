import os
from PIL import Image
from tqdm import tqdm # Useful for tracking progress in large databases

def prepare_research_database(src_folder, dest_folder, target_dim=1024):
    """
    Standardizes a database for Hayes & Efros Scene Completion.
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Processing {len(files)} images...")

    skipped_too_small = 0
    processed_count = 0

    for filename in tqdm(files):
        src_path = os.path.join(src_folder, filename)
        dest_path = os.path.join(dest_folder, filename)
        
        # Skip if already exists in destination
        if os.path.exists(dest_path):
            continue
            
        try:
            with Image.open(src_path) as img:
                w, h = img.size
                max_edge = max(w, h)
                
                # REJECTION CRITERIA:
                # If the image is smaller than our target, it lacks the 
                # resolution required for high-quality local context matching.
                if max_edge < target_dim:
                    skipped_too_small += 1
                    continue
                
                # RESIZING LOGIC:
                # Calculate ratio to ensure the longest edge is exactly target_dim
                ratio = target_dim / float(max_edge)
                new_size = (int(w * ratio), int(h * ratio))
                
                # Use LANCZOS for high-quality downsampling (prevents aliasing)
                resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Convert to RGB to ensure consistency (removes Alpha channels if any)
                if resized_img.mode != "RGB":
                    resized_img = resized_img.convert("RGB")
                
                resized_img.save(dest_path, "JPEG", quality=95)
                processed_count += 1
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"\n--- Database Preparation Complete ---")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped (too small): {skipped_too_small}")

# Run this once before your main script
prepare_research_database("beaches", "beaches_1024")