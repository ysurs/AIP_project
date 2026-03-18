import os
from PIL import Image
from tqdm import tqdm

def prepare_research_database(src_root, dest_folder, target_dim=1024):
    """
    Flattens dataset:
    data/city/images/*.jpg → dest_folder/*.jpg
    """

    os.makedirs(dest_folder, exist_ok=True)

    all_files = []

    # Collect all image paths
    for city in os.listdir(src_root):
        images_path = os.path.join(src_root, city)

        if not os.path.isdir(images_path):
            continue

        for f in os.listdir(images_path):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(images_path, f)
                all_files.append((city, full_path, f))

    print(f"Processing {len(all_files)} images...")

    skipped_too_small = 0
    processed_count = 0

    for city, src_path, filename in tqdm(all_files):
        # Prefix filename to avoid collisions
        new_filename = f"{city}_{filename}"
        dest_path = os.path.join(dest_folder, new_filename)

        if os.path.exists(dest_path):
            continue

        try:
            with Image.open(src_path) as img:
                w, h = img.size
                max_edge = max(w, h)

                if max_edge < target_dim:
                    skipped_too_small += 1
                    continue

                ratio = target_dim / float(max_edge)
                new_size = (int(w * ratio), int(h * ratio))

                resized_img = img.resize(new_size, Image.Resampling.LANCZOS)

                if resized_img.mode != "RGB":
                    resized_img = resized_img.convert("RGB")

                resized_img.save(dest_path, "JPEG", quality=95)
                processed_count += 1

        except Exception as e:
            print(f"Error processing {src_path}: {e}")

    print(f"\n--- Database Preparation Complete ---")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped (too small): {skipped_too_small}")
    
prepare_research_database("data/images", "skyline_1024")