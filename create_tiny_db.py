"""
create_tiny_db.py  –  EF3 helper (pure Python, PIL only)

Creates a tiny-resolution database from skyline_1024/ for EF3 pipeline.
Target: 256px longest edge → enables exact 4x SR upsampling back to ~1024px.

Usage:
    python create_tiny_db.py                            # default: skyline_1024 → skyline_tiny
    python create_tiny_db.py --src skyline_1024 --dst skyline_tiny --dim 256
"""

import os
import argparse
from PIL import Image


def create_tiny_db(src_folder, dest_folder, target_dim=256):
    """
    Downscale every image in src_folder to target_dim (longest edge) and save to dest_folder.
    Uses only PIL – no third-party libraries.
    """
    if not os.path.isdir(src_folder):
        print(f"ERROR: Source folder '{src_folder}' not found.")
        return

    os.makedirs(dest_folder, exist_ok=True)

    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(src_folder) if f.lower().endswith(valid_ext)]

    if not files:
        print(f"No image files found in '{src_folder}'.")
        return

    print(f"Creating tiny DB ({target_dim}px) from {len(files)} images in '{src_folder}'...")
    print(f"Output → '{dest_folder}/'")

    skipped = 0
    processed = 0

    for i, filename in enumerate(sorted(files)):
        src_path = os.path.join(src_folder, filename)
        dest_path = os.path.join(dest_folder, filename)

        if os.path.exists(dest_path):
            skipped += 1
            continue

        try:
            with Image.open(src_path) as img:
                w, h = img.size
                longest = max(w, h)

                # Scale so that the longest edge becomes target_dim
                ratio = target_dim / float(longest)
                new_w = max(1, int(round(w * ratio)))
                new_h = max(1, int(round(h * ratio)))

                resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                if resized.mode != 'RGB':
                    resized = resized.convert('RGB')

                # Save as JPEG to keep file sizes small
                out_name = os.path.splitext(filename)[0] + '.jpg'
                dest_path = os.path.join(dest_folder, out_name)
                resized.save(dest_path, 'JPEG', quality=90)
                processed += 1

        except Exception as e:
            print(f"  ERROR processing {filename}: {e}")

        if (i + 1) % 25 == 0 or (i + 1) == len(files):
            print(f"  {i + 1}/{len(files)} done  ({processed} saved, {skipped} skipped)")

    print(f"\nDone. {processed} images saved to '{dest_folder}/' at {target_dim}px.")
    if skipped:
        print(f"({skipped} already existed and were skipped.)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a tiny-resolution DB for EF3.")
    parser.add_argument('--src', default='skyline_1024', help='Source DB folder (default: skyline_1024)')
    parser.add_argument('--dst', default='skyline_tiny', help='Output tiny DB folder (default: skyline_tiny)')
    parser.add_argument('--dim', type=int, default=256, help='Target longest edge in pixels (default: 256)')
    args = parser.parse_args()

    create_tiny_db(args.src, args.dst, args.dim)
