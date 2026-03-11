import tkinter as tk
from tkinter import filedialog
import sys
import math
import os
import subprocess

# Configuration arguments
BRUSH_SIZE = 15
OUTPUT_FILE = "generated_mask.png"

# Declare placeholders for dynamic resolution
WIDTH = 800
HEIGHT = 600
mask = []

def draw(event):
    x, y = event.x, event.y
    
    # Optional feedback on canvas (drawing an oval)
    canvas.create_oval(x - BRUSH_SIZE, y - BRUSH_SIZE,
                       x + BRUSH_SIZE, y + BRUSH_SIZE,
                       fill="white", outline="white")
    
    # Rasterize the circle into our standard python 2D array (the mask)
    for i in range(max(0, y - BRUSH_SIZE), min(HEIGHT, y + BRUSH_SIZE + 1)):
        for j in range(max(0, x - BRUSH_SIZE), min(WIDTH, x + BRUSH_SIZE + 1)):
            # Check if point is inside the circle
            if (j - x)**2 + (i - y)**2 <= BRUSH_SIZE**2:
                mask[i][j] = 255

def finish_drawing(event):
    print(f"Finished drawing! Saving mask to '{OUTPUT_FILE}'...")
    
    # Save as PGM temporarily and convert to PNG using macOS 'sips' command
    temp_pgm = "temp_generated_mask.pgm"
    try:
        with open(temp_pgm, 'wb') as f:
            # Header: P5 (Binary PGM), width, height, max color value (255)
            header = f"P5\n{WIDTH} {HEIGHT}\n255\n"
            f.write(header.encode('ascii'))
            
            # Write the pixel data
            # Flatten the 2D array into a contiguous sequence of bytes
            pixel_data = bytearray()
            for row in mask:
                pixel_data.extend(row)
            f.write(pixel_data)
            
        # Convert to final PNG format
        subprocess.run(["sips", "-s", "format", "png", temp_pgm, "--out", OUTPUT_FILE], capture_output=True)
        print(f"Success! Mask saved as {OUTPUT_FILE}.")
    except Exception as e:
        print(f"Failed to save mask: {e}")
        
    # Cleanup temporary files
    if os.path.exists("temp_ui_background.png"):
        os.remove("temp_ui_background.png")
    if os.path.exists(temp_pgm):
        os.remove(temp_pgm)
        
    # Close GUI
    root.destroy()
    
    print("\\nFinding matching scenes using GIST and Color descriptors...")
    script = f"""
import sys
import cv2
import numpy as np
from match_scenes import find_k_best_matches
from local_context_matching import local_context_matching
from poisson_blending import blend_match

try:
    print("\\nFinding global matches...")
    matches = find_k_best_matches('{image_path}', '{OUTPUT_FILE}', 'beaches', k=12)
    print("\\nTop 12 global matches:")
    
    match_img_bgr_list = []
    valid_matches = []
    
    for rank, (score, path, fname) in enumerate(matches, 1):
        print(f"{{rank}}. {{fname}} (Score: {{score:.4f}})")
        img = cv2.imread(path)
        if img is not None:
            match_img_bgr_list.append(img)
            valid_matches.append(fname)
            
    print("\\nRunning local context matching on top 12 matches...")
    query_img = cv2.imread('{image_path}')
    mask_img = cv2.imread('{OUTPUT_FILE}', cv2.IMREAD_GRAYSCALE)
    if mask_img is None or query_img is None:
        raise ValueError("Could not read query image or generated mask for local context matching.")
        
    # Resize the GUI mask to match the original high-resolution query image
    h_q, w_q = query_img.shape[:2]
    mask_img = cv2.resize(mask_img, (w_q, h_q), interpolation=cv2.INTER_NEAREST)
        
    mask_bool = (mask_img > 127).astype(np.uint8)
    
    local_results = local_context_matching(query_img, mask_bool, match_img_bgr_list[1:]) # Skip the top match which is often the same image
    
    if local_results:
        # Sort by local context score
        local_results.sort(key=lambda x: x['best_score'])
        print("\\nTop Matches after Local Context Verification:")
        for rank, res in enumerate(local_results, 1):
            # +1 because we skipped index 0 in match_img_bgr_list
            actual_idx = res['match_idx'] + 1 
            fname = valid_matches[actual_idx]
            score = res['best_score']
            tex_score = res['texture_ssd']
            scale, ty, tx = res['best_placement']
            print(f"{{rank}}. {{fname}} (Local Score: {{score:.4f}}, Texture SSD: {{tex_score:.4f}}, Placement: scale={{scale}}, y={{ty}}, x={{tx}})")
            
        print("\\nPerforming Poisson Blending on Top 3 matches...")
        for i in range(min(3, len(local_results))):
            res = local_results[i]
            actual_idx = res['match_idx'] + 1
            placement = res['best_placement']
            fname = valid_matches[actual_idx]
            match_img = match_img_bgr_list[actual_idx]
            
            # Poisson blend
            blended_img = blend_match(query_img, mask_bool, match_img, placement)
            
            # Save the result
            out_filename = f"result_rank{{i+1}}_{{fname}}"
            cv2.imwrite(out_filename, blended_img)
            print(f"Saved seamlessly cloned result to: {{out_filename}}")
            
    else:
        print("\\nNo local context matches found (mask might be empty).")
        
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error finding matches: {{e}}")
"""
    subprocess.run([sys.executable, "-c", script])
    sys.exit(0)

def main():
    global root, canvas, WIDTH, HEIGHT, mask, bg_img, image_path
    root = tk.Tk()
    root.withdraw() # Hide initially so we just show the file dialog
    
    # Let the user pick any image file interactively
    image_path = filedialog.askopenfilename(
        title="Select an Image to Paint Over",
        initialdir=os.path.join(os.getcwd(), "beaches"), # Default to beaches folder if it exists
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.gif *.bmp"), ("All Files", "*.*")]
    )
    
    if not image_path:
        print("No image selected! Exiting.")
        sys.exit(0)
        
    root.deiconify() # Restore the main UI window
    root.title(f"Paint Mask: Drag to draw, press 'q' to save & exit | {os.path.basename(image_path)}")
    
    print(f"Loading image: {image_path}")
    # Tkinter natively supports PNG/GIF. We can use macOS 'sips' command to seamlessly convert jpg for the UI
    # We also use 'sips' with -Z to ensure the longest edge is at most 1000px so it fits on most screens
    temp_png = "temp_ui_background.png"
    subprocess.run(["sips", "-Z", "1000", "-s", "format", "png", image_path, "--out", temp_png], capture_output=True)
    img_to_load = temp_png
        
    try:
        bg_img = tk.PhotoImage(file=img_to_load)
        WIDTH = bg_img.width()
        HEIGHT = bg_img.height()
    except Exception as e:
        print(f"Error initializing background image: {e}")
        sys.exit(1)
        
    # Initialize mask data array with exact image dimensions
    mask = [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]
    
    # Create Canvas and draw the background image
    canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="black", cursor="crosshair")
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, image=bg_img, anchor="nw")
    
    # Bind left-click and mouse dragging to the draw function
    canvas.bind("<B1-Motion>", draw)
    canvas.bind("<Button-1>", draw)
    
    # Bind the 'q' key to finish the drawing and save the file
    root.bind("q", finish_drawing)
    root.bind("Q", finish_drawing)
    
    print(f"Application started.")
    print(" - Click and Drag mouse to draw the mask.")
    print(" - Press the 'q' key on your keyboard to finish and save.")
    
    root.mainloop()

if __name__ == "__main__":
    main()