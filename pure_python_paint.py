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
def color_transfer(source, target):
    # Convert to LAB color space
    s_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    t_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Get mean and std dev for each channel
    s_mean, s_std = cv2.meanStdDev(s_lab)
    t_mean, t_std = cv2.meanStdDev(t_lab)

    # Shift the source colors to match the target's distribution
    s_lab -= s_mean.flatten()
    s_lab = (s_lab * (t_std.flatten() / (s_std.flatten() + 1e-5))) + t_mean.flatten()

    # Clip values and convert back to BGR
    s_lab = np.clip(s_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(s_lab, cv2.COLOR_LAB2BGR)

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
        
        # # 2. Get the Best Match
        # for i in range(4):
        #     best_match = local_results[i]
        #     best_img_idx = best_match['match_idx']
        #     best_scale, min_x, min_y = best_match['placement']

        #     print(f"Top match found: Image Index {best_img_idx} with score {best_match['score']:.4f}")

        #     # 3. Reload Query & Masks to get the Bounding Box 
        #     q_bgr = cv2.imread("image_1024.png")
        #     mask_img = cv2.imread("mask_1024.png", cv2.IMREAD_GRAYSCALE)

        #     # Recreate the context mask logic to find the exact same y1, x1, y2, x2
        #     mask_bool = mask_img > 127
        #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (161, 161)) 
        #     dilated_hole = cv2.dilate(mask_img, kernel)
        #     context_mask = ((dilated_hole > 0) & (~mask_bool)).astype(np.uint8) * 255

        #     coords = np.argwhere(context_mask > 0)
        #     orig_y1, orig_x1 = coords[:,0].min(), coords[:,1].min()
        #     orig_y2, orig_x2 = coords[:,0].max()+1, coords[:,1].max()+1

        #     # 4. Safely Pad the Query
        #     pad = 100
        #     y1 = max(0, orig_y1 - pad)
        #     x1 = max(0, orig_x1 - pad)
        #     y2 = min(q_bgr.shape[0], orig_y2 + pad)
        #     x2 = min(q_bgr.shape[1], orig_x2 + pad)

        #     pad_top = orig_y1 - y1
        #     pad_left = orig_x1 - x1

        #     box_h, box_w = y2 - y1, x2 - x1

        #     q_crop = q_bgr[y1:y2, x1:x2]
        #     hole_mask_crop = mask_img[y1:y2, x1:x2]
        #     context_mask_crop = context_mask[y1:y2, x1:x2]

        #     # 5. Extract the Matched Patch (BULLETPROOF VERSION)
        #     best_img = match_img_bgr_list[best_img_idx]
        #     sh, sw = int(best_img.shape[0] * best_scale), int(best_img.shape[1] * best_scale)
        #     best_img_scaled = cv2.resize(best_img, (sw, sh))

        #     # NEW: Add a mirrored border to the database image so it NEVER runs out of bounds!
        #     best_img_padded = cv2.copyMakeBorder(best_img_scaled, pad, pad, pad, pad, cv2.BORDER_REFLECT)

        #     # Calculate the exact crop coordinates inside the padded image
        #     start_y = int(min_y) + pad - pad_top
        #     start_x = int(min_x) + pad - pad_left
            
        #     m_crop = best_img_padded[start_y : start_y + box_h, start_x : start_x + box_w]
            
        #     # Remove the "if m_crop.shape != q_crop.shape" safety net. We don't need it anymore!
            
        #     # ------------------------------------------------------------------
        #     # MANDATORY: Gentle Color Transfer so Graph Cut finds a jagged seam
        #     # ------------------------------------------------------------------
        #     m_crop = color_transfer(m_crop, q_crop)
            
        #     # 6. Find Optimal Seam (Graph Cut)
        #     print("Calculating Graph Cut seam...")
        #     seam_mask = find_optimal_seam(q_crop, m_crop, hole_mask_crop, context_mask_crop)

        #     # # 7. Poisson Blending
        #     # print("Applying Poisson Blending...")
        #     # # cv2.seamlessClone requires the center (x,y) of where the patch will be placed in the target image
        #     # center_x = int(x1 + (box_w / 2))
        #     # center_y = int(y1 + (box_h / 2))
        #     # center_point = (center_x, center_y)

        #     # # Apply standard Poisson blending
        #     # final_result = cv2.seamlessClone(
        #     #     src=m_crop, 
        #     #     dst=q_bgr, 
        #     #     mask=seam_mask, 
        #     #     p=center_point, 
        #     #     flags=cv2.NORMAL_CLONE
        #     # )
            
        #     # DEBUG: Save this to your disk to prove the graph cut is working!
        #     # You should see a jagged white shape, not a perfect rectangle.
        #     cv2.imwrite("debug_seam_mask.png", seam_mask)

        #     # 7. Poisson Blending
        #     print("Applying Poisson Blending...")

        #     # THE FIX: Find the bounding box of the actual graph cut mask
        #     x, y, w_mask, h_mask = cv2.boundingRect(seam_mask)

        #     # Calculate the center of the *mask's bounding box* relative to the whole destination image
        #     center_x = int(x1 + x + (w_mask / 2))
        #     center_y = int(y1 + y + (h_mask / 2))
        #     center_point = (center_x, center_y)

        #     # Apply standard Poisson blending
        #     final_result = cv2.seamlessClone(
        #         src=m_crop, 
        #         dst=q_bgr, 
        #         mask=seam_mask, 
        #         p=center_point, 
        #         flags=cv2.NORMAL_CLONE
        #     )

        #     # 8. Save/Display the result
        #     cv2.imwrite(f"final_completed_image_{i}.png", final_result)
        #     print("Pipeline complete! Saved as final_completed_image.png")
        
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

                mask_bool = mask_img > 127
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (161, 161)) 
                dilated_hole = cv2.dilate(mask_img, kernel)
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
                best_img_scaled = cv2.resize(best_img, (sw, sh))
                best_img_padded = cv2.copyMakeBorder(best_img_scaled, pad, pad, pad, pad, cv2.BORDER_REFLECT)

                start_y = int(min_y) + pad - pad_top
                start_x = int(min_x) + pad - pad_left
                m_crop = best_img_padded[start_y : start_y + box_h, start_x : start_x + box_w]
                
                # m_crop = color_transfer(m_crop, q_crop) # Uncomment if using color_transfer
                
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
            x, y, w_mask, h_mask = cv2.boundingRect(winner['seam_mask'])
            center_x = int(winner['x1'] + x + (w_mask / 2))
            center_y = int(winner['y1'] + y + (h_mask / 2))
            
            final_result = cv2.seamlessClone(
                src=winner['m_crop'], dst=q_bgr, mask=winner['seam_mask'], 
                p=(center_x, center_y), flags=cv2.NORMAL_CLONE
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
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (161, 161)) 
                dilated_hole = cv2.dilate(mask_img, kernel)
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
                best_img_scaled = cv2.resize(best_img, (sw, sh))
                best_img_padded = cv2.copyMakeBorder(best_img_scaled, pad, pad, pad, pad, cv2.BORDER_REFLECT)

                start_y = int(min_y) + pad - pad_top
                start_x = int(min_x) + pad - pad_left
                m_crop = best_img_padded[start_y : start_y + box_h, start_x : start_x + box_w]
                
                # Unpack both values, but we only care about the mask in the base pipeline
                seam_mask= find_optimal_seam(q_crop, m_crop, hole_mask_crop, context_mask_crop,first_component=False)

                x, y, w_mask, h_mask = cv2.boundingRect(seam_mask)
                center_x = int(x1 + x + (w_mask / 2))
                center_y = int(y1 + y + (h_mask / 2))

                final_result = cv2.seamlessClone(
                    src=m_crop, dst=q_bgr, mask=seam_mask, p=(center_x, center_y), flags=cv2.NORMAL_CLONE
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