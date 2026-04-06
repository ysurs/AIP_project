# AIP Project — Scene Completion Pipeline

An implementation of the Hays & Efros *Scene Completion Using Millions of Photographs* pipeline, extended with three enhancement flags (EF1, EF2, EF3).

---

## Pipeline Overview

```
Input Image + Hole Mask
        │
        ▼
[1] Scene Matching (GIST + Color features)
        │  → finds k nearest scenes from a database
        ▼
[2] Local Context Matching
        │  → aligns each candidate patch to the hole's context region
        ▼
[3] Graph Cut Seam Finding
        │  → computes the optimal seam (Dinic's min-cut via C)
        ▼
[4] Poisson / Seamless Blending
        │  → composites the patch into the query image
        ▼
Output Completed Image
```

### Enhancement Flags

| Flag | Component | Description |
|------|-----------|-------------|
| `--use_ef1` | EF1 — Auto Ranking | Re-ranks candidates using seam energy (min-cut cost) from LCM; picks the best composite automatically |
| `--use_ef2` | EF2 — SAM Segmentation | Replaces the hand-drawn mask UI with a click-to-segment interface backed by Meta's Segment Anything Model (SAM ViT-B) |
| `--use_ef3` | EF3 — Super Resolution | Matches against a compact "tiny" database; super-resolves each matched candidate image up to query resolution using Real-ESRGAN x4plus before blending |

---

## Component Descriptions

| File | Role |
|------|------|
| `main.py` | **Main entry point.** Tkinter GUI for brush-painting the hole mask; orchestrates the full pipeline. Also contains `seamless_clone_pure` — Poisson/seamless blending via iterative Jacobi solver. |
| `feature_extraction.py` | GIST descriptor and color-histogram feature extraction for scene-level retrieval. |
| `match_scenes.py` | Loads/caches GIST+color features for every database image; ranks them against the query using weighted SSD. |
| `local_context_matching.py` | Crops a "context donut" around the hole and runs masked SSD template matching at multiple scales to place each candidate. Wraps `lcm_solver.c` via ctypes. |
| `graph_cut.py` | Builds a 4-connected pixel graph and calls the C max-flow solver to find the optimal seam mask. |
| `ef2_segmentation.py` | SAM ViT-B wrapper for point-prompted segmentation; pure-Python mask merge/remove/overlay helpers. |
| `super_resolve.py` | Real-ESRGAN x4plus upsampler (EF3). Loaded lazily; uses tiled inference to avoid OOM. |
| `lcm_solver.py` | ctypes wrapper for `lcm_solver.c` — BGR→gray, BGR→LAB, morphological dilation, bilinear resize, texture map (Sobel+median), masked SSD. Auto-compiles on first import. |
| `maxflow_solver.py` | ctypes wrapper for `graph_cut_solver.c` — Dinic's max-flow. Auto-compiles on first import. |
| `skyline_data.py` | One-time utility to flatten a multi-city dataset into a flat `skyline_1024/` image database. |
| `create_tiny_db.py` | Samples a small subset of the full database into `skyline_tiny/` for EF3 fast matching. |
| `metric.py` | Evaluation utilities (PSNR / SSIM). |

### C Shared Libraries

| Source | Output | Purpose |
|--------|--------|---------|
| `lcm_solver.c` | `lcm_solver_lib.so` | Fast color-space conversions, dilation, resize, SSD for local context matching |
| `graph_cut_solver.c` | `graph_cut_solver.so` | Dinic's max-flow algorithm for graph-cut seam finding |

> Both `.so` files are auto-compiled by their Python wrappers on first import if `gcc` is available. You can also compile them manually (see below).

---

## Environment Setup

### 1. Create and activate the conda environment

```bash
conda create -n aip_project python=3.11.4 -y
conda activate aip_project
```

### 2. Install core dependencies

```bash
pip install "numpy==2.4.2" "Pillow==12.1.1" "opencv-python==4.13.0.92" "matplotlib==3.10.8" "tqdm==4.67.3"
```

### 3. Install PyTorch (required for EF2 and EF3)

For Apple Silicon (MPS):
```bash
pip install "torch==2.10.0" "torchvision==0.25.0"
```

For CUDA (Linux/Windows):
```bash
pip install "torch==2.10.0" "torchvision==0.25.0" --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install EF2 — Segment Anything Model

```bash
pip install segment-anything
```

Download the SAM ViT-B checkpoint (~375 MB) and place it in the project root:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

Or set the environment variable to point to a different location:
```bash
export SAM_CHECKPOINT=/path/to/sam_vit_b_01ec64.pth
```

### 5. Install EF3 — Real-ESRGAN

```bash
pip install realesrgan basicsr facexlib gfpgan
```

> The Real-ESRGAN model weights are downloaded automatically on first use from GitHub releases.

---

## Compiling the C Libraries

Both libraries compile automatically when first imported. To compile manually:

```bash
# Local Context Matching solver
gcc -O3 -march=native -shared -fPIC -o lcm_solver_lib.so lcm_solver.c -lm

# Graph Cut (Dinic's max-flow) solver
gcc -O3 -shared -fPIC -o graph_cut_solver.so graph_cut_solver.c
```

> Requires `gcc`. On macOS, install via `xcode-select --install` or `brew install gcc`.

---

## Dataset

The image databases (`skyline_1024/` and `skyline_tiny/`) are hosted on Hugging Face:
**[https://huggingface.co/datasets/YashS/Skyline](https://huggingface.co/datasets/YashS/Skyline)**

### Download the dataset

```bash
pip install huggingface_hub
python download_data.py
```

This will download and place `skyline_1024/` and `skyline_tiny/` in the project root, ready to use.

---

## Preparing the Image Database

> If you downloaded the dataset using `download_data.py` above, you can skip this step — the folders are already prepared.

```bash
# Flatten a multi-city dataset into skyline_1024/
python skyline_data.py

# Create a compact tiny subset for EF3
python create_tiny_db.py
```

---

## Running the Pipeline

### Interactive mode (GUI mask drawing)

```bash
conda activate aip_project
python main.py
```

Opens a file dialog to pick an image, then a Tkinter canvas for painting the hole mask.

### Headless mode (supply image + mask directly)

> Requires `image_1024.png` and `mask_1024.png` to already exist in the project root — these are generated automatically the first time you run the interactive mode above.

```bash
python main.py --image image_1024.png --mask mask_1024.png
```

### With enhancement flags

```bash
# EF1: automatic candidate ranking by seam energy
python main.py --image image_1024.png --mask mask_1024.png --use_ef1

# EF2: SAM click-to-segment mask interface
python main.py --image image_1024.png --use_ef2

# EF3: tiny-DB matching + super-resolution output
python main.py --image image_1024.png --mask mask_1024.png --use_ef3

```


---

## Output

All output files are written to the project root directory. The filenames depend on which flags are active:

| Mode | Output filename(s) |
|------|--------------------|
| Base pipeline (no flags) | `final_completed_image_0.png` … `final_completed_image_9.png` |
| `--use_ef1` | `final_completed_image_EF1_BEST.png` |
| `--use_ef3` | `final_completed_image_0.png` … `final_completed_image_9.png` |

> EF3 component outputs the same file names as there is code reuse happening between EF3 and base component
