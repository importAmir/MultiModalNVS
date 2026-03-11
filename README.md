## MultiModalNVS: Official implementation of “A Single Image and Multimodality Is All You Need for Novel View Synthesis”

Published at **ICLR 2026 Workshop on Multimodal Intelligence Workshop**.

The method reconstructs dense depth from extremely sparse multimodal range measurements using localized Gaussian Processes, and plugs this depth into diffusion-based rendering pipelines for more robust and geometrically consistent novel-view synthesis.

![Pipeline overview](images/pipeline_preview.png)

## Quick installation with Pixi

### 1. Install Pixi

```bash
# Linux/macOS
curl -fsSL https://pixi.sh/install.sh | bash

# macOS with Homebrew
brew install pixi

# Windows (PowerShell)
iwr https://pixi.sh/install.ps1 -useb | iex

# After installation, restart your terminal or run:
source ~/.bashrc  # or source ~/.zshrc for zsh

pixi --version
```

### 2. Clone this repository

```bash
git clone --recurse-submodules https://github.com/importAmir/MultiModalNVS.git
cd MultiModalNVS
```

### 3. Create and activate the Pixi environment

```bash
pixi install
pixi shell
```

### 4. Install additional dependencies

```bash
pixi run install-all-deps
```

### 5. Import GEN3C and download checkpoints

You need a Hugging Face account and token to download GEN3C checkpoints. Get your token from [Hugging Face Settings](https://huggingface.co/settings/tokens) (Read permission).

```bash
pixi run import
pixi run download-gen3c-checkpoints -- <HF_TOKEN>
```

### 6. DepthAnythingV2 (optional)

```bash
pixi run import-dav2
pixi run download-dav2-checkpoints
```

### 7. Copy custom code into the GEN3C pipeline

This step synchronizes the local customization code in this repository into the `GEN3C` submodule so it is available to the GEN3C pipeline. Run this again whenever you modify the local customization scripts.

```bash
pixi run copy-custom
```

## Code overview

### Dense depth from sparse range (`predict_dense_depth.py`)

`predict_dense_depth.py` predicts dense camera-Z depth on a raster of size `H x W` from sparse 3D points.

```bash
pixi run predict-dense-depth -- \
  --points-bin /path/to/000750.bin \
  --calib-txt /path/to/000750.txt \
  --image-size 1248 1920
```

#### Input format (View of Delft example)

- **`--points-bin`**: A binary file of `float32` values.
  - View of Delft radar format (default `--points-format vod7`): shape `(N, 7)` with columns
    `x, y, z, RCS, v_r, v_r_comp, t_id`.
  - Example path: `view_of_delft_PUBLIC/radar/training/velodyne/00000.bin`
- **`--calib-txt`**: A KITTI-style calibration text file that contains:
  - `Tr_velo_to_cam:` (12 numbers forming a 3×4 extrinsic matrix)
  - `P2:` (3×4 projection matrix; intrinsics are read from this by default)
  - Example path: `view_of_delft_PUBLIC/radar/training/calib/00000.txt`

### Rendering

Rendering uses the predicted depth to produce warps/masks for a target camera trajectory.

```bash
pixi run run_render_only_image_generic -- \
  --input_image_path /path/to/image.jpg \
  --lidar_path /path/to/depth.npy \
  --filter_points_threshold 1.0 \
  --trajectory_generation_method action_based_movement \
  --trajectory left \
  --camera_rotation center_facing \
  --movement_distance 1 \
  --default_fx 1495.468642 \
  --default_fy 1495.468642 \
  --default_cx 961.272442 \
  --default_cy 624.89592 \
  --rendered_images_path rendered_warp_images.pt \
  --rendered_masks_path rendered_warp_masks.pt
```

### Rendering preview video

```bash
pixi run run_generate_rendering_video_generic -- \
  --rendered_images_path rendered_warp_images.pt \
  --video_save_name render_preview
```

### Diffusion

Diffusion consumes the rendered warps + masks and produces the final novel-view video.

```bash
pixi run run_diffusion_only_generic -- \
  --input_image_path /path/to/image.jpg \
  --rendered_images_path rendered_warp_images.pt \
  --rendered_masks_path rendered_warp_masks.pt \
  --offload_diffusion_transformer \
  --offload_tokenizer \
  --disable_prompt_encoder \
  --save_buffer \
  --video_save_name diffusion_result \
  --video_save_folder outputs/
```

## Results

![NVS metric](images/NVS_Metric.png)

![Depth metric](images/Depth_metric.png)

![Visual results preview](images/visual_results_preview.png)