## MultiModalNVS: Official implementation of “A Single Image and Multimodality Is All You Need for Novel View Synthesis”

Published at **ICLR 2026 Workshop on Multimodal Intelligence Workshop**.

The method reconstructs **dense depth** from extremely sparse multimodal range measurements using **localized Gaussian Processes**, and plugs this depth into diffusion-based rendering pipelines for more robust and geometrically consistent novel-view synthesis.

- **Pipeline**: [`images/pipeline.pdf`](images/pipeline.pdf)

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

This copies **only tracked scripts in this repo** into the `GEN3C` submodule.

```bash
pixi run copy-custom
```

## Code overview

### Dense depth from sparse range (`predict_dense_depth.py`)

`predict_dense_depth.py` predicts dense camera-Z depth on a raster of size `H x W` from sparse 3D points using:
- **`local_gp_mle`** (default): local GP per pixel with per-target length-scale optimization
- **`kernel`**: fast Gaussian kernel regression

```bash
python3 predict_dense_depth.py \
  --points-bin /path/to/000750.bin \
  --calib-txt /path/to/000750.txt \
  --image-size 1248 1920 \
  --prediction-model local_gp_mle
```

### Rendering

Rendering uses the predicted depth to produce warps/masks for a target camera trajectory.

### Diffusion

Diffusion consumes the rendered warps + masks and produces the final novel-view video.

```bash
# 1) Render warps/masks
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

# 2) Preview rendered visualization
pixi run run_generate_rendering_video_generic -- \
  --rendered_images_path rendered_warp_images.pt \
  --video_save_name render_preview

# 3) Diffusion refinement
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

- Visual results: [`images/visual_results.pdf`](images/visual_results.pdf)

![Visual results preview](images/visual_results_preview.png)