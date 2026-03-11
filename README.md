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

Main command:

```bash
pixi run predict-dense-depth -- \
  --points-bin /path/to/points.bin \
  --calib-txt /path/to/calib.txt \
  --image-size H W \
  --output /path/to/output_depth.npy
```

Inputs:

- **`--points-bin`**: Binary `float32` point file. View of Delft uses the default `--points-format vod7` with shape `(N, 7)`:
  `x, y, z, RCS, v_r, v_r_comp, t_id`.
- **`--calib-txt`**: KITTI-style calibration text file that contains `Tr_velo_to_cam:` (3×4 extrinsic) and `P2:` (3×4 projection; used for intrinsics).
- **`--image-size H W`**: Dense depth output resolution.
- **`--output`**: Output depth `.npy` file. Two additional sidecar files are saved alongside:
  - `*_variance.npy`
  - `*_valid.npy`

Example (included View of Delft sample):

```bash
pixi run predict-dense-depth -- \
  --points-bin "examples/vod_8350-8365/velodyne/08350.bin" \
  --calib-txt "examples/vod_8350-8365/calib/08350.txt" \
  --image-size 1248 1920 \
  --output "examples/vod_8350-8365/depth/08350_pred_depth.npy"
```

### Rendering pipeline

After you generate `examples/vod_8350-8365/depth/08350_pred_depth.npy` in the step above, run the full pipeline for a short sequence. It renders novel views from a reference image + poses using the depth map, applies diffusion, and writes the outputs/metrics to `--output_dir`.

```bash
pixi run run_evaluate_video_quality -- \
  --input_image "examples/vod_8350-8365/images/08350.jpg" \
  --poses_extrinsics_dir "examples/vod_8350-8365/poses" \
  --reference_images_folder "examples/vod_8350-8365/images" \
  --output_dir "evaluation_results/depth_sequence/08350" \
  --lidar_path "examples/vod_8350-8365/depth/08350_pred_depth.npy" \
  --default_fx 1495.468642 \
  --default_fy 1495.468642 \
  --default_cx 961.272442 \
  --default_cy 624.89592 \
  --num_video_frames 121 \
  --fps 24 \
  --offload_diffusion_transformer \
  --offload_tokenizer \
  --disable_prompt_encoder \
  --save_buffer
```

#### Example data (View of Delft)

The repo includes a small example under `examples/vod_8350-8365/` (derived from **View of Delft (VoD)**):

```bash
EX="examples/vod_8350-8365"

ls -1 "$EX/images/" | head
ls -1 "$EX/poses/"  | head

ls -lh \
  "$EX/images/08350.jpg" \
  "$EX/poses/08350.json" \
  "$EX/calib/08350.txt" \
  "$EX/velodyne/08350.bin" \
  "$EX/intrinsics/08350.json" \
  "$EX/depth/08350_dense_depth.npy"
```

## Results

![NVS metric](images/NVS_Metric.png)

![Depth metric](images/Depth_metric.png)

![Visual results preview](images/visual_results_preview.png)

## Citation

### View of Delft dataset

```bibtex
@ARTICLE{apalffy2022,
  author={Palffy, Andras and Pool, Ewoud and Baratam, Srimannarayana and Kooij, Julian F. P. and Gavrila, Dariu M.},
  journal={IEEE Robotics and Automation Letters}, 
  title={Multi-Class Road User Detection With 3+1D Radar in the View-of-Delft Dataset}, 
  year={2022},
  volume={7},
  number={2},
  pages={4961-4968},
  doi={10.1109/LRA.2022.3147324}}
```

### This work

If you use this codebase in your research, please cite:

```bibtex
@article{javadi2026single,
  title={A Single Image and Multimodality Is All You Need for Novel View Synthesis},
  author={Javadi, Amirhosein and Gau, Chi-Shiang and Polyzos, Konstantinos D and Javidi, Tara},
  journal={arXiv preprint arXiv:2602.17909},
  year={2026}
}
```