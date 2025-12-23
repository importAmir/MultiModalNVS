# GEN3C Project

<div align="center">

**Multi-modal Novel View Synthesis**

3D-Informed World-Consistent Video Generation with Precise Camera Control

</div>

## 🚀 Quick Installation with Pixi

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

# Verify Pixi installation
pixi --version
```

### 2. Navigate to the project directory
```bash
cd GEN3C-Project
```

### 3. Install the Pixi environment
```bash
pixi install
```

**Note**: This automatically installs DVC with Azure support along with all other dependencies.

### 4. Activate the environment
```bash
pixi shell
```

### 5. Install dependencies
It takes approximately two hours!
```bash
pixi run install-all-deps
```

### 6. Import GEN3C submodule
```bash
pixi run import
```

### 7. Download checkpoints
You need a Hugging Face account and token to download the GEN3C checkpoints. Get your token from [Hugging Face Settings](https://huggingface.co/settings/tokens). Set the access token to Read permission (default is Fine-grained). 
```bash
pixi run download-gen3c-checkpoints -- (Your Token)
```

### 8. Import DepthAnythingV2 submodule (Optional)
If you plan to use the DepthAnythingV2 depth estimator, import the submodule:
```bash
pixi run import-dav2
```

### 9. Download DepthAnythingV2 checkpoint (Optional)
If you plan to use the DepthAnythingV2 depth estimator, download the checkpoint:
```bash
pixi run download-dav2-checkpoints
```

### 10. Copy custom code to GEN3C pipeline
After making any changes to the custom scripts, run this command to copy them into the GEN3C pipeline:
```bash
pixi run copy-custom
```

**Important**: Run this command every time you modify any of the custom scripts to ensure your changes are available in the GEN3C pipeline.

### 11. Reading drone samples data with DVC

Once DVC is set up, you can easily manage the drone samples dataset:

```bash
# Pull the latest drone samples data from Azure storage
dvc pull

# Check the status of your local data vs remote storage
dvc status
```

**Common DVC operations:**

- `dvc pull`: Downloads data from remote storage to your local workspace
- `dvc push`: Uploads local data changes to remote storage
- `dvc status`: Shows if your local data matches the remote version
- `dvc add <folder>`: Start tracking a new folder with DVC
- `dvc remove <folder>`: Stop tracking a folder (doesn't delete the data)

**Note**: The `drone_samples` folder is automatically ignored by Git but tracked by DVC. This means you can safely delete the local folder and restore it anytime with `dvc pull`.

To update data to the latest version:
```bash
dvc pull --all-branches
```

## Rendering Strategies

The script supports three different strategies for generating camera trajectories:

### 1. Action-based Movement (Default)
Generates predefined camera motions like left, right, up, down, zoom in/out, and rotations.
- **Required inputs**: 
  - `--trajectory`: Camera movement direction. Options: `left`, `right`, `up`, `down`, `zoom_in`, `zoom_out`, `clockwise`, `counterclockwise`, `none`
  - `--camera_rotation`: How the camera rotates during movement. Options: `center_facing` (always look at center), `no_rotation` (maintain orientation), `trajectory_aligned`
  - `--movement_distance`: How far the camera moves from the center 

### 2. Pixel Focusing
Moves the camera to focus on a specific pixel in the image over time.
- **Required inputs**: 
  - `--target_pixel_x`: X-coordinate of the target pixel to focus on (0 to image width)
  - `--target_pixel_y`: Y-coordinate of the target pixel to focus on (0 to image height)
  - `--movement_ratio`: How much to move toward the target (0.0 = no movement, 1.0 = full movement to target)
  - `--start_transition_frames`: Frame number when the transition to target begins
  - `--end_transition_frames`: Frame number when the transition to target completes

### 3. Source-to-Target Linear Interpolation
Interpolates between two camera poses defined by metadata files.
- **Required inputs**: 
  - `--source_meta_path`: Path to JSON metadata file containing the starting camera pose
  - `--target_meta_path`: Path to JSON metadata file containing the ending camera pose
  - `--start_transition_frames`: Frame number when the transition begins
  - `--end_transition_frames`: Frame number when the transition completes

## Depth Estimation Models

The script supports two depth estimation approaches:

### 1. MoGe (Default)

### 2. DepthAnythingV2
- **Additional requirements**: 
  - Default camera intrinsics in original image size: `--default_fx`, `--default_fy`, `--default_cx`, `--default_cy`


## Running Your Scripts

### Example: Complete Pixel Focusing Workflow with DepthAnythingV2
This example shows how to use the generic commands to create a complete workflow:

#### Step 1: Generate rendered tensors using pixel focusing
```bash
pixi run run_render_only_image_generic -- \
  --input_image_path GEN3C/assets/diffusion/000000.png \
  --trajectory_generation_method pixel_focusing \
  --target_pixel_x 640 \
  --target_pixel_y 352 \
  --movement_ratio 0.75 \
  --start_transition_frames 30 \
  --end_transition_frames 120 \
  --depth_estimator depthanythingv2 \
  --rendered_images_path rendered_warp_images_pixel_focus_DAV2.pt \
  --rendered_masks_path rendered_warp_masks_pixel_focus_DAV2.pt
```

#### Step 2: Generate video using diffusion-only
```bash
pixi run run_diffusion_only_generic -- \
  --input_image_path GEN3C/assets/diffusion/000000.png \
  --rendered_images_path rendered_warp_images_pixel_focus_DAV2.pt \
  --rendered_masks_path rendered_warp_masks_pixel_focus_DAV2.pt \
  --video_save_name test_diffusion_only_pixel_focus_DAV2 \
  --guidance 1 \
  --prompt "" \
  --save_buffer
```

### Example: Source-to-Target Linear Interpolation Workflow

This example shows how to interpolate between two camera poses:

#### Step 1: Generate rendered tensors using source-to-target interpolation
```bash
pixi run run_render_only_image_generic -- \
  --input_image_path drone_samples/test_004/_1734644989.985579.jpg \
  --trajectory_generation_method source_to_target_linear_interpolation \
  --source_meta_path drone_samples/test_004/_1734644989.985579_metadata.json \
  --target_meta_path drone_samples/test_004/_1734644976.704000_metadata.json \
  --start_transition_frames 0 \
  --end_transition_frames 120 \
  --depth_estimator depthanythingv2 \
  --rendered_images_path rendered_warp_images_source2target_test004_DAV2.pt \
  --rendered_masks_path rendered_warp_masks_source2target_test004_DAV2.pt
```

#### Step 2: Generate video using diffusion-only
```bash
pixi run run_diffusion_only_generic -- \
  --input_image_path drone_samples/test_004/_1734644989.985579.jpg \
  --rendered_images_path rendered_warp_images_source2target_test004_DAV2.pt \
  --rendered_masks_path rendered_warp_masks_source2target_test004_DAV2.pt \
  --video_save_name test_diffusion_only_source2target_test004 \
  --guidance 1 \
  --prompt "" \
  --save_buffer
```

### Example: Typical Action-based Movement Workflow

This example shows the default trajectory generation method:

#### Step 1: Generate rendered tensors using action-based movement
```bash
pixi run run_render_only_image_generic -- \
  --input_image_path GEN3C/assets/diffusion/000000.png \
  --trajectory_generation_method action_based_movement \
  --trajectory left \
  --camera_rotation center_facing \
  --movement_distance 0.3 \
  --depth_estimator moge \
  --rendered_images_path rendered_warp_images_left.pt \
  --rendered_masks_path rendered_warp_masks_left.pt
```

#### Step 2: Generate video using diffusion-only
```bash
pixi run run_diffusion_only_generic -- \
  --input_image_path GEN3C/assets/diffusion/000000.png \
  --rendered_images_path rendered_warp_images_left.pt \
  --rendered_masks_path rendered_warp_masks_left.pt \
  --video_save_name test_diffusion_only_left \
  --guidance 1 \
  --prompt "" \
  --save_buffer
```

### Example: Create Video from Rendered Tensor

This example shows how to create a rendering video:

#### Create video from rendered tensor
```bash
pixi run run_generate_rendering_video_generic -- \
  --rendered_images_path rendered_warp_images_source2target_test004_DAV2.pt \
  --video_save_name rendering_video_source2target
```

### Example: Video Input Rendering with Drone Data

This example shows how to create rendered tensors from a sequence of drone images with specific camera parameters:

#### Step 1: Generate rendered tensors from drone video sequence
```bash
pixi run run_render_only_video_generic -- \
  --input_folder drone_samples/test_006/doer \
  --trajectory_generation_method action_based_movement \
  --trajectory zoom_out \
  --camera_rotation center_facing \
  --movement_distance 10 \
  --foreground_masking \
  --gimbal_pitch -90 \
  --gimbal_yaw 0 \
  --gimbal_roll 0 \
  --rendered_images_path rendered_warp_images_zoomout_doer.pt \
  --rendered_masks_path rendered_warp_masks_zoomout_doer.pt
```
#### Step 2: Generate video using diffusion-only
```bash
pixi run run_diffusion_only_generic -- \
  --input_image_path drone_samples/test_006/doer/front_camera_1753232045.813325.jpg \
  --rendered_images_path rendered_warp_images_zoomout_doer.pt \
  --rendered_masks_path rendered_warp_masks_zoomout_doer.pt \
  --video_save_name test_diffusion_only_doer_zoom_out \
  --guidance 1 \
  --prompt "" \
  --save_buffer
```

### Example: Video Input Rendering with Pixel Focusing (Watcher Data)

This example shows how to create rendered tensors using pixel focusing trajectory on watcher drone data:

#### Step 1: Generate rendered tensors from drone video sequence (Pixel Focusing)
```bash
pixi run run_render_only_video_generic -- \
  --input_folder drone_samples/test_006/watcher \
  --trajectory_generation_method pixel_focusing \
  --target_pixel_x 1051 \
  --target_pixel_y 392 \
  --movement_ratio 0.75 \
  --start_transition_frames 30 \
  --end_transition_frames 120 \
  --foreground_masking \
  --gimbal_pitch -90 \
  --gimbal_yaw 0 \
  --gimbal_roll 0 \
  --rendered_images_path rendered_warp_images_pixel_focus_watcher_1051_392.pt \
  --rendered_masks_path rendered_warp_masks_pixel_focus_watcher_1051_392.pt
```

#### Step 2: Generate video using diffusion-only (Pixel Focusing)
```bash
pixi run run_diffusion_only_generic -- \
  --input_image_path drone_samples/test_006/watcher/front_camera_1753232045.812008.jpg \
  --rendered_images_path rendered_warp_images_pixel_focus_watcher_1051_392.pt \
  --rendered_masks_path rendered_warp_masks_pixel_focus_watcher_1051_392.pt \
  --video_save_name test_diffusion_only_pixel_focus_watcher_1051_392 \
  --guidance 1 \
  --prompt "" \
  --save_buffer
```

### Example: Video Input Rendering with Source-to-Target Linear Interpolation (Watcher to Doer)

This example shows how to create rendered tensors by interpolating between two camera poses from different drone sequences:

#### Step 1: Generate rendered tensors from drone video sequence (Source-to-Target Interpolation)
```bash
pixi run run_render_only_video_generic -- \
  --input_folder drone_samples/test_006/watcher \
  --trajectory_generation_method source_to_target_linear_interpolation \
  --source_meta_path drone_samples/test_006/watcher/front_camera_1753232045.812008_metadata.json \
  --target_meta_path drone_samples/test_006/doer/front_camera_1753232074.463422_metadata.json \
  --start_transition_frames 30 \
  --end_transition_frames 120 \
  --foreground_masking \
  --gimbal_pitch -90 \
  --gimbal_yaw 0 \
  --gimbal_roll 0 \
  --rendered_images_path rendered_warp_images_source2target_watcher_to_doer.pt \
  --rendered_masks_path rendered_warp_masks_source2target_watcher_to_doer.pt
```

#### Step 2: Generate video using diffusion-only (Source-to-Target Interpolation)
```bash
pixi run run_diffusion_only_generic -- \
  --input_image_path drone_samples/test_006/watcher/front_camera_1753232045.812008.jpg \
  --rendered_images_path rendered_warp_images_source2target_watcher_to_doer.pt \
  --rendered_masks_path rendered_warp_masks_source2target_watcher_to_doer.pt \
  --video_save_name test_diffusion_only_source2target_watcher_to_doer \
  --guidance 1 \
  --prompt "" \
  --save_buffer
```

### Example: Video Input Rendering with Target Folder Trajectory (Doer to Watcher Perspective)

This example shows how to create rendered tensors by using multiple target poses from a watcher metadata folder to generate trajectories from doer images:

#### Step 1: Generate rendered tensors from drone video sequence (Target Folder Trajectory)
```bash
pixi run run_render_only_video_generic -- \
  --input_folder drone_samples/test_006/doer \
  --trajectory_generation_method target_folder_trajectory \
  --target_meta_folder drone_samples/test_006/watcher \
  --foreground_masking \
  --gimbal_pitch -90 \
  --gimbal_yaw 0 \
  --gimbal_roll 0 \
  --rendered_images_path rendered_warp_images_target_folder_doer_to_watcher.pt \
  --rendered_masks_path rendered_warp_masks_target_folder_doer_to_watcher.pt
```

#### Step 2: Generate video using diffusion-only (Target Folder Trajectory)
```bash
pixi run run_diffusion_only_generic -- \
  --input_image_path drone_samples/test_006/watcher/front_camera_1753232045.812008.jpg \
  --rendered_images_path rendered_warp_images_target_folder_doer_to_watcher.pt \
  --rendered_masks_path rendered_warp_masks_target_folder_doer_to_watcher.pt \
  --video_save_name test_diffusion_only_target_folder_doer_to_watcher \
  --guidance 1 \
  --prompt "" \
  --save_buffer
```

### Example: Video Input Rendering with ViPE Data

This example shows how to create rendered tensors using ViPE (Video Pose Engine) output data instead of traditional image/metadata folders:

#### Step 1: Generate rendered tensors from ViPE data using action-based movement
```bash
pixi run run_render_only_video_generic -- \
  --vipe_path ../PoseDepthEstimation/vipe_results \
  --trajectory_generation_method action_based_movement \
  --trajectory zoom_out \
  --camera_rotation center_facing \
  --movement_distance 1 \
  --foreground_masking \
  --rendered_images_path rendered_warp_images_vipe_zoom_out.pt \
  --rendered_masks_path rendered_warp_masks_vipe_zoom_out.pt
```

#### Step 2: Generate video using diffusion-only with ViPE data
```bash
pixi run run_diffusion_only_generic -- \
  --vipe_path ../PoseDepthEstimation/vipe_results \
  --rendered_images_path rendered_warp_images_vipe_zoom_out.pt \
  --rendered_masks_path rendered_warp_masks_vipe_zoom_out.pt \
  --video_save_name test_diffusion_only_vipe_zoomout \
  --guidance 1 \
  --prompt "" \
  --save_buffer
```

### Example: Video Input Rendering with ViPE Data and LiDAR Alignment

This example shows how to create rendered tensors using ViPE data with LiDAR depth alignment:

#### Step 1: Generate rendered tensors from ViPE data with LiDAR alignment
```bash
pixi run run_render_only_video_generic -- \
  --vipe_path ../PoseDepthEstimation/vipe_results \
  --trajectory_generation_method action_based_movement \
  --trajectory zoom_out \
  --camera_rotation center_facing \
  --movement_distance 1 \
  --foreground_masking \
  --align_depth_with_lidar \
  --lidar_path ../PoseDepthEstimation/vipe_lidar_data.npy \
  --rendered_images_path rendered_warp_images_vipe_lidar_aligned.pt \
  --rendered_masks_path rendered_warp_masks_vipe_lidar_aligned.pt
```

#### Step 2: Generate video using diffusion-only with ViPE data and LiDAR alignment
```bash
pixi run run_diffusion_only_generic -- \
  --vipe_path ../PoseDepthEstimation/vipe_results \
  --rendered_images_path rendered_warp_images_vipe_lidar_aligned.pt \
  --rendered_masks_path rendered_warp_masks_vipe_lidar_aligned.pt \
  --video_save_name test_diffusion_only_vipe_lidar_aligned \
  --guidance 1 \
  --prompt "" \
  --save_buffer
```

### Example: Multiview Image Input Rendering

This example shows how to create rendered tensors from multiple input images of the same scene using `Cache3D_BufferSelector` for multiview rendering:

#### Step 1: Generate rendered tensors from multiview images
```bash
pixi run run_render_only_multiview_image_generic -- \
  --input_folder drone_samples/test_001 \
  --trajectory_generation_method action_based_movement \
  --trajectory right \
  --camera_rotation no_rotation \
  --movement_distance 5 \
  --reference_frame 0 \
  --rendered_images_path multiview_warp_images_right.pt \
  --rendered_masks_path multiview_warp_masks_right.pt
```

#### Step 2: Generate video using diffusion-only with multiview data
```bash
pixi run run_diffusion_only_generic -- \
  --input_image_path drone_samples/test_001/building-May-14-m3-no-pose_front_camera_1747254834.832100.jpg \
  --rendered_images_path multiview_warp_images_right.pt \
  --rendered_masks_path multiview_warp_masks_right.pt \
  --video_save_name test_diffusion_only_multiview_right \
  --guidance 1 \
  --prompt "" \
  --save_buffer
```

#### Multiview with Source-to-Target Linear Interpolation
```bash
pixi run run_render_only_multiview_image_generic -- \
  --input_folder drone_samples/test_004 \
  --trajectory_generation_method source_to_target_linear_interpolation \
  --source_meta_path drone_samples/test_004/_1734644976.704000_metadata.json \
  --target_meta_path drone_samples/test_004/_1734644989.985579_metadata.json \
  --reference_frame 0 \
  --start_transition_frames 30 \
  --end_transition_frames 120 \
  --rendered_images_path multiview_warp_images_source_to_target.pt \
  --rendered_masks_path multiview_warp_masks_source_to_target.pt
```

#### Step 2: Generate video using diffusion-only with source-to-target data
```bash
pixi run run_diffusion_only_generic -- \
  --input_image_path drone_samples/test_004/_1734644976.704000.jpg \
  --rendered_images_path multiview_warp_images_source_to_target.pt \
  --rendered_masks_path multiview_warp_masks_source_to_target.pt \
  --video_save_name test_diffusion_only_multiview_source_to_target \
  --guidance 1 \
  --prompt "" \
  --save_buffer
```

**Note**: For multiview rendering, ensure your input folder contains:
- Multiple images of the same scene from different viewpoints
- Corresponding metadata files (image_name_metadata.json) for each image
- Use `--reference_frame` to control which camera view is used for trajectory generation

### Example: VGGT Integration for Multiview Rendering

This example shows how to use VGGT (Visual Geometry Grounded Transformer) output for multiview rendering, which provides more accurate camera poses and depth estimation:

#### Step 1: Run VGGT to estimate poses, intrinsics, and depth maps
```bash
# First, run VGGT on test_004 images
cd ../VGGT-Project
pixi run run-vggt-args -- --image_folder drone_samples/test_004 --output_folder test_004_result
```

#### Step 2: Generate rendered tensors using VGGT output
```bash
# Return to GEN3C project
cd ../GEN3C-Project
pixi run run_render_only_multiview_image_generic -- \
  --vggt_output_folder ../VGGT-Project/test_004_result \
  --trajectory_generation_method action_based_movement \
  --trajectory right \
  --camera_rotation center_facing \
  --movement_distance 15 \
  --reference_frame 0 \
  --rendered_images_path vggt_multiview_warp_images_right.pt \
  --rendered_masks_path vggt_multiview_warp_masks_right.pt
```

#### Step 3: Generate video using diffusion-only with VGGT data
```bash
pixi run run_diffusion_only_generic -- \
  --input_image_path ../VGGT-Project/test_004_result/images/_1734644976.704000.jpg \
  --rendered_images_path vggt_multiview_warp_images_right.pt \
  --rendered_masks_path vggt_multiview_warp_masks_right.pt \
  --video_save_name test_diffusion_only_vggt_multiview_right \
  --guidance 1 \
  --prompt "" \
  --save_buffer
```

#### VGGT with Pixel Focusing Trajectory
```bash
pixi run run_render_only_multiview_image_generic -- \
  --vggt_output_folder ../VGGT-Project/test_004_result \
  --trajectory_generation_method pixel_focusing \
  --target_pixel_x 256 \
  --target_pixel_y 256 \
  --movement_ratio 0.3 \
  --start_transition_frames 10 \
  --end_transition_frames 50 \
  --reference_frame 0 \
  --rendered_images_path vggt_multiview_warp_images_pixel_focus.pt \
  --rendered_masks_path vggt_multiview_warp_masks_pixel_focus.pt
```

#### Step 2: Generate video using diffusion-only with pixel focusing
```bash
pixi run run_diffusion_only_generic -- \
  --input_image_path ../VGGT-Project/test_004_result/images/_1734644976.704000.jpg \
  --rendered_images_path vggt_multiview_warp_images_pixel_focus.pt \
  --rendered_masks_path vggt_multiview_warp_masks_pixel_focus.pt \
  --video_save_name test_diffusion_only_vggt_pixel_focus \
  --guidance 1 \
  --prompt "" \
  --save_buffer
```

#### VGGT with Source-to-Target Linear Interpolation
```bash
pixi run run_render_only_multiview_image_generic -- \
  --vggt_output_folder ../VGGT-Project/test_004_result \
  --trajectory_generation_method source_to_target_linear_interpolation \
  --source_pose_path ../VGGT-Project/test_004_result/extrinsics/_1734644976.704000.npy \
  --target_pose_path ../VGGT-Project/test_004_result/extrinsics/_1734644989.985579.npy \
  --start_transition_frames 0 \
  --end_transition_frames 120 \
  --reference_frame 0 \
  --rendered_images_path vggt_multiview_warp_images_source_to_target.pt \
  --rendered_masks_path vggt_multiview_warp_masks_source_to_target.pt
```

#### Step 2: Generate video using diffusion-only with source-to-target
```bash
pixi run run_diffusion_only_generic -- \
  --input_image_path ../VGGT-Project/test_004_result/images/_1734644976.704000.jpg \
  --rendered_images_path vggt_multiview_warp_images_source_to_target.pt \
  --rendered_masks_path vggt_multiview_warp_masks_source_to_target.pt \
  --video_save_name test_diffusion_only_vggt_source_to_target \
  --guidance 1 \
  --prompt "" \
  --save_buffer
```

**VGGT Integration Benefits**:
- **More accurate poses**: Uses VGGT's estimated camera poses instead of metadata
- **Better depth estimation**: Uses VGGT's depth maps instead of DepthAnythingV2
- **Automatic intrinsics**: Uses VGGT's estimated camera intrinsics
- **No metadata required**: Works directly with VGGT output folders
- **Automatic scaling**: Intrinsics are properly scaled for target resolution

**VGGT Output Structure Required**:
```
vggt_output_folder/
├── images/               # Original images
├── intrinsics/           # 3x3 intrinsic matrices (.npy files)
├── extrinsics/           # 4x4 world2camera matrices (.npy files)
├── depth_maps/           # Depth maps in meters (.npy files)
└── mask/                 # Binary masks (.npy files)
```
- For source-to-target interpolation, poses are used directly without making them relative


### Example: Multiview Waymo Image Input Rendering (Source-to-Target Interpolation)

Render a sequence by interpolating the camera from `pose/1.npy` to `pose/3.npy` in `drone_samples/test_008`:

```bash
pixi run run_render_only_multiview_waymo_image_input_generic -- \
  --input_folder drone_samples/test_008 \
  --trajectory_generation_method source_to_target_linear_interpolation \
  --source_pose_path drone_samples/test_008/pose/1.npy \
  --target_pose_path drone_samples/test_008/pose/3.npy \
  --start_transition_frames 0 \
  --end_transition_frames 120 \
  --reference_frame 0 \
  --rendered_images_path waymo_multiview_warp_images_source_to_target.pt \
  --rendered_masks_path waymo_multiview_warp_masks_source_to_target.pt
```

#### Step 2: Generate video using diffusion-only with source-to-target

```bash
pixi run run_diffusion_only_generic -- \
  --input_image_path drone_samples/test_008/images/1.png \
  --rendered_images_path waymo_multiview_warp_images_source_to_target.pt \
  --rendered_masks_path waymo_multiview_warp_masks_source_to_target.pt \
  --video_save_name test_diffusion_only_waymo_source_to_target \
  --guidance 1 \
  --prompt "" \
  --save_buffer
```

### Example: Multiview Waymo Input Rendering with Depth Estimation (Source-to-Target Interpolation)

Render a sequence with LiDAR depth alignment by interpolating the camera from `pose/1.npy` to `pose/3.npy` in `drone_samples/test_008`:

```bash
pixi run run_render_only_multiview_waymo_image_input_generic -- \
  --input_folder drone_samples/test_008 \
  --trajectory_generation_method source_to_target_linear_interpolation \
  --source_pose_path drone_samples/test_008/pose/1.npy \
  --target_pose_path drone_samples/test_008/pose/3.npy \
  --start_transition_frames 0 \
  --end_transition_frames 120 \
  --align_depth_with_lidar \
  --reference_frame 0 \
  --rendered_images_path waymo_multiview_warp_images_source_to_target_lidar.pt \
  --rendered_masks_path waymo_multiview_warp_masks_source_to_target_lidar.pt
```

#### Step 2: Generate video using diffusion-only with LiDAR-aligned depth

```bash
pixi run run_diffusion_only_generic -- \
  --input_image_path drone_samples/test_008/images/1.png \
  --rendered_images_path waymo_multiview_warp_images_source_to_target_lidar.pt \
  --rendered_masks_path waymo_multiview_warp_masks_source_to_target_lidar.pt \
  --video_save_name test_diffusion_only_waymo_source_to_target_lidar \
  --guidance 1 \
  --prompt "" \
  --save_buffer
```

Notes:
- The script reads `images/`, `pose/`, `intrinsics/`, `mask/`, and `lidar/` from `--input_folder`.
- Add `--align_depth_with_lidar` to align DepthAnythingV2 depth to LiDAR where available.

### Example: Multiview Waymo Video Input Rendering (Source-to-Target Interpolation)

```bash
pixi run run_render_only_multiview_waymo_video_input_generic -- \
  --input_folder drone_samples/test_video_121_frames \
  --trajectory_generation_method source_to_target_linear_interpolation \
  --source_pose_path drone_samples/test_video_121_frames/source_pose.npy \
  --target_pose_path drone_samples/test_video_121_frames/target_pose.npy \
  --start_transition_frames 0 \
  --end_transition_frames 120 \
  --align_depth_with_lidar \
  --reference_frame 0 \
  --rendered_images_path waymo_video_input_multiview_warp_images_source_to_target_lidar.pt \
  --rendered_masks_path waymo_video_input_multiview_warp_masks_source_to_target_lidar.pt
```

```bash
pixi run run_diffusion_only_generic -- \
  --input_image_path  drone_samples/test_video_121_frames/source_image.png \
  --rendered_images_path waymo_video_input_multiview_warp_images_source_to_target_lidar.pt \
  --rendered_masks_path waymo_video_input_multiview_warp_masks_source_to_target_lidar.pt \
  --video_save_name test_diffusion_only_waymo_multiview_video_source_to_target_lidar \
  --guidance 1 \
  --prompt "" \
  --save_buffer
```


## LiDAR Depth Alignment

The scripts now support aligning predicted depth maps with LiDAR depth data for improved accuracy. This feature is available for both single image and video input processing.

### How LiDAR Alignment Works

1. **Depth Prediction**: The system first predicts depth using either MoGe or DepthAnythingV2
2. **LiDAR Loading**: Corresponding LiDAR depth maps are loaded (single file for image input, folder for video input)
3. **Alignment**: The predicted depth is aligned to match LiDAR depth where LiDAR data is available (non-zero values)
4. **Sparse Preservation**: LiDAR data is resized using specialized sparse-aware methods that preserve the exact sparse structure without interpolation artifacts

### Example: Single Image Input with LiDAR Alignment

This example shows how to align depth with LiDAR data for a single image:

#### Step 1: Generate rendered tensors with LiDAR alignment
```bash
pixi run run_render_only_image_generic -- \
  --input_image_path GEN3C/assets/diffusion/000000.png \
  --trajectory_generation_method pixel_focusing \
  --target_pixel_x 640 \
  --target_pixel_y 352 \
  --movement_ratio 0.75 \
  --start_transition_frames 30 \
  --end_transition_frames 120 \
  --depth_estimator depthanythingv2 \
  --align_depth_with_lidar \
  --lidar_path GEN3C/assets/diffusion/lidar/000000.npy \
  --rendered_images_path rendered_warp_images_lidar_aligned.pt \
  --rendered_masks_path rendered_warp_masks_lidar_aligned.pt
```

#### Step 2: Generate video using diffusion-only with LiDAR-aligned depth
```bash
pixi run run_diffusion_only_generic -- \
  --input_image_path GEN3C/assets/diffusion/000000.png \
  --rendered_images_path rendered_warp_images_lidar_aligned.pt \
  --rendered_masks_path rendered_warp_masks_lidar_aligned.pt \
  --video_save_name test_diffusion_only_lidar_aligned \
  --guidance 1 \
  --prompt "" \
  --save_buffer
```

### Example: Video Input with LiDAR Alignment

This example shows how to align depth with LiDAR data for each frame in a video sequence:

#### Step 1: Generate rendered tensors with per-frame LiDAR alignment
```bash
pixi run run_render_only_video_generic -- \
  --input_folder drone_samples/test_006/doer \
  --trajectory_generation_method action_based_movement \
  --trajectory zoom_out \
  --camera_rotation center_facing \
  --movement_distance 10 \
  --foreground_masking \
  --gimbal_pitch -90 \
  --gimbal_yaw 0 \
  --gimbal_roll 0 \
  --align_depth_with_lidar \
  --lidar_path drone_samples/test_006/doer_lidar.npy \
  --rendered_images_path rendered_warp_images_lidar_aligned_video.pt \
  --rendered_masks_path rendered_warp_masks_lidar_aligned_video.pt
```

#### Step 2: Generate video using diffusion-only with LiDAR-aligned depth
```bash
pixi run run_diffusion_only_generic -- \
  --input_image_path drone_samples/test_006/doer/front_camera_1753232045.813325.jpg \
  --rendered_images_path rendered_warp_images_lidar_aligned_video.pt \
  --rendered_masks_path rendered_warp_masks_lidar_aligned_video.pt \
  --video_save_name test_diffusion_only_doer_lidar_aligned \
  --guidance 1 \
  --prompt "" \
  --save_buffer
```

### LiDAR Data Requirements

**For Single Image Input:**
- `--lidar_path`: Path to a single `.npy` file containing the LiDAR depth map
- The LiDAR depth map should be in meters (0 = no depth data)

**For Video Input (Traditional or ViPE):**
- `--lidar_path`: Path to a single `.npy` file containing LiDAR depth maps for all frames
- The file should have shape `(T, H, W)` where T is the number of frames
- Each frame's LiDAR data is accessed as `lidar_data[frame_index]` with shape `(H, W)`
- Works with both traditional input folders and ViPE data

**LiDAR Depth Map Format:**
- **Single Image**: 2D numpy array with shape `(height, width)`
- **Video**: 3D numpy array with shape `(T, height, width)` where T is the number of frames
- Values in meters (0 = invalid/no depth data)
- Will be automatically resized to match target resolution if needed

### Benefits of LiDAR Alignment

- **Improved Accuracy**: Aligns predicted depth with ground truth LiDAR measurements
- **Better 3D Consistency**: More accurate depth leads to better 3D scene understanding
- **Sparse Data Handling**: Uses specialized sparse-aware resizing that preserves the exact sparse structure of LiDAR data without interpolation artifacts
- **Fallback Support**: Uses predicted depth where LiDAR data is not available

### Single-view Waymo video input (waypo_path)

Use a Waymo-formatted folder that contains `videos/`, `poses/`, `intrinsics/`, and `masks/` for a single camera view. The script will pick the only camera under `videos/` and render from that view.

```bash
pixi run run_render_only_video_generic -- \
  --waypo_path drone_samples/waymo_single_view_video \
  --trajectory_generation_method action_based_movement \
  --trajectory zoom_out \
  --camera_rotation no_rotation \
  --movement_distance 10 \
  --rendered_images_path waymo_single_view_warp_images_zoomout.pt \
  --rendered_masks_path waymo_single_view_warp_masks_zoomout.pt
```

```bash
pixi run run_diffusion_only_generic -- \
  --input_image_path  drone_samples/waymo_single_view_video/source_image.png \
  --rendered_images_path waymo_single_view_warp_images_zoomout.pt \
  --rendered_masks_path waymo_single_view_warp_masks_zoomout.pt \
  --video_save_name test_diffusion_only_waymo_single_view_zoomout \
  --guidance 1 \
  --prompt "" \
  --save_buffer
```


```bash
pixi run run_render_only_video_generic -- \
  --waypo_path drone_samples/waymo_single_view_video \
  --trajectory_generation_method action_based_movement \
  --trajectory zoom_in \
  --camera_rotation no_rotation \
  --movement_distance 3 \
  --rendered_images_path waymo_single_view_warp_images_zoomin.pt \
  --rendered_masks_path waymo_single_view_warp_masks_zoomin.pt
```


```bash
pixi run run_diffusion_only_generic -- \
  --input_image_path  drone_samples/waymo_single_view_video/source_image.png \
  --rendered_images_path waymo_single_view_warp_images_zoomin.pt \
  --rendered_masks_path waymo_single_view_warp_masks_zoomin.pt \
  --video_save_name test_diffusion_only_waymo_single_view_zoomin \
  --guidance 1 \
  --prompt "" \
  --save_buffer
```