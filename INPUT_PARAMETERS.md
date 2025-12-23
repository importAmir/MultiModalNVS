# Input Parameters Guide

This document explains all the input parameters for the different scripts in the GEN3C-Project.

## Table of Contents
1. [Common Arguments (add_common_arguments)](#common-arguments-add_common_arguments)
2. [Image Input Rendering Script (create_rendering_image_input.py)](#image-input-rendering-script-create_rendering_image_inputpy)
3. [Video Input Rendering Script (create_rendering_video_input.py)](#video-input-rendering-script-create_rendering_video_inputpy)
4. [Diffusion Only Script (diffusion_only.py)](#diffusion-only-script-diffusion_onlypy)

---

## Common Arguments (add_common_arguments)

These arguments are shared across multiple scripts and provide core functionality.

### Model Configuration
- **`--checkpoint_dir`** (str, default: "checkpoints")
  - Base directory containing model checkpoints
  - Example: `--checkpoint_dir "GEN3C/checkpoints"`

- **`--tokenizer_dir`** (str, default: "Cosmos-Tokenize1-CV8x8x8-720p")
  - Tokenizer weights directory relative to checkpoint_dir
  - Example: `--tokenizer_dir "Cosmos-Tokenize1-CV8x8x8-720p"`

### Output Configuration
- **`--video_save_name`** (str, default: "output")
  - Output filename for generating a single video
  - Example: `--video_save_name "my_generated_video"`

- **`--video_save_folder`** (str, default: "outputs/")
  - Output folder for generating a batch of videos
  - Example: `--video_save_folder "my_outputs/"`

### Text Prompts
- **`--prompt`** (str, optional)
  - Text prompt for generating a single video
  - Example: `--prompt "A beautiful sunset over the ocean"`

- **`--batch_input_path`** (str, optional)
  - Path to a JSONL file of input prompts for generating a batch of videos
  - Example: `--batch_input_path "prompts.jsonl"`

- **`--negative_prompt`** (str, default: long negative prompt)
  - Text prompt describing undesired attributes
  - Can be customized to avoid specific visual issues

### Generation Parameters
- **`--num_steps`** (int, default: 35)
  - Number of diffusion sampling steps
  - Higher values = better quality but slower generation
  - Example: `--num_steps 50`

- **`--guidance`** (float, default: 1)
  - Classifier-free guidance scale
  - Higher values = more adherence to prompt but less creative
  - **Formula**: `ε_θ(x_t, c) = ε_θ(x_t, ∅) + s * (ε_θ(x_t, c) - ε_θ(x_t, ∅))`
    - Where `s` is the guidance scale value
    - `ε_θ(x_t, c)` is the noise prediction with condition `c` (prompt)
    - `ε_θ(x_t, ∅)` is the noise prediction without condition (unconditional)
    - Higher `s` values amplify the difference between conditional and unconditional predictions
  - **Range**: Typically 1.0 to 10.0
    - `s = 1.0`: No guidance (unconditional generation)
    - `s = 2.0`: Moderate prompt adherence
    - `s = 5.0`: Strong prompt adherence
    - `s > 7.0`: Very strict prompt adherence (may reduce quality)
  - Example: `--guidance 2.5`

- **`--num_video_frames`** (int, default: 121)
  - Number of video frames to sample
  - Example: `--num_video_frames 121`

### Video Dimensions
- **`--height`** (int, default: 704)
  - Height of video to sample
  - Example: `--height 704`

- **`--width`** (int, default: 1280)
  - Width of video to sample
  - Example: `--width 1280`

- **`--fps`** (int, default: 24)
  - FPS of the sampled video
  - Example: `--fps 30`

### System Configuration
- **`--seed`** (int, default: 1)
  - Random seed for reproducibility
  - Example: `--seed 42`

- **`--num_gpus`** (int, default: 1)
  - Number of GPUs used to run inference in parallel
  - Example: `--num_gpus 2`

### Memory Optimization Flags
- **`--disable_prompt_upsampler`** (flag)
  - Disable prompt upsampling to save memory

- **`--offload_diffusion_transformer`** (flag)
  - Offload DiT after inference to free GPU memory

- **`--offload_tokenizer`** (flag)
  - Offload tokenizer after inference to free GPU memory

- **`--offload_text_encoder_model`** (flag)
  - Offload text encoder model after inference to free GPU memory

- **`--offload_prompt_upsampler`** (flag)
  - Offload prompt upsampler after inference to free GPU memory

- **`--offload_guardrail_models`** (flag)
  - Offload guardrail models after inference to free GPU memory

- **`--disable_guardrail`** (flag)
  - Disable guardrail models to save memory

- **`--disable_prompt_encoder`** (flag)
  - Disable prompt encoder to save memory, returns dummy embeddings instead

---

## Image Input Rendering Script (create_rendering_image_input.py)

This script creates rendered warp images and masks from a single input image.

### Required Arguments
- **`--input_image_path`** (str, required)
  - Path to the input image for conditioning
  - Example: `--input_image_path "GEN3C/assets/diffusion/000000.png"`

- **`--rendered_images_path`** (str, required)
  - Filename for the rendered warp images tensor
  - Example: `--rendered_images_path "rendered_warp_images.pt"`

- **`--rendered_masks_path`** (str, required)
  - Filename for the rendered warp masks tensor
  - Example: `--rendered_masks_path "rendered_warp_masks.pt"`

- **`--trajectory_generation_method`** (str, required)
  - Method for generating camera trajectory
  - Choices: `["action_based_movement", "pixel_focusing", "source_to_target_linear_interpolation"]`
  - Example: `--trajectory_generation_method "pixel_focusing"`

### Trajectory Parameters

#### Action-based Movement
- **`--trajectory`** (str, required for action_based_movement)
  - Camera movement direction
  - Choices: `["left", "right", "up", "down", "zoom_in", "zoom_out", "clockwise", "counterclockwise", "none"]`
  - Example: `--trajectory "zoom_out"`

- **`--camera_rotation`** (str, required for action_based_movement)
  - How the camera rotates during movement
  - Choices: `["center_facing", "no_rotation", "trajectory_aligned"]`
  - Example: `--camera_rotation "center_facing"`

- **`--movement_distance`** (float, required for action_based_movement)
  - How far the camera moves from the center
  - Example: `--movement_distance 0.5`

#### Pixel Focusing
- **`--target_pixel_x`** (int, required for pixel_focusing)
  - X-coordinate of the target pixel to focus on (0 to image width)
  - Example: `--target_pixel_x 640`

- **`--target_pixel_y`** (int, required for pixel_focusing)
  - Y-coordinate of the target pixel to focus on (0 to image height)
  - Example: `--target_pixel_y 352`

- **`--movement_ratio`** (float, required for pixel_focusing)
  - How much to move toward the target (0.0 = no movement, 1.0 = full movement)
  - Example: `--movement_ratio 0.75`

- **`--start_transition_frames`** (int, required for pixel_focusing)
  - Frame number when transition to target begins
  - Example: `--start_transition_frames 30`

- **`--end_transition_frames`** (int, required for pixel_focusing)
  - Frame number when transition to target completes
  - Example: `--end_transition_frames 120`

#### Source-to-Target Linear Interpolation
- **`--source_meta_path`** (str, required for source_to_target_linear_interpolation)
  - Path to source metadata JSON file containing starting camera pose
  - Example: `--source_meta_path "drone_samples/test_004/_1734644989.985579_metadata.json"`

- **`--target_meta_path`** (str, required for source_to_target_linear_interpolation)
  - Path to target metadata JSON file containing ending camera pose
  - Example: `--target_meta_path "drone_samples/test_004/_1734644976.704000_metadata.json"`

### Optional Arguments
- **`--rendered_tensor_dir`** (str, default: "../rendered_tensor_dir")
  - Directory to save rendered tensors
  - Example: `--rendered_tensor_dir "my_outputs"`

- **`--depth_estimator`** (str, default: "moge")
  - Depth estimation model to use
  - Choices: `["moge", "depthanythingv2"]`
  - Example: `--depth_estimator "depthanythingv2"`

- **`--noise_aug_strength`** (float, default: 0.0)
  - Strength of noise augmentation on warped frames
  - Example: `--noise_aug_strength 0.1`

- **`--filter_points_threshold`** (float, default: 0.05)
  - Threshold for filtering point continuity in warped images
  - Example: `--filter_points_threshold 0.1`

- **`--foreground_masking`** (flag)
  - Enable foreground masking for warped images

### DepthAnythingV2 Specific Parameters
- **`--default_fx`** (float, default: 739.75492315)
  - Default focal length x for DepthAnythingV2
  - Example: `--default_fx 800.0`

- **`--default_fy`** (float, default: 741.66148189)
  - Default focal length y for DepthAnythingV2
  - Example: `--default_fy 800.0`

- **`--default_cx`** (float, default: 605.94283506)
  - Default principal point x for DepthAnythingV2
  - Example: `--default_cx 640.0`

- **`--default_cy`** (float, default: 343.51934258)
  - Default principal point y for DepthAnythingV2
  - Example: `--default_cy 360.0`

---

## Video Input Rendering Script (create_rendering_video_input.py)

This script creates rendered warp images and masks from either a sequence of drone images or ViPE (Video Pose Engine) output data.

### Required Arguments
- **`--input_folder`** (str, required when not using ViPE)
  - Folder containing images and metadata files
  - Example: `--input_folder "drone_samples/test_006/doer"`

- **`--vipe_path`** (str, required when using ViPE)
  - Path to ViPE clip root directory or mp4 file under rgb/
  - Example: `--vipe_path "../PoseDepthEstimation/vipe_results"`

- **`--vipe_starting_frame_idx`** (int, default: 0)
  - Starting frame index within the ViPE video
  - Example: `--vipe_starting_frame_idx 10`

- **`--rendered_images_path`** (str, required)
  - Filename for the rendered warp images tensor
  - Example: `--rendered_images_path "rendered_warp_images_zoomout_doer.pt"`

- **`--rendered_masks_path`** (str, required)
  - Filename for the rendered warp masks tensor
  - Example: `--rendered_masks_path "rendered_warp_masks_zoomout_doer.pt"`

### Trajectory Parameters

#### Action-based Movement
- **`--trajectory_generation_method`** (str, required)
  - Method for generating camera trajectory
  - Choices: `["action_based_movement", "pixel_focusing", "source_to_target_linear_interpolation"]`
  - Example: `--trajectory_generation_method "action_based_movement"`

- **`--trajectory`** (str, required for action_based_movement)
  - Camera movement direction
  - Choices: `["left", "right", "up", "down", "zoom_in", "zoom_out", "clockwise", "counterclockwise", "none"]`
  - Example: `--trajectory "zoom_out"`

- **`--camera_rotation`** (str, required for action_based_movement)
  - How the camera rotates during movement
  - Choices: `["center_facing", "no_rotation", "trajectory_aligned"]`
  - Example: `--camera_rotation "center_facing"`

- **`--movement_distance`** (float, required for action_based_movement)
  - How far the camera moves from the center
  - Example: `--movement_distance 0.5`

#### Pixel Focusing
- **`--target_pixel_x`** (int, required for pixel_focusing)
  - X-coordinate of the target pixel to focus on
  - Example: `--target_pixel_x 1051`

- **`--target_pixel_y`** (int, required for pixel_focusing)
  - Y-coordinate of the target pixel to focus on
  - Example: `--target_pixel_y 392`

- **`--movement_ratio`** (float, required for pixel_focusing)
  - How much to move toward the target (0.0 = no movement, 1.0 = full movement)
  - Example: `--movement_ratio 0.75`

- **`--start_transition_frames`** (int, required for pixel_focusing)
  - Frame number when transition to target begins
  - Example: `--start_transition_frames 30`

- **`--end_transition_frames`** (int, required for pixel_focusing)
  - Frame number when transition to target completes
  - Example: `--end_transition_frames 120`

#### Source-to-Target Linear Interpolation
- **`--source_meta_path`** (str, required for source_to_target_linear_interpolation)
  - Path to source metadata JSON file containing starting camera pose
  - Example: `--source_meta_path "drone_samples/test_006/watcher/front_camera_1753232045.812008_metadata.json"`

- **`--target_meta_path`** (str, required for source_to_target_linear_interpolation)
  - Path to target metadata JSON file containing ending camera pose
  - Example: `--target_meta_path "drone_samples/test_006/doer/front_camera_1753232074.463422_metadata.json"`

### Optional Arguments
- **`--rendered_tensor_dir`** (str, default: "../rendered_tensor_dir")
  - Directory to save rendered tensors
  - Example: `--rendered_tensor_dir "my_outputs"`

- **`--noise_aug_strength`** (float, default: 0.0)
  - Strength of noise augmentation on warped frames
  - Example: `--noise_aug_strength 0.1`

- **`--filter_points_threshold`** (float, default: 0.05)
  - Threshold for filtering point continuity in warped images
  - Example: `--filter_points_threshold 0.1`

- **`--foreground_masking`** (flag)
  - Enable foreground masking for warped images

### Gimbal Control Parameters
- **`--gimbal_pitch`** (float, optional)
  - Camera pitch angle in degrees
  - Example: `--gimbal_pitch -90` (camera pointing downward)

- **`--gimbal_yaw`** (float, optional)
  - Camera yaw angle in degrees
  - Example: `--gimbal_yaw 0` (no horizontal rotation)

- **`--gimbal_roll`** (float, optional)
  - Camera roll angle in degrees
  - Example: `--gimbal_roll 0` (no roll rotation)

### DepthAnythingV2 Parameters
- **`--default_fx`** (float, default: 739.75492315)
  - Default focal length x for DepthAnythingV2
  - Example: `--default_fx 800.0`

- **`--default_fy`** (float, default: 741.66148189)
  - Default focal length y for DepthAnythingV2
  - Example: `--default_fy 800.0`

- **`--default_cx`** (float, default: 605.94283506)
  - Default principal point x for DepthAnythingV2
  - Example: `--default_cx 640.0`

- **`--default_cy`** (float, default: 343.51934258)
  - Default principal point y for DepthAnythingV2
  - Example: `--default_cy 360.0`

---

## Diffusion Only Script (diffusion_only.py)

This script generates videos using pre-rendered warp images and masks.

### Required Arguments
- **`--input_image_path`** (str, required when not using ViPE)
  - Path to the input image for conditioning
  - Example: `--input_image_path "drone_samples/test_006/doer/front_camera_1753232045.813325.jpg"`

- **`--vipe_path`** (str, required when using ViPE)
  - Path to ViPE clip root directory or mp4 file under rgb/
  - Example: `--vipe_path "../PoseDepthEstimation/vipe_results"`

- **`--vipe_starting_frame_idx`** (int, default: 0)
  - Starting frame index within the ViPE video
  - Example: `--vipe_starting_frame_idx 10`

**Note**: When using `--vipe_path`, the script automatically loads RGB frames, depth maps, camera poses, and intrinsics from the ViPE output structure. The first frame from the ViPE data is used as the input image for conditioning.

- **`--rendered_images_path`** (str, required)
  - Path to the rendered warp images tensor
  - Example: `--rendered_images_path "rendered_warp_images_zoomout_doer.pt"`

- **`--rendered_masks_path`** (str, required)
  - Path to the rendered warp masks tensor
  - Example: `--rendered_masks_path "rendered_warp_masks_zoomout_doer.pt"`

### Optional Arguments
- **`--save_buffer`** (flag)
  - Enable saving intermediate warped images
  - Example: `--save_buffer True`

- **`--video_save_name`** (str, optional)
  - Custom name for the output video
  - Example: `--video_save_name "my_custom_video_name"`

### Note
This script inherits all the common arguments from `add_common_arguments`, including:
- Model configuration (`--checkpoint_dir`, `--tokenizer_dir`)
- Generation parameters (`--num_steps`, `--guidance`, `--seed`)
- Video dimensions (`--height`, `--width`, `--fps`)
- Memory optimization flags
- Text prompts (`--prompt`, `--negative_prompt`)

---
