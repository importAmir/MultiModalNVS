#!/usr/bin/env python3
"""
Create Rendering Script for GEN3C (Multiview Waymo Video Input)
This script mirrors create_rendering_multiview_waymo_image_input.py structure and flow,
but reads multiview video data from a Waymo-style folder containing:
  videos/{camera}.npy         (T, H, W, 3) video sequences
  poses/{camera}.npy          (T, 4, 4) pose sequences
  intrinsics/{camera}.npy     (3, 3) - same for all frames
  masks/{camera}.npy          (T, H, W) mask sequences
  lidars/{camera}.npy         (T, H, W) LiDAR depth sequences
"""

import argparse
import os
import cv2
import torch
import numpy as np
from typing import List
import sys
from pathlib import Path
from tqdm import tqdm

# Import necessary modules from the original codebase
from cosmos_predict1.diffusion.inference.cache4d_multiview import Cache4D_BufferSelector
from cosmos_predict1.diffusion.inference.camera_utils import generate_camera_trajectory
from cosmos_predict1.diffusion.inference.camera_sequence_generation import (
    generate_source_to_target_trajectory, 
    generate_pixel_focused_trajectory
)
from cosmos_predict1.utils import log, misc
from cosmos_predict1.diffusion.inference.inference_utils import (
    add_common_arguments,
)

# Alignment helper (for optional depth alignment to LiDAR)
from cosmos_predict1.diffusion.inference.camera_utils import _align_inv_depth_to_depth
from cosmos_predict1.diffusion.inference.create_rendering_multiview_waymo_image_input import _resize_sparse_depth

torch.enable_grad(False)

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the rendering script (mirrors image_input)."""
    parser = argparse.ArgumentParser(description="Create rendered warp videos and masks from Waymo multiview video folder")
    # Add common arguments
    add_common_arguments(parser)

    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Waymo-exported folder containing videos/, poses/, intrinsics/, masks/, lidars/"
    )

    parser.add_argument(
        "--trajectory",
        type=str,
        choices=[
            "left", "right", "up", "down", "zoom_in", "zoom_out",
            "clockwise", "counterclockwise", "none"
        ],
        default=None,
        help="Select a trajectory type from the available options"
    )
    parser.add_argument(
        "--camera_rotation",
        type=str,
        choices=["center_facing", "no_rotation", "trajectory_aligned"],
        default=None,
        help="Controls camera rotation during movement"
    )
    parser.add_argument(
        "--movement_distance",
        type=float,
        default=None,
        help="Distance of the camera from the center of the scene"
    )

    parser.add_argument(
        "--noise_aug_strength",
        type=float,
        default=0.0,
        help="Strength of noise augmentation on warped frames"
    )

    parser.add_argument(
        "--filter_points_threshold",
        type=float,
        default=0.05,
        help="If set, filter the points continuity of the warped images.",
    )

    parser.add_argument(
        "--foreground_masking",
        action="store_true",
        help="If set, use foreground masking for the warped images.",
    )
    
    parser.add_argument(
        "--rendered_tensor_dir",
        type=str,
        default="../rendered_tensor_dir",
        help="Directory to save rendered tensors"
    )
    
    parser.add_argument(
        "--rendered_images_path",
        type=str,
        required=True,
        help="Filename for the rendered warp images tensor"
    )
    
    parser.add_argument(
        "--rendered_masks_path",
        type=str,
        required=True,
        help="Filename for the rendered warp masks tensor"
    )
    
    parser.add_argument(
        "--trajectory_generation_method",
        type=str,
        choices=["action_based_movement", "pixel_focusing", "source_to_target_linear_interpolation"],
        required=True,
        help="Method for generating camera trajectory: action_based_movement, pixel_focusing, or source_to_target_linear_interpolation"
    )
    
    parser.add_argument(
        "--target_pixel_x",
        type=int,
        default=None,
        help="Target pixel X coordinate for pixel focusing method"
    )
    
    parser.add_argument(
        "--target_pixel_y",
        type=int,
        default=None,
        help="Target pixel Y coordinate for pixel focusing method"
    )
    
    parser.add_argument(
        "--movement_ratio",
        type=float,
        default=None,
        help="Movement ratio (0-1) for pixel focusing method"
    )
    
    parser.add_argument(
        "--start_transition_frames",
        type=int,
        default=None,
        help="Frame number to start transitioning to target for pixel focusing method"
    )
    
    parser.add_argument(
        "--end_transition_frames",
        type=int,
        default=None,
        help="Frame number to end transitioning to target for pixel focusing method"
    )
    
    parser.add_argument(
        "--source_pose_path",
        type=str,
        default=None,
        help="Source pose path (for source_to_target_linear_interpolation); for Waymo, pass poses/NAME.npy"
    )
    
    parser.add_argument(
        "--target_pose_path",
        type=str,
        default=None,
        help="Target pose path (for source_to_target_linear_interpolation); for Waymo, pass poses/NAME.npy"
    )

    # Waymo-specific optional alignment
    parser.add_argument(
        "--align_depth_with_lidar",
        action="store_true",
        help="If set, align predicted depth to LiDAR depth using valid LiDAR mask"
    )
    
    # Missing arguments that are referenced in the code
    parser.add_argument(
        "--reference_frame",
        type=int,
        default=0,
        help="Index of the reference frame to use for trajectory generation (default: 0)"
    )
    
    parser.add_argument(
        "--frame_buffer_max",
        type=int,
        default=2,
        help="Maximum number of frames to keep in buffer for Cache4D_BufferSelector"
    )
    
    return parser


def _list_cameras(waymo_root: Path) -> List[str]:
    videos_dir = waymo_root / "videos"
    if not videos_dir.exists():
        raise FileNotFoundError(f"Missing videos folder: {videos_dir}")
    cams = sorted([p.stem for p in videos_dir.glob("*.npy")])
    if not cams:
        raise RuntimeError(f"No videos found in {videos_dir}")
    return cams


def create_rendering(args) -> None:
    """
    Create rendering from Waymo multiview video input folder.
    Mirrors the flow of create_rendering_multiview_waymo_image_input.py but for video data.
    """
    misc.set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup DepthAnythingV2 (import location mirrors image_input)
    # Find repo root: script may be at repo root or in GEN3C/cosmos_predict1/diffusion/inference/
    # Try to find Depth-Estimation directory by going up from script location
    script_path = Path(__file__).resolve()
    repo_root = None
    for parent in [script_path.parent] + list(script_path.parents):
        if (parent / "Depth-Estimation" / "Depth-Anything-V2").exists():
            repo_root = parent
            break
    if repo_root is None:
        # Fallback: assume repo root is 5 levels up from script (when in GEN3C)
        repo_root = script_path.parent.parent.parent.parent.parent
    sys.path.insert(0, str(repo_root / "Depth-Estimation" / "Depth-Anything-V2" / "metric_depth"))
    from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    encoder = 'vitl'
    max_depth = 80
    DepthAnythingV2_checkpoint_path = repo_root / 'Depth-Estimation' / 'Depth-Anything-V2' / 'checkpoints' / f'depth_anything_v2_metric_hypersim_{encoder}.pth'
    
    dav2_model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    dav2_model.load_state_dict(torch.load(DepthAnythingV2_checkpoint_path, map_location=device))
    dav2_model.to(device).eval()
    
    log.info(f"Loading DepthAnythingV2 model from {DepthAnythingV2_checkpoint_path}")
    
    # Load input Waymo folder
    input_folder = Path(args.input_folder)
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder {input_folder} does not exist")

    cameras = _list_cameras(input_folder)
    N = len(cameras)
    log.info(f"Found {N} camera views in {input_folder}")
    
    # Initialize tensors using args.num_video_frames instead of assuming all videos have same length
    input_images = torch.zeros(N, args.num_video_frames, 3, args.height, args.width)  # [N, T, C, H, W]
    input_depths = torch.zeros(N, args.num_video_frames, 1, args.height, args.width)  # [N, T, 1, H, W]
    input_masks = torch.ones(N, args.num_video_frames, 1, args.height, args.width)    # [N, T, 1, H, W]
    input_w2cs = torch.zeros(N, args.num_video_frames, 4, 4)                          # [N, T, 4, 4]
    input_intrinsics = torch.zeros(N, args.num_video_frames, 3, 3)  # [N, T, 3, 3] - same intrinsics repeated for all frames

    log.info("Starting depth estimation and data assembly from Waymo video folder...")
    with torch.no_grad():
        for i, cam in enumerate(tqdm(cameras, desc="Processing camera views")):
            # Load video data
            video_path = input_folder / "videos" / f"{cam}.npy"
            video_data = np.load(video_path)  # [T, H, W, 3]
            
            # Compute dimensions for this specific video
            T, H_orig, W_orig, C = video_data.shape
            log.info(f"Camera {cam} video dimensions: {T} frames, {H_orig}x{W_orig}, {C} channels")
            assert T == args.num_video_frames, f"Camera {cam} has {T} frames but expected {args.num_video_frames}"
            
            # Load pose sequence
            poses_path = input_folder / "poses" / f"{cam}.npy"
            poses_data = np.load(poses_path)  # [T, 4, 4]
            assert poses_data.shape == (T, 4, 4), f"Pose shape is {poses_data.shape}, expected (T, 4, 4)"

            input_w2cs[i] = torch.from_numpy(poses_data.astype(np.float32))

            # Load intrinsics (same for all frames)
            intrinsics_path = input_folder / "intrinsics" / f"{cam}.npy"
            K = np.load(intrinsics_path).astype(np.float32)  # [3, 3]
            assert K.shape == (3, 3), f"Intrinsics shape is {K.shape}, expected (3, 3)"
            
            # Scale intrinsics based on resizing for this specific video
            scale_h = args.height / H_orig
            scale_w = args.width / W_orig
            K[0, 0] *= scale_w  # fx
            K[1, 1] *= scale_h  # fy
            K[0, 2] *= scale_w  # cx
            K[1, 2] *= scale_h  # cy
            input_intrinsics[i] = torch.from_numpy(K).repeat(args.num_video_frames, 1, 1)

            # Load masks once for this camera (outside frame loop for efficiency)
            masks_path = input_folder / "masks" / f"{cam}.npy"
            assert masks_path.exists(), f"Masks path {masks_path} does not exist"
            
            masks_data = np.load(masks_path)  # [T, H, W]
            assert masks_data.shape[0] == T, f"Mask frames {masks_data.shape[0]} != video frames {T}"
            # Resize all masks at once
            masks_resized = np.zeros((T, args.height, args.width), dtype=np.float32)
            for t in range(T):
                if masks_data[t].shape != (args.height, args.width):
                    masks_resized[t] = cv2.resize(masks_data[t], (args.width, args.height), interpolation=cv2.INTER_NEAREST)
                else:
                    masks_resized[t] = masks_data[t]
            input_masks[i] = torch.from_numpy(masks_resized).unsqueeze(1)  # [T, 1, H, W]

            # Process each frame
            for t in range(T):
                # Get frame
                frame_bgr = video_data[t]  # [H, W, 3] in BGR
                frame_resized = cv2.resize(frame_bgr, (args.width, args.height))
                
                # DepthAnythingV2 depth estimation
                depth = dav2_model.infer_image(frame_resized)

                # Optional LiDAR alignment
                if args.align_depth_with_lidar:
                    lidar_path = input_folder / "lidars" / f"{cam}.npy"
                    if lidar_path.exists():
                        lidar_data = np.load(lidar_path)  # [T, H, W]
                        lidar_frame = lidar_data[t]  # [H, W]
                        if lidar_frame.shape != (args.height, args.width):
                            # Use sparse-aware resizing for LiDAR data
                            lidar_frame = _resize_sparse_depth(lidar_frame, (args.height, args.width))
                        pred_t = torch.from_numpy(depth).float()
                        lidar_t = torch.from_numpy(lidar_frame).float()
                        target_mask = lidar_t > 0
                        depth_aligned = _align_inv_depth_to_depth(
                            1.0 / torch.clamp_min(pred_t, 1e-6),
                            lidar_t,
                            target_mask=target_mask
                        )
                        depth = depth_aligned.numpy()

                # Convert BGR->RGB and normalize
                frame_rgb = frame_resized[..., [2, 1, 0]].copy()
                frame_tensor = torch.from_numpy(frame_rgb).float()
                frame_tensor = frame_tensor.permute(2, 0, 1)
                frame_tensor = frame_tensor / 127.5 - 1.0

                # Store
                input_images[i, t] = frame_tensor
                input_depths[i, t, 0] = torch.from_numpy(depth).float()

            # Clear intermediates
            del video_data, poses_data, masks_resized
            torch.cuda.empty_cache()

    # Model cleanup
    del dav2_model
    torch.cuda.empty_cache()
    log.info("Cleared depth model and intermediates from memory")

    # Move to device
    input_images = input_images.to(device)
    input_depths = input_depths.to(device)
    input_masks = input_masks.to(device)
    input_w2cs = input_w2cs.to(device)
    input_intrinsics = input_intrinsics.to(device)

    # Add batch dimension for Cache4D_BufferSelector: [1, N, F, C, H, W]
    input_images_bNFCHW = input_images.unsqueeze(0)
    input_depths_bNF1HW = input_depths.unsqueeze(0)
    input_masks_bNF1HW = input_masks.unsqueeze(0)
    input_w2cs_bNF44 = input_w2cs.unsqueeze(0)
    input_intrinsics_bNF33 = input_intrinsics.unsqueeze(0)

    # Create Cache4D_BufferSelector (mirrors image_input but for video)
    cache = Cache4D_BufferSelector(
        frame_buffer_max=args.frame_buffer_max,
        input_image=input_images_bNFCHW,      # [1, N, F, C, H, W]
        input_depth=input_depths_bNF1HW,      # [1, N, F, 1, H, W]
        input_mask=input_masks_bNF1HW,        # [1, N, F, 1, H, W]
        input_w2c=input_w2cs_bNF44,           # [1, N, F, 4, 4]
        input_intrinsics=input_intrinsics_bNF33,  # [1, N, F, 3, 3]
        filter_points_threshold=args.filter_points_threshold,
        input_format=["B", "N", "F", "C", "H", "W"],
        foreground_masking=args.foreground_masking,
    )
    
    # Generate trajectory (same structure as image_input)
    if args.trajectory_generation_method == "action_based_movement":
        assert args.trajectory in ["left", "right", "up", "down", "zoom_in", "zoom_out", "clockwise", "counterclockwise", "none"]
        assert args.camera_rotation in ["center_facing", "no_rotation", "trajectory_aligned"]
        assert args.movement_distance is not None
        
        if args.reference_frame >= N or args.reference_frame < 0:
            raise ValueError(f"Reference frame index {args.reference_frame} is out of range. Must be between 0 and {N-1}")
        
        initial_w2c = input_w2cs[args.reference_frame, 0]  # Use first frame of reference camera
        initial_intrinsics = input_intrinsics[args.reference_frame, 0]  # Use first frame intrinsics
        
        try:
            generated_w2cs, generated_intrinsics = generate_camera_trajectory(
                trajectory_type=args.trajectory,
                initial_w2c=initial_w2c,
                initial_intrinsics=initial_intrinsics,
                num_frames=args.num_video_frames,
                movement_distance=args.movement_distance,
                camera_rotation=args.camera_rotation,
                center_depth=1.0,
                device=device.type,
            )
        except (ValueError, NotImplementedError) as e:
            log.critical(f"Failed to generate trajectory: {e}")
            raise
    
    elif args.trajectory_generation_method == "pixel_focusing":
        assert args.target_pixel_x is not None
        assert args.target_pixel_y is not None
        assert args.movement_ratio is not None
        assert args.start_transition_frames is not None
        assert args.end_transition_frames is not None

        if args.reference_frame >= N or args.reference_frame < 0:
            raise ValueError(f"Reference frame index {args.reference_frame} is out of range. Must be between 0 and {N-1}")

        initial_w2c = input_w2cs[args.reference_frame, 0]  # Use first frame of reference camera
        initial_intrinsics = input_intrinsics[args.reference_frame, 0]  # Use first frame intrinsics

        try:
            generated_w2cs, generated_intrinsics = generate_pixel_focused_trajectory(
                initial_w2c=initial_w2c,
                initial_intrinsics=initial_intrinsics,
                target_pixel=(args.target_pixel_x, args.target_pixel_y),
                num_frames=args.num_video_frames,
                movement_ratio=args.movement_ratio,
                start_transition_frames=args.start_transition_frames,
                end_transition_frames=args.end_transition_frames,
                depth_map=input_depths[args.reference_frame, 0],  # [1, H, W] - first frame of reference camera
                device=device.type,
            )
        except (ValueError, NotImplementedError) as e:
            log.critical(f"Failed to generate trajectory: {e}")
            raise

    elif args.trajectory_generation_method == "source_to_target_linear_interpolation":
        assert args.source_pose_path is not None
        assert args.target_pose_path is not None
        
        # For Waymo, source/target are pose .npy paths (4x4 W2C); load them directly
        source_pose = np.load(args.source_pose_path).astype(np.float32)
        target_pose = np.load(args.target_pose_path).astype(np.float32)
        
        source_pose_tensor = torch.tensor(source_pose, device=device, dtype=torch.float32)
        target_pose_tensor = torch.tensor(target_pose, device=device, dtype=torch.float32)
        
        generated_w2cs = generate_source_to_target_trajectory(
            source_w2c = source_pose_tensor,
            target_w2c = target_pose_tensor, 
            num_frames = args.num_video_frames,
            start_transition_frames = args.start_transition_frames,
            end_transition_frames = args.end_transition_frames,
            device = device,
        )
        generated_intrinsics = input_intrinsics[0].unsqueeze(0)  # [1, T, 3, 3]
    
    else:
        raise ValueError(f"Unknown trajectory generation method: {args.trajectory_generation_method}")
    
    # Render using Cache4D_BufferSelector
    rendered_warp_images, rendered_warp_masks = cache.render_cache(
        generated_w2cs,
        generated_intrinsics,
    )
    
    log.info(f"Rendered warp images shape: {rendered_warp_images.shape}")
    log.info(f"Rendered warp masks shape: {rendered_warp_masks.shape}")
    
    # Save rendered tensors
    log.info("Saving rendered tensors...")
    
    warp_images_path = os.path.join(args.rendered_tensor_dir, args.rendered_images_path)
    torch.save(rendered_warp_images, warp_images_path)
    log.info(f"Saved warp images to: {warp_images_path}")
    
    warp_masks_path = os.path.join(args.rendered_tensor_dir, args.rendered_masks_path)
    torch.save(rendered_warp_masks, warp_masks_path)
    log.info(f"Saved warp masks to: {warp_masks_path}")
    
    return


def main():
    """Main function to run the rendering creation (mirrors image_input)."""
    parser = create_parser()
    args = parser.parse_args()
    if args.prompt is None:
        args.prompt = ""
    args.disable_guardrail = True
    args.disable_prompt_upsampler = True
    
    # Handle relative paths (mirror behavior)
    if not os.path.isabs(args.input_folder):
        args.input_folder = os.path.join("..", args.input_folder)
    
    if args.source_pose_path is not None and not os.path.isabs(args.source_pose_path):
        args.source_pose_path = os.path.join("..", args.source_pose_path)
    if args.target_pose_path is not None and not os.path.isabs(args.target_pose_path):
        args.target_pose_path = os.path.join("..", args.target_pose_path)

    os.makedirs(args.rendered_tensor_dir, exist_ok=True)
    
    create_rendering(args)


if __name__ == "__main__":
    main()