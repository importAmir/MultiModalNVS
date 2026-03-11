#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn.functional as F
import sys
import json
import os
from pathlib import Path
from tqdm import tqdm
import cv2
import argparse

from cosmos_predict1.diffusion.inference.cache_3d import Cache4D
from cosmos_predict1.diffusion.inference.camera_utils import generate_camera_trajectory
from cosmos_predict1.diffusion.inference.camera_sequence_generation import (
    generate_source_to_target_trajectory,
    load_map_to_camera_tf_matrix,
    generate_pixel_focused_trajectory,
    plot_camera_trajectory_comparison,
    plot_depth_statistics,
    resample_w2c_sequence,
)
from cosmos_predict1.utils import log, misc
from cosmos_predict1.diffusion.inference.inference_utils import (
    add_common_arguments,
)
from cosmos_predict1.diffusion.inference.vipe_utils import load_vipe_data

# Alignment helper (for optional depth alignment to LiDAR)
from cosmos_predict1.diffusion.inference.camera_utils import _align_inv_depth_to_depth
from cosmos_predict1.diffusion.inference.create_rendering_multiview_waymo_image_input import _resize_sparse_depth

torch.enable_grad(False)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the rendering script."""
    parser = argparse.ArgumentParser(description="Create rendered warp images and masks from input image (Image Input Version)")
    # Add common arguments
    add_common_arguments(parser)

    parser.add_argument(
        "--input_folder",
        type=str,
        required=False,
        help="Folder containing images and metadata files (required if not using --vipe_path)"
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
        choices=["action_based_movement", "pixel_focusing", "source_to_target_linear_interpolation", "target_folder_trajectory", "pose_sequence"],
        required=True,
        help="Method for generating camera trajectory: action_based_movement (default), pixel_focusing, source_to_target_linear_interpolation, target_folder_trajectory, or pose_sequence"
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
        "--default_fx",
        type=float,
        default=739.75492315,
        help="Default focal length x for DepthAnythingV2 (default: 739.75492315)"
    )
    
    parser.add_argument(
        "--default_fy",
        type=float,
        default=741.66148189,
        help="Default focal length y for DepthAnythingV2 (default: 741.66148189)"
    )
    
    parser.add_argument(
        "--default_cx",
        type=float,
        default=605.94283506,
        help="Default principal point x for DepthAnythingV2 (default: 605.94283506)"
    )
    
    parser.add_argument(
        "--default_cy",
        type=float,
        default=343.51934258,
        help="Default principal point y for DepthAnythingV2 (default: 343.51934258)"
    )
    
    parser.add_argument(
        "--source_pose_path",
        type=str,
        default=None,
        help="Optional: path to a NumPy .npy file containing a 4x4 world-to-camera pose for the source view. If provided along with --target_pose_path, these will be used instead of metadata."
    )
    
    parser.add_argument(
        "--target_pose_path",
        type=str,
        default=None,
        help="Optional: path to a NumPy .npy file containing a 4x4 world-to-camera pose for the target view. If provided along with --source_pose_path, these will be used instead of metadata."
    )

    parser.add_argument(
        "--pose_sequence_path",
        type=str,
        default=None,
        help="Path to a NumPy .npy file containing a sequence of world-to-camera poses with shape (T, 4, 4). Used when --trajectory_generation_method=pose_sequence.",
    )

    parser.add_argument(
        "--source_meta_path",
        type=str,
        default=None,
        help="Path to source metadata JSON file for source_to_target_linear_interpolation method"
    )
    
    parser.add_argument(
        "--target_meta_path",
        type=str,
        default=None,
        help="Path to target metadata JSON file for source_to_target_linear_interpolation method"
    )

    parser.add_argument(
        "--target_meta_folder",
        type=str,
        default=None,
        help="Path to folder containing target metadata files for target_folder_trajectory method"
    )

    parser.add_argument(
        "--vipe_path",
        type=str,
        default=None,
        help="Optional: path to VIPE clip root or the mp4 file under rgb/. If set, load VIPE-formatted data directly instead of using input_folder."
    )
    
    # Waymo single-view video folder loader path (reuses multiview format but picks one view)
    parser.add_argument(
        "--waypo_path",
        type=str,
        default=None,
        help="Optional: path to a Waymo-formatted folder with videos/, poses/, intrinsics/, masks/ for a single view. If set, use this instead of --input_folder."
    )
    
    parser.add_argument(
        "--vipe_starting_frame_idx",
        type=int,
        default=0,
        help="Starting frame index within the VIPE rgb mp4 to use as the reference frame."
    )

    parser.add_argument(
        '--gimbal_pitch', 
        type=float, 
        default=None,
        help='Gimbal pitch angle in degrees (default: 0)'
    )
    
    parser.add_argument(
        '--gimbal_yaw', 
        type=float, 
        default=None,
        help='Gimbal yaw angle in degrees (default: 0.0)'
    )
    
    parser.add_argument(
        '--gimbal_roll', 
        type=float, 
        default=None,
        help='Gimbal roll angle in degrees (default: 0.0)'
    )
    
    # LiDAR depth alignment
    parser.add_argument(
        "--align_depth_with_lidar",
        action="store_true",
        help="If set, align predicted depth to LiDAR depth using valid LiDAR mask for each frame"
    )
    
    parser.add_argument(
        "--lidar_path",
        type=str,
        default=None,
        help="Path to LiDAR depth maps (.npy file) for depth alignment. Required when --align_depth_with_lidar is set. Should be a single file with shape (T, H, W) where T is the number of frames."
    )

    return parser

def get_image_and_metadata_pairs(input_folder):
    """
    Get sorted pairs of (image_path, metadata_path) from the input folder.
    
    Args:
        input_folder: Path to folder containing images and metadata
        
    Returns:
        List of tuples: [(image_path, metadata_path), ...]
    """
    input_folder = Path(input_folder)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(input_folder.glob(f'*{ext}'))
        image_files.extend(input_folder.glob(f'*{ext.upper()}'))
    
    if not image_files:
        raise FileNotFoundError(f"No image files found in {input_folder}")
    
    # Sort images alphabetically
    image_files.sort()
    
    # Create pairs with metadata
    pairs = []
    for img_path in image_files:
        # Create metadata path by adding _metadata.json
        metadata_path = img_path.with_suffix('').with_name(f"{img_path.stem}_metadata.json")
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found for {img_path.name}: {metadata_path}")
        
        pairs.append((img_path, metadata_path))
    
    return pairs


def create_rendering(args):
    misc.set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validate input arguments: exactly one input source must be provided
    provided_inputs = [
        args.vipe_path is not None,
        args.waypo_path is not None,
        args.input_folder is not None,
    ]
    if sum(provided_inputs) != 1:
        raise ValueError("Exactly one of --vipe_path, --waypo_path, or --input_folder must be provided")
    

    # Only require default intrinsics for traditional input folder approach
    if args.input_folder is not None:
        assert args.default_fx is not None
        assert args.default_fy is not None
        assert args.default_cx is not None
        assert args.default_cy is not None

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
    
    from depth_anything_v2.dpt import DepthAnythingV2
    
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

    # Check if using ViPE data, Waymo single-view data, or traditional input folder
    if args.vipe_path is not None:
        log.info(f"Loading ViPE data from: {args.vipe_path}")
        
        # Load ViPE data directly
        try:
            (
                image_bchw_float,
                depth_b1hw,
                mask_b1hw,
                initial_w2c_b44,
                intrinsics_b33,
            ) = load_vipe_data(
                vipe_root_or_mp4=args.vipe_path,
                starting_frame_idx=args.vipe_starting_frame_idx,
                resize_hw=(args.height, args.width),
                crop_hw=(args.height, args.width),
                num_frames=args.num_video_frames,
            )
            
            # Optional LiDAR alignment for ViPE data
            if args.align_depth_with_lidar:
                if args.lidar_path is None:
                    raise ValueError("--lidar_path is required when --align_depth_with_lidar is set")
                
                lidar_path = Path(args.lidar_path)
                if not lidar_path.exists():
                    raise FileNotFoundError(f"LiDAR file not found: {lidar_path}")
                
                lidar_data = np.load(lidar_path).astype(np.float32)
                if len(lidar_data.shape) != 3:
                    raise ValueError(f"LiDAR data should have shape (T, H, W), got {lidar_data.shape}")
                
                T_lidar, H_lidar, W_lidar = lidar_data.shape
                T_vipe = depth_b1hw.shape[0]
                if T_lidar != T_vipe:
                    raise ValueError(f"LiDAR data has {T_lidar} frames but ViPE has {T_vipe} frames")
                
                log.info(f"Loaded LiDAR data with shape {lidar_data.shape} for ViPE alignment")
                
                # Align each frame's depth with corresponding LiDAR data
                with torch.no_grad():
                    for i in range(T_vipe):
                        # Get LiDAR depth for this frame
                        lidar_depth = lidar_data[i]  # Shape: (H, W)
                        
                        if lidar_depth.shape != (args.height, args.width):
                            # Use sparse-aware resizing for LiDAR data
                            lidar_depth = _resize_sparse_depth(lidar_depth, (args.height, args.width))
                        
                        # Align predicted depth with LiDAR depth
                        pred_depth = depth_b1hw[i, 0].cpu().numpy()  # Shape: (H, W)
                        pred_t = torch.from_numpy(pred_depth).float()
                        lidar_t = torch.from_numpy(lidar_depth).float()
                        target_mask = lidar_t > 0
                        
                        depth_aligned = _align_inv_depth_to_depth(
                            1.0 / torch.clamp_min(pred_t, 1e-6),
                            lidar_t,
                            target_mask=target_mask
                        )
                        depth_b1hw[i, 0] = depth_aligned.to(device)
                
                log.info("Completed LiDAR alignment for ViPE data")
            
            # Move to device
            image_bchw_float = image_bchw_float.to(device)
            depth_b1hw = depth_b1hw.to(device)
            mask_b1hw = mask_b1hw.to(device)
            initial_w2c_b44 = initial_w2c_b44.to(device)
            intrinsics_b33 = intrinsics_b33.to(device)
            
            log.info(f"Successfully loaded ViPE data: {image_bchw_float.shape}, {depth_b1hw.shape}")
            
        except Exception as e:
            log.critical(f"Failed to load ViPE data: {e}")
            return
            
    elif args.waypo_path is not None:
        # Waymo single-view loader using multiview folder structure
        input_folder = Path(args.waypo_path)
        if not input_folder.exists():
            raise FileNotFoundError(f"Input folder {input_folder} does not exist")

        videos_dir = input_folder / "videos"
        if not videos_dir.exists():
            raise FileNotFoundError(f"Missing videos folder: {videos_dir}")
        camera_files = sorted(videos_dir.glob("*.npy"))
        if not camera_files:
            raise RuntimeError(f"No videos found in {videos_dir}")
        
        assert len(camera_files) == 1, "Waymo single-view loader only supports one camera view"

        # Pick the first camera
        cam_name = camera_files[0].stem
        log.info(f"Using Waymo camera view: {cam_name}")

        video_path = input_folder / "videos" / f"{cam_name}.npy"
        poses_path = input_folder / "poses" / f"{cam_name}.npy"
        intrinsics_path = input_folder / "intrinsics" / f"{cam_name}.npy"
        masks_path = input_folder / "masks" / f"{cam_name}.npy"

        video_data = np.load(video_path)  # [T, H, W, 3] in BGR
        T, H_orig, W_orig, C = video_data.shape
        assert T == args.num_video_frames, f"Video has {T} frames but expected {args.num_video_frames}"

        poses_data = np.load(poses_path).astype(np.float32)  # [T, 4, 4]
        assert poses_data.shape == (T, 4, 4), f"Pose shape is {poses_data.shape}, expected (T, 4, 4)"

        K = np.load(intrinsics_path).astype(np.float32)  # [3,3]
        assert K.shape == (3, 3), f"Intrinsics shape is {K.shape}, expected (3, 3)"

        # Scale intrinsics for resize
        scale_h = args.height / H_orig
        scale_w = args.width / W_orig
        K_scaled = K.copy()
        K_scaled[0, 0] *= scale_w  # fx
        K_scaled[1, 1] *= scale_h  # fy
        K_scaled[0, 2] *= scale_w  # cx
        K_scaled[1, 2] *= scale_h  # cy

        # Allocate tensors
        image_bchw_float = torch.zeros(T, 3, args.height, args.width)
        depth_b1hw = torch.zeros(T, 1, args.height, args.width)
        mask_b1hw = torch.ones(T, 1, args.height, args.width)
        initial_w2c_b44 = torch.from_numpy(poses_data)
        intrinsics_b33 = torch.from_numpy(np.tile(K_scaled[None], (T, 1, 1))).float()

        # Load masks
        assert masks_path.exists(), f"Masks path {masks_path} does not exist"
        masks_data = np.load(masks_path)  # [T, H, W]
        assert masks_data.shape[0] == T, f"Mask frames {masks_data.shape[0]} != video frames {T}"
        masks_resized = np.zeros((T, args.height, args.width), dtype=np.float32)
        for t in range(T):
            if masks_data[t].shape != (args.height, args.width):
                masks_resized[t] = cv2.resize(masks_data[t], (args.width, args.height), interpolation=cv2.INTER_NEAREST)
            else:
                masks_resized[t] = masks_data[t]
        mask_b1hw = torch.from_numpy(masks_resized).unsqueeze(1)

        # If aligning with LiDAR, prefer explicit path; else try lidars/{cam}.npy
        lidar_frames = None
        if args.align_depth_with_lidar:
            if args.lidar_path is not None:
                lidar_frames = np.load(Path(args.lidar_path)).astype(np.float32)
            else:
                auto_lidar = input_folder / "lidars" / f"{cam_name}.npy"
                if auto_lidar.exists():
                    lidar_frames = np.load(auto_lidar).astype(np.float32)
            if lidar_frames is not None and lidar_frames.shape[0] != T:
                raise ValueError(f"LiDAR data has {lidar_frames.shape[0]} frames but video has {T}")

        log.info("Starting depth estimation and video processing (Waymo single view)...")
        with torch.no_grad():
            for t in range(T):
                frame_bgr = video_data[t]
                frame_resized = cv2.resize(frame_bgr, (args.width, args.height))
                depth = dav2_model.infer_image(frame_resized)

                if args.align_depth_with_lidar and lidar_frames is not None:
                    lidar_frame = lidar_frames[t]
                    if lidar_frame.shape != (args.height, args.width):
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

                frame_rgb = frame_resized[..., [2, 1, 0]].copy()
                frame_tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1)
                frame_tensor = frame_tensor / 127.5 - 1.0

                image_bchw_float[t] = frame_tensor
                depth_b1hw[t, 0] = torch.from_numpy(depth).float()

        # Cleanup model to save memory before moving on
        del dav2_model
        torch.cuda.empty_cache()
        log.info("Cleared depth anything v2 model and other variables from memory")

        # Move to device
        image_bchw_float = image_bchw_float.to(device)
        depth_b1hw = depth_b1hw.to(device)
        mask_b1hw = mask_b1hw.to(device)
        initial_w2c_b44 = initial_w2c_b44.to(device)
        intrinsics_b33 = intrinsics_b33.to(device)
            
    else:
        # Traditional input folder approach
        input_folder = Path(args.input_folder)

        if not input_folder.exists():
            raise FileNotFoundError(f"Input folder {input_folder} does not exist")

        # Get all image/metadata pairs
        pairs = get_image_and_metadata_pairs(input_folder)
        T = len(pairs)
        assert T == args.num_video_frames, f"Expected {args.num_video_frames} images, found {T} in {input_folder}"

        first_img = cv2.imread(str(pairs[0][0]))
        H, W = first_img.shape[:2]

        scale_h = args.height / H
        scale_w = args.width / W

        fx = args.default_fx * scale_w
        fy = args.default_fy * scale_h
        cx = args.default_cx * scale_w
        cy = args.default_cy * scale_h
        
        intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        image_bchw_float = torch.zeros(T, 3, args.height, args.width)  # RGB format
        depth_b1hw = torch.zeros(T, 1, args.height, args.width)
        mask_b1hw = torch.ones(T, 1, args.height, args.width)
        initial_w2c_b44 = torch.zeros(T, 4, 4)
        intrinsics_b33 = torch.from_numpy(np.tile(intrinsics[None], (T, 1, 1))).float()

        if args.gimbal_pitch is not None and args.gimbal_yaw is not None and args.gimbal_roll is not None:
            gimbal_angles = {
                'pitch': args.gimbal_pitch,
                'yaw': args.gimbal_yaw,
                'roll': args.gimbal_roll
            }
        else:
            gimbal_angles = None

        # Load LiDAR data once if alignment is enabled
        lidar_data = None
        if args.align_depth_with_lidar:
            if args.lidar_path is None:
                raise ValueError("--lidar_path is required when --align_depth_with_lidar is set")
            
            lidar_path = Path(args.lidar_path)
            if not lidar_path.exists():
                raise FileNotFoundError(f"LiDAR file not found: {lidar_path}")
            
            lidar_data = np.load(lidar_path).astype(np.float32)
            if len(lidar_data.shape) != 3:
                raise ValueError(f"LiDAR data should have shape (T, H, W), got {lidar_data.shape}")
            
            T_lidar, H_lidar, W_lidar = lidar_data.shape
            if T_lidar != T:
                raise ValueError(f"LiDAR data has {T_lidar} frames but input has {T} frames")
            
            log.info(f"Loaded LiDAR data with shape {lidar_data.shape}")

        log.info("Starting depth estimation and image processing...")
        with torch.no_grad():
            for i, (img_path, meta_path) in enumerate(tqdm(pairs, desc="Processing images")):
                # Load metadata and compute pose
                pose = load_map_to_camera_tf_matrix(meta_path, camera_angle=gimbal_angles)
                initial_w2c_b44[i] = torch.from_numpy(pose).float()

                # Load and resize image first to reduce memory usage
                img = cv2.imread(str(img_path))
                
                # Resize image first to target dimensions (much more memory efficient)
                img_resized = cv2.resize(img, (args.width, args.height))
                
                # Estimate depth on the smaller resized image (saves significant memory)
                depth = dav2_model.infer_image(img_resized)
                
                # Optional LiDAR alignment for each frame
                if args.align_depth_with_lidar and lidar_data is not None:
                    # Get LiDAR depth for this frame
                    lidar_depth = lidar_data[i]  # Shape: (H, W)
                    
                    if lidar_depth.shape != (args.height, args.width):
                        # Use sparse-aware resizing for LiDAR data
                        lidar_depth = _resize_sparse_depth(lidar_depth, (args.height, args.width))
                    
                    # Align predicted depth with LiDAR depth
                    pred_t = torch.from_numpy(depth).float()
                    lidar_t = torch.from_numpy(lidar_depth).float()
                    target_mask = lidar_t > 0
                    
                    depth_aligned = _align_inv_depth_to_depth(
                        1.0 / torch.clamp_min(pred_t, 1e-6),
                        lidar_t,
                        target_mask=target_mask
                    )
                    depth = depth_aligned.numpy()
                
                # Convert BGR to RGB and normalize
                frame_rgb = img_resized[..., [2, 1, 0]].copy()  # BGR to RGB
                frame_tensor = torch.from_numpy(frame_rgb).float()
                frame_tensor = frame_tensor.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
                frame_tensor = frame_tensor / 127.5 - 1.0  # Normalize to [-1, 1]
                
                # Store directly without additional resizing
                image_bchw_float[i] = frame_tensor
                depth_b1hw[i, 0] = torch.from_numpy(depth).float()
                
                # Clear intermediate variables to save memory
                del img_resized, frame_rgb, frame_tensor, depth, img, pose
                torch.cuda.empty_cache()
        
        # Delete the depth model after processing all images
        del dav2_model
        torch.cuda.empty_cache() 
        log.info("Cleared depth anything v2 model and other variables from memory")
        
        image_bchw_float = image_bchw_float.to(device)
        depth_b1hw = depth_b1hw.to(device)
        mask_b1hw = mask_b1hw.to(device)
        initial_w2c_b44 = initial_w2c_b44.to(device)
        intrinsics_b33 = intrinsics_b33.to(device)

    cache = Cache4D(
        input_image=image_bchw_float.clone(), # [T, C, H, W]
        input_depth=depth_b1hw,       # [T, 1, H, W]
        input_mask=mask_b1hw,         # [T, 1, H, W]
        input_w2c=initial_w2c_b44,  # [T, 4, 4]
        input_intrinsics=intrinsics_b33,# [T, 3, 3]
        filter_points_threshold=args.filter_points_threshold,
        input_format=["F", "C", "H", "W"],
        foreground_masking=args.foreground_masking,
    )

    initial_cam_w2c_for_traj = initial_w2c_b44
    initial_cam_intrinsics_for_traj = intrinsics_b33

    if args.trajectory_generation_method == "action_based_movement":
        assert args.trajectory in ["left", "right", "up", "down", "zoom_in", "zoom_out", "clockwise", "counterclockwise", "none"]
        assert args.camera_rotation in ["center_facing", "no_rotation", "trajectory_aligned"]
        assert args.movement_distance is not None
    
        try:
            generated_w2cs, generated_intrinsics = generate_camera_trajectory(
                trajectory_type=args.trajectory,
                initial_w2c=initial_cam_w2c_for_traj,
                initial_intrinsics=initial_cam_intrinsics_for_traj,
                num_frames=args.num_video_frames,
                movement_distance=args.movement_distance,
                camera_rotation=args.camera_rotation,
                center_depth=1.0,
                device=device.type,
            )
        except (ValueError, NotImplementedError) as e:
            log.critical(f"Failed to generate trajectory: {e}")
            return
        
    elif args.trajectory_generation_method == "pixel_focusing":
        assert args.target_pixel_x is not None
        assert args.target_pixel_y is not None
        assert args.movement_ratio is not None
        assert args.start_transition_frames is not None
        assert args.end_transition_frames is not None

        try:
            generated_w2cs, generated_intrinsics = generate_pixel_focused_trajectory(
                initial_w2c=initial_cam_w2c_for_traj,
                initial_intrinsics=initial_cam_intrinsics_for_traj,
                target_pixel=(args.target_pixel_x, args.target_pixel_y),
                num_frames=args.num_video_frames,
                movement_ratio=args.movement_ratio,
                start_transition_frames=args.start_transition_frames,
                end_transition_frames=args.end_transition_frames,
                depth_map=depth_b1hw, # [T, 1, H, W]
                device=device.type,
            )
        except (ValueError, NotImplementedError) as e:
            log.critical(f"Failed to generate trajectory: {e}")
            return

    elif args.trajectory_generation_method == "source_to_target_linear_interpolation":
        assert args.start_transition_frames is not None
        assert args.end_transition_frames is not None

        # Prefer direct NumPy poses if provided; otherwise fall back to metadata
        if args.source_pose_path is not None and args.target_pose_path is not None:
            source_pose = np.load(args.source_pose_path).astype(np.float32)
            target_pose = np.load(args.target_pose_path).astype(np.float32)
        else:
            assert args.source_meta_path is not None, "Provide --source_pose_path/--target_pose_path or --source_meta_path/--target_meta_path"
            assert args.target_meta_path is not None, "Provide --source_pose_path/--target_pose_path or --source_meta_path/--target_meta_path"
            source_pose = load_map_to_camera_tf_matrix(args.source_meta_path, camera_angle=gimbal_angles)
            target_pose = load_map_to_camera_tf_matrix(args.target_meta_path, camera_angle=gimbal_angles)
        
        source_pose_tensor = torch.tensor(source_pose, device=device, dtype=torch.float32)
        target_pose_tensor = torch.tensor(target_pose, device=device, dtype=torch.float32)

        generated_w2cs = generate_source_to_target_trajectory(
            source_w2c=source_pose_tensor,
            target_w2c=target_pose_tensor, 
            num_frames=args.num_video_frames,
            start_transition_frames=args.start_transition_frames,
            end_transition_frames=args.end_transition_frames,
            device=device,
        )
        generated_intrinsics = initial_cam_intrinsics_for_traj.unsqueeze(0)
    
    elif args.trajectory_generation_method == "target_folder_trajectory":
        assert args.target_meta_folder is not None

        target_meta_files = sorted(Path(args.target_meta_folder).glob("*.json"))
        if not target_meta_files:
            raise FileNotFoundError(f"No metadata files found in {args.target_meta_folder}")
        
        generated_w2cs = []
        generated_intrinsics = []

        for meta_path in tqdm(target_meta_files, desc="Processing target metadata"):
            pose = load_map_to_camera_tf_matrix(meta_path, gimbal_angles=gimbal_angles)
            generated_w2cs.append(torch.from_numpy(pose).float())
        
        generated_w2cs = torch.stack(generated_w2cs, dim=0).unsqueeze(0).to(device)
        generated_intrinsics = initial_cam_intrinsics_for_traj.unsqueeze(0).to(device)

    elif args.trajectory_generation_method == "pose_sequence":
        assert args.pose_sequence_path is not None, "Provide --pose_sequence_path for pose_sequence method"

        seq_np = np.load(args.pose_sequence_path).astype(np.float32)
        if seq_np.ndim != 3 or seq_np.shape[1:] != (4, 4):
            raise ValueError(f"--pose_sequence_path must be (T,4,4). Got {seq_np.shape}")

        seq_t = torch.from_numpy(seq_np).to(device=device, dtype=torch.float32)  # (Tin,4,4)
        seq_resampled = resample_w2c_sequence(seq_t, num_frames=args.num_video_frames, device=device)  # (T,4,4)

        # Make relative to first frame, then apply on top of the input's first pose
        seq0_inv = torch.inverse(seq_resampled[0])
        rel_seq = torch.matmul(seq_resampled, seq0_inv)
        base_w2c0 = initial_cam_w2c_for_traj[0] if initial_cam_w2c_for_traj.dim() == 3 else initial_cam_w2c_for_traj
        generated_w2cs = torch.matmul(rel_seq, base_w2c0).unsqueeze(0)  # (1,T,4,4)
        generated_intrinsics = initial_cam_intrinsics_for_traj[0].unsqueeze(0) if initial_cam_intrinsics_for_traj.dim() == 3 else initial_cam_intrinsics_for_traj.unsqueeze(0)

    else:
        raise ValueError(f"Invalid trajectory generation method: {args.trajectory_generation_method}")
    
    # Generate depth statistics visualization
    visualization_folder = "../visualization"
    plot_depth_statistics(
        depth_b1hw=depth_b1hw,
        output_path=visualization_folder,
        rendered_images_path=args.rendered_images_path
    )
    
    # Generate camera trajectory comparison visualization
    plot_camera_trajectory_comparison(
        initial_w2c=initial_cam_w2c_for_traj,
        generated_w2cs=generated_w2cs,
        output_path=visualization_folder,
        rendered_images_path=args.rendered_images_path
    )
    
    rendered_warp_images, rendered_warp_masks = cache.render_cache(
        generated_w2cs,
        generated_intrinsics,
    )
    
    log.info(f"Rendered warp images shape: {rendered_warp_images.shape}")
    log.info(f"Rendered warp masks shape: {rendered_warp_masks.shape}")
    
    # Save rendered tensors
    log.info("Saving rendered tensors...")
    
    # Save warp images
    warp_images_path = os.path.join(args.rendered_tensor_dir, args.rendered_images_path)
    torch.save(rendered_warp_images, warp_images_path)
    log.info(f"Saved warp images to: {warp_images_path}")
    
    # Save warp masks
    warp_masks_path = os.path.join(args.rendered_tensor_dir, args.rendered_masks_path)
    torch.save(rendered_warp_masks, warp_masks_path)
    log.info(f"Saved warp masks to: {warp_masks_path}")
    
    return

def main():
    """Main function to run the rendering creation."""
    parser = create_parser()
    args = parser.parse_args()
    if args.prompt is None:
        args.prompt = ""
    args.disable_guardrail = True
    args.disable_prompt_upsampler = True

    if args.input_folder is not None and not os.path.isabs(args.input_folder):
        args.input_folder = os.path.join("..", args.input_folder)

    if args.vipe_path is not None and not os.path.isabs(args.vipe_path):
        args.vipe_path = os.path.join("..", args.vipe_path)
    
    if args.waypo_path is not None and not os.path.isabs(args.waypo_path):
        args.waypo_path = os.path.join("..", args.waypo_path)

    if args.pose_sequence_path is not None and not os.path.isabs(args.pose_sequence_path):
        args.pose_sequence_path = os.path.join("..", args.pose_sequence_path)

    if args.source_meta_path is not None and not os.path.isabs(args.source_meta_path):
        args.source_meta_path = os.path.join("..", args.source_meta_path)
    if args.target_meta_path is not None and not os.path.isabs(args.target_meta_path):
        args.target_meta_path = os.path.join("..", args.target_meta_path)
    if args.target_meta_folder is not None and not os.path.isabs(args.target_meta_folder):
        args.target_meta_folder = os.path.join("..", args.target_meta_folder)
    if args.source_pose_path is not None and not os.path.isabs(args.source_pose_path):
        args.source_pose_path = os.path.join("..", args.source_pose_path)
    if args.target_pose_path is not None and not os.path.isabs(args.target_pose_path):
        args.target_pose_path = os.path.join("..", args.target_pose_path)
    if args.lidar_path is not None and not os.path.isabs(args.lidar_path):
        args.lidar_path = os.path.join("..", args.lidar_path)

    os.makedirs(args.rendered_tensor_dir, exist_ok=True)
    
    create_rendering(args)

if __name__ == "__main__":
    main() 