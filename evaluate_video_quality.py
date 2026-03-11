#!/usr/bin/env python3
"""
Video Quality Evaluation Script

This script:
1. Generates video frames from an input image and pose sequence using GEN3C
2. Evaluates generated frames against reference images
3. Resamples generated frames to match reference images (using same logic as pose resampling)
4. Resizes if needed
5. Computes quality metrics (PSNR, SSIM, LPIPS, tLPIPS, FID, FVD, KVD, IS)
6. Saves results as JSON and output video
"""

import argparse
import os
import torch
import numpy as np
from pathlib import Path
import imageio
from typing import List, Tuple, Optional
import cv2
import sys
import torch.nn.functional as F

# Import necessary modules from GEN3C
from moge.model.v1 import MoGeModel
from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_Buffer
from cosmos_predict1.diffusion.inference.gen3c_pipeline import Gen3cPipeline
from cosmos_predict1.utils import log, misc 
from cosmos_predict1.diffusion.inference.gen3c_single_image import _predict_moge_depth 
from cosmos_predict1.diffusion.inference.inference_utils import add_common_arguments
from cosmos_predict1.diffusion.inference.camera_utils import _align_inv_depth_to_depth
from cosmos_predict1.diffusion.inference.create_rendering_multiview_waymo_image_input import _resize_sparse_depth

from camera_sequence_generation import resample_w2c_sequence
from video_metrics import compute_all_video_metrics
import json

torch.enable_grad(False)


def load_multiline_json(path: Path) -> dict:
    """Loads a file containing one JSON object per line and merges them into one dict."""
    merged = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            merged.update(json.loads(line))
    return merged


def sorted_pose_files(poses_dir: Path) -> List[Path]:
    """Sort pose JSON files numerically by filename."""
    files = list(poses_dir.glob("*.json"))
    
    def key(p: Path):
        try:
            return (0, int(p.stem))
        except ValueError:
            return (1, p.stem)
    
    return sorted(files, key=key)


def invert_se3(T: np.ndarray) -> np.ndarray:
    """Invert a rigid 4x4 transform (assumes last row is [0,0,0,1])."""
    R = T[:3, :3]
    t = T[:3, 3:4]
    Rt = R.T
    tinv = -Rt @ t
    Tinv = np.eye(4, dtype=T.dtype)
    Tinv[:3, :3] = Rt
    Tinv[:3, 3] = tinv[:, 0]
    return Tinv


def _to_gen3c_world_axes_4x4() -> np.ndarray:
    """Change-of-basis matrix to convert world axes to GEN3C Y-up convention."""
    S = np.eye(4, dtype=np.float64)
    S[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    return S


def _camera_transform_matrix(name: str) -> np.ndarray:
    """Return a 4x4 matrix C that changes CAMERA coordinates for a w2c matrix."""
    C = np.eye(4, dtype=np.float64)
    
    if name == "identity" or name == "opencv":
        return C
    
    if name == "swap_xz_negx":
        C[:3, :3] = np.array(
            [
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        return C
    
    if name == "swap_xz_negz":
        C[:3, :3] = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        return C
    
    if name == "swap_xz_negy":
        C[:3, :3] = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        return C
    
    raise ValueError(f"Unknown camera transform: {name}")


def export_poses_from_extrinsics(
    poses_dir: str,
    world_key: str = "mapToCamera",
    camera_transform: str = "opencv",
    relative_to_first: bool = False,
    to_gen3c_world: bool = True,
) -> np.ndarray:
    """
    Export poses from View-of-Delft Poses_Extrinsics directory.
    
    Args:
        poses_dir: Directory containing per-frame pose JSON files
        world_key: Which pose stream to use (mapToCamera, odomToCamera, UTMToCamera)
        camera_transform: Camera axis transform (opencv, swap_xz_negx, swap_xz_negz, swap_xz_negy)
        relative_to_first: If True, make poses relative to first frame
        to_gen3c_world: If True, convert world axes to GEN3C Y-up convention
    
    Returns:
        NumPy array of shape (T, 4, 4) containing w2c poses
    """
    poses_dir_path = Path(poses_dir)
    pose_files = sorted_pose_files(poses_dir_path)
    
    if not pose_files:
        raise FileNotFoundError(f"No pose JSON files found in: {poses_dir_path}")
    
    w2cs = []
    S_world = _to_gen3c_world_axes_4x4()
    C_cam = _camera_transform_matrix(camera_transform)
    
    for pf in pose_files:
        d = load_multiline_json(pf)
        if world_key not in d:
            raise KeyError(f"Missing '{world_key}' in {pf}")
        
        T = np.array(d[world_key], dtype=np.float64).reshape(4, 4)
        
        # Interpret input as c2w, convert to w2c for GEN3C
        T_w2c = invert_se3(T)
        # Camera-axis conversion: left-multiply (changes camera coordinates)
        T_w2c = C_cam @ T_w2c
        if to_gen3c_world:
            # Change world axes to GEN3C convention (Y-up) via right-multiplication
            T_w2c = T_w2c @ S_world
        w2cs.append(T_w2c.astype(np.float32))
    
    w2c_arr = np.stack(w2cs, axis=0)  # (T, 4, 4)
    
    if relative_to_first:
        w2c0_inv = np.linalg.inv(w2c_arr[0].astype(np.float64))
        w2c_arr = (w2c_arr.astype(np.float64) @ w2c0_inv).astype(np.float32)
    
    return w2c_arr


def load_images_from_folder(folder_path: str, sort: bool = True) -> List[np.ndarray]:
    """Load all images from a folder and return as list of HxWx3 uint8 RGB arrays."""
    folder = Path(folder_path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(folder.glob(f'*{ext}'))
        image_files.extend(folder.glob(f'*{ext.upper()}'))
    
    if sort:
        image_files = sorted(image_files)
    
    images = []
    for img_path in image_files:
        img = imageio.imread(str(img_path))
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        images.append(img)
    
    print(f"Loaded {len(images)} images from {folder_path}")
    return images


def subsample_frames(
    frames: List[np.ndarray],
    num_frames: int,
) -> List[np.ndarray]:
    """
    Uniformly subsample frames to a target length (no interpolation, no repetition).
    
    Args:
        frames: List of HxWx3 uint8 RGB frames
        num_frames: desired output length (must be <= len(frames))
    
    Returns:
        List of subsampled frames
    """
    T_in = len(frames)
    if T_in <= 0:
        raise ValueError("frames must have at least 1 frame")
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    if num_frames > T_in:
        raise ValueError(f"Cannot subsample {T_in} frames to {num_frames} frames (need more frames, not fewer)")
    
    if T_in == num_frames:
        return frames.copy()
    
    # Uniform subsampling (keep endpoints)
    idx = np.linspace(0, T_in - 1, num=num_frames)
    idx = np.round(idx).astype(np.int64)
    # Enforce monotonic non-decreasing and valid range
    idx[0] = 0
    idx[-1] = T_in - 1
    for i in range(1, num_frames):
        if idx[i] < idx[i - 1]:
            idx[i] = idx[i - 1]
        if idx[i] >= T_in:
            idx[i] = T_in - 1
    
    return [frames[int(i)].copy() for i in idx]


def resize_frames_to_match(
    frames: List[np.ndarray],
    target_h: int,
    target_w: int
) -> List[np.ndarray]:
    """Resize all frames to target resolution."""
    return [cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_LINEAR) for f in frames]


def tensor_to_frames(rendered_tensor: torch.Tensor) -> List[np.ndarray]:
    """
    Convert rendered tensor to list of frames.
    Expected shape: (1, T, 1, 3, H, W) or similar.
    Returns list of HxWx3 uint8 RGB frames.
    """
    # Handle different tensor shapes
    if rendered_tensor.dim() == 6:  # (1, T, 1, 3, H, W)
        tensor = rendered_tensor.squeeze(0).squeeze(1)  # (T, 3, H, W)
    elif rendered_tensor.dim() == 5:  # (1, T, 3, H, W)
        tensor = rendered_tensor.squeeze(0)  # (T, 3, H, W)
    elif rendered_tensor.dim() == 4:  # (T, 3, H, W)
        tensor = rendered_tensor
    else:
        raise ValueError(f"Unexpected tensor shape: {rendered_tensor.shape}")
    
    # Convert to numpy and permute to (T, H, W, 3)
    frames_np = tensor.permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, 3)
    
    # Convert from [-1, 1] or [0, 1] to [0, 255] uint8
    if frames_np.min() < 0:
        # Assume [-1, 1] range
        frames_np = (frames_np * 0.5 + 0.5) * 255.0
    elif frames_np.max() <= 1.0:
        # Assume [0, 1] range
        frames_np = frames_np * 255.0
    # Otherwise assume already in [0, 255] range
    
    frames_np = np.clip(frames_np, 0, 255).astype(np.uint8)
    
    # Convert to list
    frames = [frames_np[i] for i in range(frames_np.shape[0])]
    
    return frames


def generate_frames(
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rendered warp images and masks using GEN3C's rendering pipeline.
    This integrates the full rendering logic from create_rendering_image_input.py
    
    Returns:
        Tuple of (rendered_warp_images, rendered_warp_masks) tensors
    """
    misc.set_random_seed(args.seed)
    
    # Handle depth estimation and input preparation
    if args.lidar_path is not None and not args.align_depth_with_lidar:
        # Use LiDAR depth directly
        depth_path = Path(args.lidar_path)
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth file not found: {depth_path}")

        depth_np = np.load(depth_path).astype(np.float32)
        if depth_np.ndim == 3 and depth_np.shape[0] == 1:
            depth_np = depth_np[0]

        if depth_np.shape != (args.height, args.width):
            depth_np = _resize_sparse_depth(depth_np, (args.height, args.width))

        mask_np = (depth_np > 0).astype(np.float32)

        input_image_np = cv2.imread(args.input_image_path)
        input_image_rgb = cv2.cvtColor(input_image_np, cv2.COLOR_BGR2RGB)
        input_image_resized = cv2.resize(input_image_rgb, (args.width, args.height))
        input_image = (
            torch.from_numpy(input_image_resized)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 127.5
            - 1.0
        ).to(device)

        input_depth = torch.from_numpy(depth_np).float().unsqueeze(0).unsqueeze(0).to(device)
        input_mask = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(device)

        initial_w2c = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)

        actual_height, actual_width = input_image_np.shape[:2]
        fx = args.default_fx * (args.width / actual_width)
        fy = args.default_fy * (args.height / actual_height)
        cx = args.default_cx * (args.width / actual_width)
        cy = args.default_cy * (args.height / actual_height)

        intrinsics_matrix = torch.tensor(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            device=device,
            dtype=torch.float32,
        )
        initial_intrinsics = intrinsics_matrix.unsqueeze(0)

    elif args.depth_estimator == "moge":
        moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
        
        (
            moge_image_b1chw_float,
            moge_depth_b11hw,
            moge_mask_b11hw,
            moge_initial_w2c_b144,
            moge_intrinsics_b133,
        ) = _predict_moge_depth(args.input_image_path, args.height, args.width, device, moge_model)
        
        input_image = moge_image_b1chw_float[:, 0].clone()
        input_depth = moge_depth_b11hw[:, 0]
        input_mask = moge_mask_b11hw[:, 0]
        initial_w2c = moge_initial_w2c_b144[:, 0]
        initial_intrinsics = moge_intrinsics_b133[:, 0]
        
        if args.align_depth_with_lidar:
            if args.lidar_path is None:
                raise ValueError("--lidar_path is required when --align_depth_with_lidar is set")
            
            lidar_path = Path(args.lidar_path)
            if not lidar_path.exists():
                raise FileNotFoundError(f"LiDAR file not found: {lidar_path}")
            
            lidar_depth = np.load(lidar_path).astype(np.float32)
            if lidar_depth.shape != (args.height, args.width):
                lidar_depth = _resize_sparse_depth(lidar_depth, (args.height, args.width))
            
            pred_depth = input_depth[0].cpu().numpy()
            lidar_t = torch.from_numpy(lidar_depth).float()
            pred_t = torch.from_numpy(pred_depth).float()
            target_mask = lidar_t > 0
            
            depth_aligned = _align_inv_depth_to_depth(
                1.0 / torch.clamp_min(pred_t, 1e-6),
                lidar_t,
                target_mask=target_mask
            )
            input_depth[0] = depth_aligned
        
    elif args.depth_estimator == "depthanythingv2":
        assert args.default_fx is not None
        assert args.default_fy is not None
        assert args.default_cx is not None
        assert args.default_cy is not None
        
        # Find repo root
        script_path = Path(__file__).resolve()
        repo_root = None
        for parent in [script_path.parent] + list(script_path.parents):
            if (parent / "Depth-Estimation" / "Depth-Anything-V2").exists():
                repo_root = parent
                break
        if repo_root is None:
            repo_root = script_path.parent.parent.parent.parent.parent
        sys.path.insert(0, str(repo_root / "Depth-Estimation" / "Depth-Anything-V2" / "metric_depth"))
        
        from depth_anything_v2.dpt import DepthAnythingV2
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        encoder = 'vitb'
        max_depth = 80
        DepthAnythingV2_checkpoint_path = repo_root / 'Depth-Estimation' / 'Depth-Anything-V2' / 'checkpoints' / f'depth_anything_v2_metric_hypersim_{encoder}.pth'
        
        dav2_model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
        dav2_model.load_state_dict(torch.load(DepthAnythingV2_checkpoint_path, map_location=device))
        dav2_model.to(device).eval()

        input_image_np = cv2.imread(args.input_image_path)
        actual_height, actual_width = input_image_np.shape[:2]
        
        input_image_rgb = cv2.cvtColor(input_image_np, cv2.COLOR_BGR2RGB)
        input_image_resized = cv2.resize(input_image_rgb, (args.width, args.height))
        
        input_image_tensor = torch.from_numpy(input_image_resized).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        input_image_tensor = input_image_tensor.to(device)

        with torch.no_grad():
            depth_map = dav2_model.infer_image(input_image_np)
            depth_map = cv2.resize(depth_map, (args.width, args.height))

        depth_tensor = torch.from_numpy(depth_map).float().unsqueeze(0).unsqueeze(0).to(device)
        
        if args.align_depth_with_lidar:
            if args.lidar_path is None:
                raise ValueError("--lidar_path is required when --align_depth_with_lidar is set")
            
            lidar_path = Path(args.lidar_path)
            if not lidar_path.exists():
                raise FileNotFoundError(f"LiDAR file not found: {lidar_path}")
            
            lidar_depth = np.load(lidar_path).astype(np.float32)
            if lidar_depth.shape != (args.height, args.width):
                lidar_depth = _resize_sparse_depth(lidar_depth, (args.height, args.width))
            
            pred_depth = depth_tensor[0, 0].cpu().numpy()
            lidar_t = torch.from_numpy(lidar_depth).float()
            pred_t = torch.from_numpy(pred_depth).float()
            target_mask = lidar_t > 0
            
            depth_aligned = _align_inv_depth_to_depth(
                1.0 / torch.clamp_min(pred_t, 1e-6),
                lidar_t,
                target_mask=target_mask
            )
            depth_tensor[0, 0] = depth_aligned
        
        input_image = input_image_tensor
        input_depth = depth_tensor
        input_mask = torch.ones_like(depth_tensor)

        initial_w2c = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
        
        fx = args.default_fx * (args.width / actual_width)
        fy = args.default_fy * (args.height / actual_height)
        cx = args.default_cx * (args.width / actual_width)
        cy = args.default_cy * (args.height / actual_height)
        
        intrinsics_matrix = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], device=device, dtype=torch.float32)
        
        initial_intrinsics = intrinsics_matrix.unsqueeze(0)

    else:
        raise ValueError(f"Unknown depth estimator: {args.depth_estimator}")
    
    # Create cache
    frame_buffer_max = 2
    generator = torch.Generator(device=device).manual_seed(args.seed)

    cache_input_mask = input_mask if (args.lidar_path is not None and not args.align_depth_with_lidar) or args.depth_estimator == "moge" else None
    
    cache = Cache3D_Buffer(
        frame_buffer_max=frame_buffer_max,
        generator=generator,
        noise_aug_strength=args.noise_aug_strength,
        input_image=input_image,
        input_depth=input_depth,
        input_mask=cache_input_mask,
        input_w2c=initial_w2c,
        input_intrinsics=initial_intrinsics,
        filter_points_threshold=args.filter_points_threshold,
        foreground_masking=args.foreground_masking,
    )
    
    initial_cam_w2c_for_traj = initial_w2c[0]
    initial_cam_intrinsics_for_traj = initial_intrinsics[0]
    
    # Generate trajectory from pose sequence
    assert args.pose_sequence_path is not None, "Provide --pose_sequence_path"
    
    seq_np = np.load(args.pose_sequence_path).astype(np.float32)
    if seq_np.ndim != 3 or seq_np.shape[1:] != (4, 4):
        raise ValueError(f"--pose_sequence_path must be (T,4,4). Got {seq_np.shape}")

    seq_t = torch.from_numpy(seq_np).to(device=device, dtype=torch.float32)
    seq_resampled = resample_w2c_sequence(seq_t, num_frames=args.num_video_frames, device=device)

    # Make the sequence relative to its first frame, then apply on top of initial pose
    seq0_inv = torch.inverse(seq_resampled[0])
    rel_seq = torch.matmul(seq_resampled, seq0_inv)
    generated_w2cs = torch.matmul(rel_seq, initial_cam_w2c_for_traj)
    generated_w2cs = generated_w2cs.unsqueeze(0)  # (1,T_out,4,4)
    generated_intrinsics = initial_cam_intrinsics_for_traj.unsqueeze(0).repeat(args.num_video_frames, 1, 1).unsqueeze(0)
    
    # Render frames
    print("Rendering frames...")
    rendered_warp_images, rendered_warp_masks = cache.render_cache(
        generated_w2cs,
        generated_intrinsics,
    )
    
    log.info(f"Rendered warp images shape: {rendered_warp_images.shape}")
    log.info(f"Rendered warp masks shape: {rendered_warp_masks.shape}")
    
    # Store rendered tensors for diffusion
    # These will be used by apply_diffusion_model
    return rendered_warp_images, rendered_warp_masks


def apply_diffusion_model(
    args: argparse.Namespace,
    rendered_warp_images: torch.Tensor,
    rendered_warp_masks: torch.Tensor,
    device: torch.device,
) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    """
    Apply diffusion model to rendered frames to generate final video.
    
    Args:
        args: Parsed arguments
        rendered_warp_images: Rendered warp images tensor (1, T, 1, 3, H, W)
        rendered_warp_masks: Rendered warp masks tensor (1, T, 1, 1, H, W)
        device: torch device
    
    Returns:
        Tuple of (final_frames, buffer_video):
        - final_frames: List of HxWx3 uint8 RGB frames from diffusion model (for evaluation)
        - buffer_video: Optional numpy array (T, H, W_total, C) with buffer concatenated if save_buffer=True
    """
    print("\nApplying diffusion model to rendered frames...")
    
    # Initialize buffer collection if save_buffer is enabled
    all_rendered_warps = []
    if args.save_buffer:
        all_rendered_warps.append(rendered_warp_images.clone().cpu())
    
    # Initialize the diffusion pipeline
    pipeline = Gen3cPipeline(
        inference_type="video2world",
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name="Gen3C-Cosmos-7B",
        prompt_upsampler_dir=args.prompt_upsampler_dir,
        enable_prompt_upsampler=not args.disable_prompt_upsampler,
        offload_network=args.offload_diffusion_transformer,
        offload_tokenizer=args.offload_tokenizer,
        offload_text_encoder_model=args.offload_text_encoder_model,
        offload_prompt_upsampler=args.offload_prompt_upsampler,
        offload_guardrail_models=args.offload_guardrail_models,
        disable_guardrail=args.disable_guardrail,
        disable_prompt_encoder=args.disable_prompt_encoder,
        guidance=args.guidance,
        num_steps=args.num_steps,
        height=args.height,
        width=args.width,
        fps=args.fps,
        num_video_frames=args.num_video_frames,
        seed=args.seed,
    )
    
    # Generate video using diffusion model
    generated_output = pipeline.generate(
        prompt=args.prompt,
        image_path=args.input_image_path,
        negative_prompt=args.negative_prompt,
        rendered_warp_images=rendered_warp_images,
        rendered_warp_masks=rendered_warp_masks,
    )
    
    if generated_output is None:
        raise RuntimeError("Video generation with diffusion model failed!")
    
    video, _ = generated_output
    log.info("Diffusion video generation completed!")
    log.info(f"Generated video shape: {video.shape}")
    
    # Convert video (T, H, W, C) numpy array to list of frames for evaluation
    # video is already in uint8 format [0, 255]
    final_frames = [video[i] for i in range(video.shape[0])]
    
    # Process buffer video if save_buffer is enabled
    buffer_video = None
    if args.save_buffer and all_rendered_warps:
        squeezed_warps = [t.squeeze(0) for t in all_rendered_warps]  # Each is (T_chunk, n_i, C, H, W)
        
        if squeezed_warps:
            n_max = max(t.shape[1] for t in squeezed_warps)
            
            padded_t_list = []
            for sq_t in squeezed_warps:
                # sq_t shape: (T_chunk, n_i, C, H, W)
                current_n_i = sq_t.shape[1]
                padding_needed_dim1 = n_max - current_n_i
                
                pad_spec = (0, 0,  # W
                           0, 0,  # H
                           0, 0,  # C
                           0, padding_needed_dim1,  # n_i
                           0, 0)  # T_chunk
                padded_t = F.pad(sq_t, pad_spec, mode='constant', value=-1.0)
                padded_t_list.append(padded_t)
            
            full_rendered_warp_tensor = torch.cat(padded_t_list, dim=0)
            
            T_total, _, C_dim, H_dim, W_dim = full_rendered_warp_tensor.shape
            buffer_video_TCHnW = full_rendered_warp_tensor.permute(0, 2, 3, 1, 4)
            buffer_video_TCHWstacked = buffer_video_TCHnW.contiguous().view(T_total, C_dim, H_dim, n_max * W_dim)
            buffer_video_TCHWstacked = (buffer_video_TCHWstacked * 0.5 + 0.5) * 255.0
            buffer_numpy_TCHWstacked = buffer_video_TCHWstacked.cpu().numpy().astype(np.uint8)
            buffer_numpy_THWC = np.transpose(buffer_numpy_TCHWstacked, (0, 2, 3, 1))
            
            # Concatenate buffer with generated video horizontally
            buffer_video = np.concatenate([buffer_numpy_THWC, video], axis=2)
            log.info(f"Created buffer video with {n_max} warp buffers. Final width: {buffer_video.shape[2]}")
        else:
            log.info("No warp buffers to save.")
    
    print(f"Converted diffusion output to {len(final_frames)} frames")
    return final_frames, buffer_video


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(description="Evaluate video quality metrics with full rendering pipeline")
    
    # Add common arguments from GEN3C
    add_common_arguments(parser)
    
    # Input arguments
    parser.add_argument(
        "--input_image",
        type=str,
        required=True,
        help="Path to input image for generation"
    )
    parser.add_argument(
        "--pose_sequence",
        type=str,
        default=None,
        help="Path to pose sequence .npy file (T, 4, 4). If not provided, will export from --poses_extrinsics_dir"
    )
    parser.add_argument(
        "--poses_extrinsics_dir",
        type=str,
        default=None,
        help="Directory containing View-of-Delft pose JSON files (Poses_Extrinsics). "
             "If provided, poses will be exported automatically."
    )
    parser.add_argument(
        "--world_key",
        type=str,
        default="mapToCamera",
        choices=["mapToCamera", "odomToCamera", "UTMToCamera"],
        help="Which pose stream to use when exporting from extrinsics directory"
    )
    parser.add_argument(
        "--camera_transform",
        type=str,
        default="opencv",
        choices=["opencv", "swap_xz_negx", "swap_xz_negz", "swap_xz_negy"],
        help="Camera axis transform when exporting poses"
    )
    parser.set_defaults(relative_to_first=True)
    parser.add_argument(
        "--no_to_gen3c_world",
        action="store_true",
        help="Disable world axis conversion to GEN3C Y-up convention"
    )
    parser.add_argument(
        "--reference_images_folder",
        type=str,
        required=True,
        help="Folder containing reference/ground truth images"
    )
    
    # Rendering arguments
    parser.add_argument(
        "--depth_estimator",
        type=str,
        choices=["moge", "depthanythingv2"],
        default="moge",
        help="Depth estimation model to use: moge (default) or depthanythingv2"
    )
    parser.add_argument(
        "--default_fx",
        type=float,
        default=739.75492315,
        help="Default focal length x for DepthAnythingV2"
    )
    parser.add_argument(
        "--default_fy",
        type=float,
        default=741.66148189,
        help="Default focal length y for DepthAnythingV2"
    )
    parser.add_argument(
        "--default_cx",
        type=float,
        default=605.94283506,
        help="Default principal point x for DepthAnythingV2"
    )
    parser.add_argument(
        "--default_cy",
        type=float,
        default=343.51934258,
        help="Default principal point y for DepthAnythingV2"
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
        help="Filter the points continuity of the warped images"
    )
    parser.add_argument(
        "--foreground_masking",
        action="store_true",
        help="Use foreground masking for the warped images"
    )
    parser.add_argument(
        "--align_depth_with_lidar",
        action="store_true",
        help="Align predicted depth to LiDAR depth"
    )
    parser.add_argument(
        "--lidar_path",
        type=str,
        default=None,
        help="Path to LiDAR depth map (.npy)"
    )
    parser.add_argument(
        "--pose_sequence_path",
        type=str,
        default=None,
        help="Path to pose sequence .npy file (alias for --pose_sequence)"
    )
    parser.add_argument(
        "--prompt_upsampler_dir",
        type=str,
        default="Pixtral-12B",
        help="Prompt upsampler weights directory relative to checkpoint_dir"
    )
    parser.add_argument(
        "--save_buffer",
        action="store_true",
        help="Whether to save the warped images buffer alongside the output video"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for computation (default: cuda)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for metrics JSON and video"
    )
    parser.add_argument(
        "--output_video_name",
        type=str,
        default="generated_video.mp4",
        help="Name for output video file"
    )
    parser.add_argument(
        "--name_suffix",
        type=str,
        default=None,
        help="Optional suffix for output files"
    )
    
    
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # Set defaults for GEN3C
    if args.prompt is None:
        args.prompt = ""
    args.disable_guardrail = True
    args.disable_prompt_upsampler = True
    
    args.relative_to_first = True
    
    # Handle path resolution (similar to create_rendering_image_input.py)
    if not os.path.isabs(args.input_image):
        args.input_image = os.path.join("..", args.input_image)
    args.input_image_path = args.input_image  # For compatibility with rendering code
    
    # Resolve output directory path (relative paths need to go up from GEN3C)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join("..", args.output_dir)
    
    # Create output directory early (needed for temp pose file)
    print(f"Creating output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory created (or already exists): {os.path.abspath(args.output_dir)}")
    
    # Handle pose sequence: either load from file or export from extrinsics directory
    if args.pose_sequence_path is None:
        args.pose_sequence_path = args.pose_sequence
    
    # If pose sequence not provided but extrinsics directory is, export poses
    if args.pose_sequence_path is None and args.poses_extrinsics_dir is not None:
        print(f"Exporting poses from {args.poses_extrinsics_dir}...")
        if not os.path.isabs(args.poses_extrinsics_dir):
            args.poses_extrinsics_dir = os.path.join("..", args.poses_extrinsics_dir)
        
        # Export poses
        w2c_poses = export_poses_from_extrinsics(
            poses_dir=args.poses_extrinsics_dir,
            world_key=args.world_key,
            camera_transform=args.camera_transform,
            relative_to_first=args.relative_to_first,
            to_gen3c_world=not args.no_to_gen3c_world,
        )
        
        # Save to temporary file
        temp_pose_file = os.path.join(args.output_dir, "temp_exported_poses.npy")
        print(f"Saving temp pose file to: {os.path.abspath(temp_pose_file)}")
        np.save(temp_pose_file, w2c_poses)
        args.pose_sequence_path = temp_pose_file
        print(f"Exported {w2c_poses.shape[0]} poses to {temp_pose_file}")
    elif args.pose_sequence_path is not None:
        if not os.path.isabs(args.pose_sequence_path):
            args.pose_sequence_path = os.path.join("..", args.pose_sequence_path)
    
    if args.lidar_path is not None and not os.path.isabs(args.lidar_path):
        args.lidar_path = os.path.join("..", args.lidar_path)
    
    # Resolve reference images folder path (relative paths need to go up from GEN3C)
    if not os.path.isabs(args.reference_images_folder):
        args.reference_images_folder = os.path.join("..", args.reference_images_folder)
    
    # Set device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")
    
    # Output directory already created earlier (for temp pose file)
    
    # Load reference images
    print("Loading reference images...")
    reference_frames = load_images_from_folder(args.reference_images_folder)
    num_reference = len(reference_frames)
    print(f"Loaded {num_reference} reference frames")
    
    if num_reference == 0:
        raise ValueError(f"No images found in {args.reference_images_folder}")
    
    # Get reference resolution
    ref_h, ref_w = reference_frames[0].shape[:2]
    print(f"Reference image resolution: {ref_h}x{ref_w}")
    
    # Generate rendered warp images and masks using GEN3C
    print("\nRendering frames using GEN3C...")
    rendered_warp_images, rendered_warp_masks = generate_frames(args, device)
    print(f"Rendered warp images shape: {rendered_warp_images.shape}")
    print(f"Rendered warp masks shape: {rendered_warp_masks.shape}")
    
    # Apply diffusion model to generate final video frames
    generated_frames, buffer_video = apply_diffusion_model(
        args,
        rendered_warp_images,
        rendered_warp_masks,
        device
    )
    num_generated = len(generated_frames)
    print(f"Generated {num_generated} frames from diffusion model with type {type(generated_frames[0])} and min value {generated_frames[0].min()} and max value {generated_frames[0].max()}")
    
    # Get generated resolution
    gen_h, gen_w = generated_frames[0].shape[:2]
    print(f"Generated image resolution: {gen_h}x{gen_w}")
    
    # Subsample the longer sequence to match the shorter one (before resizing)
    if num_generated > num_reference:
        # Generated has more frames: subsample generated to match reference
        print(f"Subsampling generated frames from {num_generated} to {num_reference} frames...")
        generated_frames = subsample_frames(generated_frames, num_reference)
        final_len = num_reference
    elif num_generated < num_reference:
        # Reference has more frames: subsample reference to match generated
        print(f"Subsampling reference frames from {num_reference} to {num_generated} frames...")
        reference_frames = subsample_frames(reference_frames, num_generated)
        final_len = num_generated
    else:
        # Same length
        final_len = num_generated
    
    # Resize reference frames to match generated resolution (compute metrics at generated size)
    if gen_h != ref_h or gen_w != ref_w:
        print(f"Resizing reference frames from {ref_h}x{ref_w} to {gen_h}x{gen_w} to match generated size")
        reference_frames = resize_frames_to_match(reference_frames, gen_h, gen_w)
    else:
        print(f"Reference and generated frames already have the same resolution: {gen_h}x{gen_w}")
    
    final_generated = generated_frames
    final_reference = reference_frames
    
    # Verify both sequences have the same resolution
    final_gen_h, final_gen_w = final_generated[0].shape[:2]
    final_ref_h, final_ref_w = final_reference[0].shape[:2]
    assert final_gen_h == final_ref_h and final_gen_w == final_ref_w, \
        f"Resolution mismatch: generated={final_gen_h}x{final_gen_w}, reference={final_ref_h}x{final_ref_w}"
    
    print(f"Final sequence length: {final_len} frames (generated: {len(final_generated)}, reference: {len(final_reference)})")
    print(f"Metrics will be computed at generated resolution: {final_gen_h}x{final_gen_w}")
    print(f"Both sequences confirmed at same resolution: {final_gen_h}x{final_gen_w}")
    
    # Compute metrics at generated size
    print("\nComputing video quality metrics...")
    compute_all_video_metrics(
        original_frames=final_reference,
        generated_frames=final_generated,
        device=device,
        output_dir=args.output_dir,
        name_suffix=args.name_suffix
    )
    
    # Save output video using GEN3C's save_video function
    from cosmos_predict1.utils.io import save_video as save_video_gen3c
    
    output_video_path = os.path.join(args.output_dir, args.output_video_name)
    print(f"\nSaving output video to {output_video_path}...")
    
    # Convert frames list to numpy array (T, H, W, C)
    video_array = np.stack(final_generated, axis=0)
    
    # Calculate FPS for 5-second video
    target_duration_seconds = 5.0
    video_fps = final_len / target_duration_seconds
    print(f"Setting video FPS to {video_fps:.2f} for {target_duration_seconds}s duration ({final_len} frames)")
    
    save_video_gen3c(
        video=video_array,
        fps=video_fps,
        H=final_generated[0].shape[0],
        W=final_generated[0].shape[1],
        video_save_quality=5,
        video_save_path=output_video_path,
    )
    
    # Create side-by-side comparison video (generated | original)
    comparison_video_path = os.path.join(args.output_dir, args.output_video_name.replace(".mp4", "_comparison.mp4"))
    print(f"\nCreating side-by-side comparison video: {comparison_video_path}...")
    
    # Convert reference frames to numpy array
    reference_array = np.stack(final_reference, axis=0)
    
    # Concatenate horizontally: [generated | original]
    comparison_array = np.concatenate([video_array, reference_array], axis=2)  # (T, H, W_gen+W_ref, C)
    
    save_video_gen3c(
        video=comparison_array,
        fps=video_fps,
        H=comparison_array.shape[1],
        W=comparison_array.shape[2],
        video_save_quality=5,
        video_save_path=comparison_video_path,
    )
    print(f"Comparison video saved to: {comparison_video_path}")
    
    # Save buffer video if available (with rendered warps concatenated)
    if buffer_video is not None:
        buffer_video_path = os.path.join(args.output_dir, args.output_video_name.replace(".mp4", "_with_buffer.mp4"))
        print(f"\nSaving buffer video (with rendered warps) to {buffer_video_path}...")
        
        # Adjust buffer video FPS for 5-second duration
        buffer_fps = buffer_video.shape[0] / target_duration_seconds
        print(f"Setting buffer video FPS to {buffer_fps:.2f} for {target_duration_seconds}s duration ({buffer_video.shape[0]} frames)")
        
        save_video_gen3c(
            video=buffer_video,
            fps=buffer_fps,
            H=buffer_video.shape[1],
            W=buffer_video.shape[2],
            video_save_quality=5,
            video_save_path=buffer_video_path,
        )
        print(f"Buffer video saved to: {buffer_video_path}")
    
    print("\nEvaluation complete!")
    print(f"Metrics saved to: {os.path.join(args.output_dir, 'metrics.json')}")
    print(f"Output video saved to: {output_video_path}")
    print(f"Comparison video saved to: {comparison_video_path}")


if __name__ == "__main__":
    main()
