#!/usr/bin/env python3
"""
Create Rendering Video from Tensor
This script loads a rendered tensor and creates a video from it.
"""

import argparse
import os
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
import imageio

# Import the save_video function from the existing utils
def save_video(video, fps, H, W, video_save_quality, video_save_path):
    """Save video frames to file.

    Args:
        grid (np.ndarray): Video frames array [T,H,W,C]
        fps (int): Frames per second
        H (int): Frame height
        W (int): Frame width
        video_save_quality (int): Video encoding quality (0-10)
        video_save_path (str): Output video file path
    """
    kwargs = {
        "fps": fps,
        "quality": video_save_quality,
        "macro_block_size": 1,
        "ffmpeg_params": ["-s", f"{W}x{H}"],
        "output_params": ["-f", "mp4"],
    }
    imageio.mimsave(video_save_path, video, "mp4", **kwargs)

def load_tensor_from_path(tensor_path: str, device: torch.device) -> torch.Tensor:
    """Load a tensor from a file path."""
    if not os.path.exists(tensor_path):
        raise FileNotFoundError(f"Tensor file not found: {tensor_path}")
    
    tensor = torch.load(tensor_path, map_location=device)
    print(f"Loaded tensor from {tensor_path} with shape: {tensor.shape}")
    return tensor

def create_rendering_video(
    args: argparse.Namespace,
) -> None:
    """
    Create a video from a rendered tensor.
    
    Args:
        args: argparse.Namespace containing all arguments
    """
    # Set device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu") 
        print("CUDA not available, using CPU")
    
    # Load tensor
    print(f"Loading tensor from: {args.rendered_images_path}")
    warp_images_path = os.path.join(args.rendered_tensor_dir, args.rendered_images_path)
    rendered_warp_images = load_tensor_from_path(warp_images_path, device)
    print("rendered_warp_images loaded")
    # assert rendered_warp_images.shape == (1, 121, 1, 3, 704, 1280), f"rendered_warp_images must have shape [1, 121, 1, 3, 704, 1280], got {rendered_warp_images.shape}"
    _, T, n_i, C, H, W = rendered_warp_images.shape
    rendered_warp_images = rendered_warp_images.to(device)

    all_rendered_warps = []
    all_rendered_warps.append(rendered_warp_images.clone().cpu())

    squeezed_warps = [t.squeeze(0) for t in all_rendered_warps] 
    
    # Initialize variables
    final_video_to_save = None
    final_width = W
    final_height = H
    
    if squeezed_warps:
        n_max = max(t.shape[1] for t in squeezed_warps)
        padded_t_list = []
        for sq_t in squeezed_warps:
            current_n_i = sq_t.shape[1]
            padding_needed_dim1 = n_max - current_n_i

            pad_spec = (0,0, # W
                        0,0, # H
                        0,0, # C
                        0,padding_needed_dim1, # n_i
                        0,0) # T_chunk
            padded_t = F.pad(sq_t, pad_spec, mode='constant', value=-1.0)
            padded_t_list.append(padded_t)

        full_rendered_warp_tensor = torch.cat(padded_t_list, dim=0)

        T_total, _, C_dim, H_dim, W_dim = full_rendered_warp_tensor.shape
        buffer_video_TCHnW = full_rendered_warp_tensor.permute(0, 2, 3, 1, 4)
        buffer_video_TCHWstacked = buffer_video_TCHnW.contiguous().view(T_total, C_dim, H_dim, n_max * W_dim)
        buffer_video_TCHWstacked = (buffer_video_TCHWstacked * 0.5 + 0.5) * 255.0
        buffer_numpy_TCHWstacked = buffer_video_TCHWstacked.cpu().numpy().astype(np.uint8)
        buffer_numpy_THWC = np.transpose(buffer_numpy_TCHWstacked, (0, 2, 3, 1))

        final_video_to_save = buffer_numpy_THWC
        final_width = n_max * W
        final_height = H
    
    # Ensure we have a video to save
    if final_video_to_save is None:
        raise ValueError("No video data was processed successfully")

    video_save_path = os.path.join(
        args.video_save_folder,
        f"{args.video_save_name}.mp4"
    )
    
    os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
    
    # Save video
    save_video(
        video=final_video_to_save,
        fps=args.fps,
        H=final_height,
        W=final_width,
        video_save_quality=args.quality,
        video_save_path=video_save_path, 
    )
    
    print(f"Video saved successfully to: {video_save_path}")

def main():
    """Main function to run the rendering video creation."""
    parser = argparse.ArgumentParser(description="Create rendering video from tensor")
    
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
        help="Path to the rendered tensor file (.pt)"
    )
    
    parser.add_argument(
        "--video_save_folder",
        type=str,
        default="../rendered_videos",
        help="Output folder for the video file"
    )
    
    parser.add_argument(
        "--video_save_name",
        type=str,
        required=True,
        help="Name for the output video file (without .mp4 extension)"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second for the output video (default: 24)"
    )
    
    parser.add_argument(
        "--quality",
        type=int,
        default=5,
        help="Video quality from 0-10 (default: 5)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device to use for tensor loading (default: cpu)"
    )
    
    args = parser.parse_args()
    
    # Ensure output path has .mp4 extension
    if not args.video_save_name.endswith('.mp4'):
        args.video_save_name += '.mp4'
    
    create_rendering_video(
        args=args
    )

if __name__ == "__main__":
    main() 