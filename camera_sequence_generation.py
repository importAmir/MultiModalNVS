import torch
import numpy as np
from scipy.spatial.transform import Rotation as Rotation
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from pathlib import Path

from typing import Optional, Tuple 
import numpy as np
import json


def load_drone_to_map_frame_tf_matrix(meta_data_file_path):
    """
    Loads the transformation from the drone frame to the map frame from a JSON file.

    Args:
        json_file (str): Path to the JSON file containing the transformation.

    Returns:
        np.ndarray: 4x4 transformation matrix from the drone frame to the map frame.
    """
    
    with open(meta_data_file_path, 'r') as file:
        meta_data = json.load(file)

    # Extract translation and rotation
    if "Frame Transformations" in meta_data:
        if "drone_1/base_link" in meta_data["Frame Transformations"]['map']:
            translation = meta_data["Frame Transformations"]["map"]["drone_1/base_link"]["translation"]
            rotation = meta_data["Frame Transformations"]["map"]["drone_1/base_link"]["rotation"]
        elif "drone_0/base_link" in meta_data["Frame Transformations"]['map']:
            translation = meta_data["Frame Transformations"]["map"]["drone_0/base_link"]["translation"]
            rotation = meta_data["Frame Transformations"]["map"]["drone_0/base_link"]["rotation"]
    elif "local_pose" in meta_data:
        translation = meta_data["local_pose"]["position"]
        rotation = meta_data["local_pose"]["orientation"]

    # Create translation vector
    translation_vector = np.array([translation["x"], translation["y"], translation["z"]])

    # Create rotation matrix from quaternion
    quaternion = [rotation["x"], rotation["y"], rotation["z"], rotation["w"]]
    rotation_matrix = Rotation.from_quat(quaternion).as_matrix()

    # print(f"rotation matrix: {rotation_matrix}")
    # print(f"translation vector: {translation_vector}")
    # Construct 4x4 homogeneous transformation matrix
    tf_matrix = np.eye(4)
    tf_matrix[:3, :3] = rotation_matrix
    tf_matrix[:3, 3] = translation_vector

    return tf_matrix

def load_map_to_drone_frame_tf_matrix(meta_data_file_path):
    """
    Loads the transformation from the map frame to the drone frame by inverting the transformation read on JSON file.

    Args:
        meta_data_file_path (str): Path to the JSON file containing the transformation.

    Returns:
        np.ndarray: 4x4 transformation matrix from the map frame to the drone frame.
    """
    with open(meta_data_file_path, 'r') as file:
        meta_data = json.load(file)

    # Extract translation and rotation
    if "Frame Transformations" in meta_data:        
        if "drone_1/base_link" in meta_data["Frame Transformations"]['map']:
            translation = meta_data["Frame Transformations"]["map"]["drone_1/base_link"]["translation"]
            rotation = meta_data["Frame Transformations"]["map"]["drone_1/base_link"]["rotation"]
        elif "drone_0/base_link" in meta_data["Frame Transformations"]['map']:
            translation = meta_data["Frame Transformations"]["map"]["drone_0/base_link"]["translation"]
            rotation = meta_data["Frame Transformations"]["map"]["drone_0/base_link"]["rotation"]
    elif "local_pose" in meta_data:
        translation = meta_data["local_pose"]["position"]
        rotation = meta_data["local_pose"]["orientation"]

    
    # Create translation vector
    translation_vector = np.array([translation["x"], translation["y"], translation["z"]])

    # Create rotation matrix from quaternion
    quaternion = [rotation["x"], rotation["y"], rotation["z"], rotation["w"]]
    rotation_matrix = Rotation.from_quat(quaternion).as_matrix()

    # Invert the rotation matrix
    inverse_rotation_matrix = rotation_matrix.T

    # Invert the translation vector
    inverse_translation_vector = -np.dot(inverse_rotation_matrix, translation_vector)

    # Construct the inverse 4x4 homogeneous transformation matrix
    inverse_tf_matrix = np.eye(4)
    inverse_tf_matrix[:3, :3] = inverse_rotation_matrix
    inverse_tf_matrix[:3, 3] = inverse_translation_vector

    return inverse_tf_matrix

def load_camera_to_drone_frame_tf_matrix(meta_data_file_path, camera_angle=(0,0,0)):
    
    # X_drone = Z_camera, Y_drone = -X_camera, Z_drone = -Y_camera
    # Define the 3x3 Rotation Matrix (Camera Frame to Drone Frame)
    # transformation_matrix * [x_camera; y_camera; z_camera] = [x_drone; y_drone; z_drone]
    
    if meta_data_file_path:
        try:
            with open(meta_data_file_path, 'r') as file:
                meta_data = json.load(file)
                if "gimbal_angles" in meta_data:        
                    x_pitch = meta_data["gimbal_angles"]['pitch']
                    y_yaw = meta_data["gimbal_angles"]['yaw']
                    z_roll = meta_data["gimbal_angles"]['roll']
                    print("Loaded camera angles from metadata!")
                else:
                    print(f"Warning: there is no camera_angle info in metadata '{meta_data_file_path}': {e} - will use camera_angle if provided")
                    x_pitch, y_yaw, z_roll = camera_angle if camera_angle is not None else (0, 0, 0)        
        except Exception as e:
            print(f"Warning: Failed to open metadata file '{meta_data_file_path}': {e} - will use camera_angles if provided")
            x_pitch, y_yaw, z_roll = camera_angle if camera_angle is not None else (0, 0, 0)
    elif camera_angle is not None:
        x_pitch, y_yaw, z_roll = camera_angle
    else:
        x_pitch, y_yaw, z_roll = (0,0,0)
        
    print(f"Camera angles are: ({x_pitch},{y_yaw},{z_roll})")
    # Convert angles to radians
    pitch = np.radians(x_pitch)
    yaw = np.radians(y_yaw)
    roll = np.radians(z_roll)
    
    # Rotation matrix for roll (X-axis)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    
    # Rotation matrix for pitch (Y-axis)
    R_y = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    
    # Rotation matrix for yaw (Z-axis)
    R_z = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix (Z-Y-X order)
    combined_rotation = R_z @ R_y @ R_x  # Matrix multiplication

    # the transformation matrix from camera frame to drone frame
    transformation_matrix = np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0]
    ])
    
    final_transformation_matrix = transformation_matrix @ combined_rotation

    # Convert the 3x3 Rotation Matrix to a 4x4 Transformation Matrix
    T_cd = np.eye(4)  # Start with an identity matrix
    T_cd[:3, :3] = final_transformation_matrix  # Replace the top-left 3x3 part with the rotation matrix
    # Translation remains [0, 0, 0], so no modification needed

    return T_cd

def load_drone_to_camera_frame_tf_matrix(meta_data_file_path, camera_angle=(0,0,0)):
    
    # X_camera = -Y_drone, Y_camera = -Z_drone, Z_camera = X_drone
    # Define the 3x3 Rotation Matrix (Drone Frame to Camera Frame)
    # transformation_matrix * [x_drone; y_drone; z_drone] = [x_camera; y_camera; z_camera]
    

    # transformation_matrix = np.array([
    #     [0, -1, 0],
    #     [0, 0, -1],
    #     [1, 0, 0]
    # ])

    
    camera_to_drone_tf_matrix = load_camera_to_drone_frame_tf_matrix(meta_data_file_path=meta_data_file_path, camera_angle=camera_angle)
    
    rotation_matrix = camera_to_drone_tf_matrix[:3, :3]
    R_inv = rotation_matrix.T
    
    # Convert the 3x3 Rotation Matrix to a 4x4 Transformation Matrix
    T_dc = np.eye(4)  # Start with an identity matrix
    T_dc[:3, :3] = R_inv  # Replace the top-left 3x3 part with the rotation matrix
    # Translation remains [0, 0, 0], so no modification needed

    return T_dc

def load_map_to_camera_tf_matrix(meta_data_file_path, camera_angle=(0,0,0)):
    
    map_to_drone_tf_matrix = load_map_to_drone_frame_tf_matrix(meta_data_file_path)
    drone_to_camera_tf_matrix = load_drone_to_camera_frame_tf_matrix(meta_data_file_path=meta_data_file_path, camera_angle=camera_angle)
    
    return np.dot(drone_to_camera_tf_matrix, map_to_drone_tf_matrix)

def load_camera_to_map_tf_matrix(meta_data_file_path, camera_angle=(0,0,0)):
    
    camera_to_drone_tf_matrix = load_camera_to_drone_frame_tf_matrix(meta_data_file_path=meta_data_file_path, camera_angle=camera_angle)
    drone_to_map_tf_matrix = load_drone_to_map_frame_tf_matrix(meta_data_file_path)
    
    
    return np.dot(drone_to_map_tf_matrix, camera_to_drone_tf_matrix)

def interpolate_rotation_matrix(R1, R2, time_step):
    """
    Interpolate between two rotation matrices using Spherical Linear Interpolation (SLERP)
    with scipy.spatial.transform
    """
    # Convert rotation matrices to quaternions using scipy
    q1 = Rotation.from_matrix(R1.cpu().numpy()).as_quat()  # returns [x, y, z, w]
    q2 = Rotation.from_matrix(R2.cpu().numpy()).as_quat()

    # Convert to unit quaternions
    q1_norm = np.linalg.norm(q1)
    q2_norm = np.linalg.norm(q2)
    if q1_norm > 0:
        q1 = q1 / q1_norm
    if q2_norm > 0:
        q2 = q2 / q2_norm

    # Compute dot product
    dot = np.clip(np.sum(q1 * q2), -1.0, 1.0)

    # If the dot product is negative, negate one of the quaternions
    if dot < 0:
        q2 = -q2
        dot = -dot

    # Perform spherical linear interpolation
    if dot > 0.9995:
        # If quaternions are very close, use linear interpolation
        result = q1 + time_step * (q2 - q1)
        result = result / np.linalg.norm(result)  # Normalize
    else:
        # Use spherical linear interpolation
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * time_step
        sin_theta = np.sin(theta)
        s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s2 = sin_theta / sin_theta_0
        result = (s1 * q1) + (s2 * q2)
        result = result / np.linalg.norm(result)

    # Convert interpolated quaternion back to rotation matrix using scipy
    rot_matrix = Rotation.from_quat(result).as_matrix()
    
    # Convert back to torch tensor on the correct device
    return torch.tensor(rot_matrix, device=R1.device, dtype=R1.dtype)

def generate_source_to_target_trajectory(source_w2c, target_w2c, num_frames, start_transition_frames = 0, end_transition_frames = 120, device="cuda"):
    """
    Generate camera trajectory from source to target pose
    Only transitions between start_transition_frames and end_transition_frames
    
    Uses proper camera center interpolation to ensure straight-line camera movement
    in world coordinates, avoiding curved paths caused by naive translation interpolation.
    """
    # Initialize output tensors
    w2cs = torch.zeros((1, num_frames, 4, 4), device=device)
    
    # Extract rotation and translation from source and target poses
    R1 = source_w2c[:3, :3]
    trans1 = source_w2c[:3, 3]
    R2 = target_w2c[:3, :3]
    trans2 = target_w2c[:3, 3]
    
    # Compute camera centers in world coordinates
    # Camera center C = -R^T * t (in world coordinates)
    C1 = -R1.T @ trans1  # Source camera center in world
    C2 = -R2.T @ trans2  # Target camera center in world

    # Generate frames
    for i in range(num_frames):
        if i < start_transition_frames:
            # Before transition: stay at source pose
            w2c = source_w2c.clone()
        elif i > end_transition_frames:
            # After transition: stay at target pose
            w2c = target_w2c.clone()
        else:
            # During transition: interpolate between source and target
            # Calculate interpolation factor within transition range
            transition_duration = end_transition_frames - start_transition_frames
            if transition_duration > 0:
                time_step = (i - start_transition_frames) / transition_duration
            else:
                time_step = 0.0
            
            # Clamp time_step to [0, 1]
            time_step = max(0.0, min(1.0, time_step))
            
            # Interpolate rotation using SLERP
            interpolated_rotation = interpolate_rotation_matrix(R1, R2, time_step)
            
            # Interpolate camera center in world coordinates (straight line)
            C_interp = (1 - time_step) * C1 + time_step * C2
            
            # Recompute translation from interpolated center and rotation
            # t = -R * C (in camera coordinates)
            translation = -interpolated_rotation @ C_interp
            
            # Construct transformation matrix
            w2c = torch.eye(4, device=device)
            w2c[:3, :3] = interpolated_rotation
            w2c[:3, 3] = translation
        
        w2cs[0, i] = w2c

    return w2cs

def generate_pixel_focused_trajectory(
    initial_w2c,
    initial_intrinsics,
    target_pixel,
    num_frames,
    movement_ratio,
    start_transition_frames,
    end_transition_frames,
    depth_map,
    device="cuda",
):
    """Generate camera trajectory that moves towards a specific pixel without rotation.
    
    Args:
        initial_w2c: Initial world-to-camera transform matrix [4, 4] or video sequence [T, 4, 4]
        initial_intrinsics: Camera intrinsics matrix [3, 3] or video sequence [T, 3, 3]
        target_pixel: Target pixel coordinates (x, y) in the target resolution
        num_frames: Number of frames to generate
        movement_ratio: How much to move towards the target (0-1)
        start_transition_frames: Frame number to start transitioning to target
        end_transition_frames: Frame number to end transitioning to target
        depth_map: Depth map tensor [1, H, W] or video depth [T, 1, H, W]
        device: Device to use for computations
    
    Returns:
        Tuple of (camera poses, camera intrinsics) for all frames
    """
    # Handle both single pose and video sequence inputs
    if initial_w2c.dim() == 2:  # Single pose [4, 4]
        is_video_input = False
        input_frames = 1
        initial_w2c = initial_w2c.unsqueeze(0)  # Add time dimension
        initial_intrinsics = initial_intrinsics.unsqueeze(0)  # Add time dimension
        depth_map = depth_map.unsqueeze(0)  # Add time dimension
    else:  # Video sequence [T, 4, 4]
        is_video_input = True
        input_frames = initial_w2c.shape[0]
    
    # Ensure we have enough input frames
    if is_video_input and input_frames < start_transition_frames:
        raise ValueError(f"Input video has {input_frames} frames but start_transition_frames is {start_transition_frames}")
    
    # Get the pose at start_transition_frames for target calculation
    transition_w2c = initial_w2c[start_transition_frames] if is_video_input else initial_w2c[0]
    transition_intrinsics = initial_intrinsics[start_transition_frames] if is_video_input else initial_intrinsics[0]
    transition_depth = depth_map[start_transition_frames] if is_video_input else depth_map[0]
    
    # Get camera position (in world space) from transition frame
    transition_position = -torch.matmul(
        transition_w2c[:3, :3].transpose(-2, -1),
        transition_w2c[:3, 3]
    )
    
    # Convert target pixel to camera space direction using transition intrinsics
    x, y = target_pixel
    
    # Convert to camera space using intrinsics
    fx = transition_intrinsics[0, 0]  # Focal length x
    fy = transition_intrinsics[1, 1]  # Focal length y
    cx = transition_intrinsics[0, 2]  # Principal point x
    cy = transition_intrinsics[1, 2]  # Principal point y

    ray_dir = torch.tensor([
        (x - cx) / fx,
        (y - cy) / fy,
        1.0
    ], device=device)
    
    # Transform ray direction to world space using transition frame
    world_ray_dir = torch.matmul(
        transition_w2c[:3, :3].transpose(-2, -1),  # Rotation from camera to world
        ray_dir
    )
    world_ray_dir = F.normalize(world_ray_dir, dim=-1)
    
    # Get depth at target pixel from depth map
    x, y = target_pixel
    depth_at_target = transition_depth[0, y, x]  # [1, H, W] -> scalar
    
    # Calculate target position in world space using actual depth and movement ratio
    target_position = transition_position + movement_ratio * depth_at_target * world_ray_dir
    
    # Generate interpolated positions
    positions = torch.zeros((num_frames, 3), device=device)
    
    # Create transition trajectory with three phases: initial hold, transition, and final hold
    for i in range(num_frames):
        if i < start_transition_frames:
            if is_video_input:
                # Use input video pose sequence before transition
                current_position = -torch.matmul(
                    initial_w2c[i][:3, :3].transpose(-2, -1),
                    initial_w2c[i][:3, 3]
                )
            else:
                # Stay at initial position
                current_position = -torch.matmul(
                    initial_w2c[0][:3, :3].transpose(-2, -1),
                    initial_w2c[0][:3, 3]
                )
        elif i <= end_transition_frames:
            # Linear interpolation during transition: t goes from 0 to 1 linearly
            t = (i - start_transition_frames) / (end_transition_frames - start_transition_frames)
            current_position = transition_position * (1 - t) + target_position * t
        else:
            # Stay at target position
            current_position = target_position
        
        positions[i] = current_position
    
    # Create w2c matrices for all frames
    w2cs = torch.zeros((num_frames, 4, 4), device=device)
    
    for i in range(num_frames):
        if i < start_transition_frames and is_video_input:
            # Use input video poses before transition
            w2cs[i] = initial_w2c[i]
        else:
            # Use transition frame rotation for all other frames
            w2cs[i, :3, :3] = transition_w2c[:3, :3]
            # Set translation (camera position in camera space)
            w2cs[i, :3, 3] = -torch.matmul(
                transition_w2c[:3, :3],
                positions[i].unsqueeze(-1)
            ).squeeze(-1)
            w2cs[i, 3, 3] = 1.0
    
    # Handle intrinsics
    intrinsics = torch.zeros((num_frames, 3, 3), device=device)
    for i in range(num_frames):
        if i < start_transition_frames and is_video_input:
            # Use input video intrinsics before transition
            intrinsics[i] = initial_intrinsics[i]
        else:
            # Use transition frame intrinsics for all other frames
            intrinsics[i] = transition_intrinsics
    
    # Add batch dimension of size 1 to match expected format [B=1, T, 4, 4] and [B=1, T, 3, 3]
    return w2cs.unsqueeze(0), intrinsics.unsqueeze(0)


def plot_camera_trajectory_comparison(initial_w2c, generated_w2cs, output_path, rendered_images_path):
    """Plot camera trajectory comparison with optimized single-loop processing."""
    # Convert to numpy and handle batch dimension
    initial_w2c = initial_w2c.cpu().numpy() if isinstance(initial_w2c, torch.Tensor) else initial_w2c
    generated_w2cs = generated_w2cs.cpu().numpy() if isinstance(generated_w2cs, torch.Tensor) else generated_w2cs
    if initial_w2c.ndim == 4: 
        initial_w2c = initial_w2c[0]
    if generated_w2cs.ndim == 4: 
        generated_w2cs = generated_w2cs[0]
    
    T = initial_w2c.shape[0]
    frames = np.arange(T)
    
    # Single loop: extract positions, orientations, and convert to Euler angles
    pos_init, pos_gen = np.zeros((T, 3)), np.zeros((T, 3))
    euler_init, euler_gen = np.zeros((T, 3)), np.zeros((T, 3))
    
    for i in range(T):
        # Extract and compute camera positions: -R^T * t
        R_init, t_init = initial_w2c[i, :3, :3], initial_w2c[i, :3, 3]
        R_gen, t_gen = generated_w2cs[i, :3, :3], generated_w2cs[i, :3, 3]
        
        pos_init[i] = -R_init.T @ t_init
        pos_gen[i] = -R_gen.T @ t_gen
        
        # Convert orientations to Euler angles
        euler_init[i] = Rotation.from_matrix(R_init.T).as_euler('xyz', degrees=True)
        euler_gen[i] = Rotation.from_matrix(R_gen.T).as_euler('xyz', degrees=True)
    
    # Create plots
    fig, axes = plt.subplots(6, 1, figsize=(10, 12))
    fig.suptitle('Camera Trajectory Comparison', fontsize=14)
    
    labels = ['Position X (m)', 'Position Y (m)', 'Position Z (m)', 'Roll (deg)', 'Pitch (deg)', 'Yaw (deg)']
    data_init = [pos_init[:, i] for i in range(3)] + [euler_init[:, i] for i in range(3)]
    data_gen = [pos_gen[:, i] for i in range(3)] + [euler_gen[:, i] for i in range(3)]
    
    for i, (ax, label, init_data, gen_data) in enumerate(zip(axes, labels, data_init, data_gen)):
        ax.plot(frames, init_data, 'b-', label='Initial', linewidth=2)
        ax.plot(frames, gen_data, 'r--', label='Generated', linewidth=2)
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
        if i == 5: ax.set_xlabel('Frame Index')
    
    plt.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"{Path(rendered_images_path).stem}_camera_trajectory_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Camera trajectory comparison saved to: {output_file}")
    return output_file


def plot_depth_statistics(depth_b1hw, output_path, rendered_images_path):
    """
    Create depth statistics visualization showing mean, max, min, and std of depth over frames.
    
    Args:
        depth_b1hw: Depth tensor of shape [T, 1, H, W] or [T, H, W]
        output_path: Directory to save the visualization
        rendered_images_path: Base name for the output file
    """
    # Convert to numpy and handle different input shapes
    if isinstance(depth_b1hw, torch.Tensor):
        depth_data = depth_b1hw.cpu().numpy()
    else:
        depth_data = depth_b1hw
    
    # Handle different input shapes
    if depth_data.ndim == 4:  # [T, 1, H, W]
        depth_data = depth_data.squeeze(1)  # [T, H, W]
    elif depth_data.ndim == 3:  # [T, H, W]
        pass
    else:
        raise ValueError(f"Expected depth data with 3 or 4 dimensions, got {depth_data.ndim}")
    
    T = depth_data.shape[0]
    frames = np.arange(T)
    
    # Calculate statistics for each frame
    depth_mean = np.mean(depth_data, axis=(1, 2))  # [T]
    depth_max = np.max(depth_data, axis=(1, 2))    # [T]
    depth_min = np.min(depth_data, axis=(1, 2))    # [T]
    depth_std = np.std(depth_data, axis=(1, 2))    # [T]
    
    # Create plots
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    fig.suptitle('Depth Statistics Over Frames', fontsize=14)
    
    labels = ['Mean Depth (m)', 'Max Depth (m)', 'Min Depth (m)', 'Std Depth (m)']
    data = [depth_mean, depth_max, depth_min, depth_std]
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (ax, label, data_series, color) in enumerate(zip(axes, labels, data, colors)):
        ax.plot(frames, data_series, color=color, linewidth=2, marker='o', markersize=3)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{label} Over Time')
        if i == 3:  # Last subplot
            ax.set_xlabel('Frame Index')
    
    plt.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"{Path(rendered_images_path).stem}_depth_statistics.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Depth statistics visualization saved to: {output_file}")
    return output_file