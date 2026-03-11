#!/usr/bin/env python3
# pyright: reportAttributeAccessIssue=false
"""
Monocular depth estimation utility.

Supports:
- MoGe (https://github.com/microsoft/MoGe) - default
- DepthAnythingV2 (https://github.com/DepthAnything/Depth-Anything-V2)

Outputs:
- depth .npy (float32) with invalid pixels set to 0 (so it can be used as a sparse depth map)
- optional visualization .png (colored depth)
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

import cv2 as _cv2  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
from moge.model.v1 import MoGeModel

# OpenCV typing stubs are often incomplete (pyright reports missing members).
# To keep the file type-check clean, access OpenCV APIs via getattr().
_cv2_any: Any = _cv2

_imread: Callable[..., Any] = getattr(_cv2_any, "imread")
_imwrite: Callable[..., Any] = getattr(_cv2_any, "imwrite")
_cvtColor: Callable[..., Any] = getattr(_cv2_any, "cvtColor")
_resize: Callable[..., Any] = getattr(_cv2_any, "resize")
_applyColorMap: Callable[..., Any] = getattr(_cv2_any, "applyColorMap", None)

_COLOR_BGR2RGB: int = int(getattr(_cv2_any, "COLOR_BGR2RGB"))
_COLORMAP_TURBO: int = int(getattr(_cv2_any, "COLORMAP_TURBO", getattr(_cv2_any, "COLORMAP_JET", 2)))


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _colorize_depth(depth_hw: np.ndarray, valid_mask_hw: np.ndarray) -> np.ndarray:
    """Return a BGR uint8 colormap image for quick inspection."""
    depth = depth_hw.astype(np.float32)
    valid = valid_mask_hw.astype(bool)

    if valid.sum() == 0:
        # all invalid -> black
        return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

    # Robust normalization
    d = depth[valid]
    d_min = float(np.percentile(d, 2))
    d_max = float(np.percentile(d, 98))
    if not np.isfinite(d_min) or not np.isfinite(d_max) or d_max <= d_min:
        d_min = float(np.min(d))
        d_max = float(np.max(d) + 1e-6)

    norm = (np.clip(depth, d_min, d_max) - d_min) / (d_max - d_min + 1e-8)
    norm_uint8 = (norm * 255.0).astype(np.uint8)
    if _applyColorMap is None:
        colored = np.repeat(norm_uint8[:, :, None], 3, axis=2)
    else:
        colored = _applyColorMap(norm_uint8, _COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def estimate_depth_moge(
    input_image_path: str,
    device: torch.device,
    checkpoint_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      depth_hw: float32 in meters (?) as produced by MoGe, resized to (height,width)
      mask_hw:  uint8 {0,1}, resized to (height,width)
    """
    img_bgr = _imread(input_image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Input image not found: {input_image_path}")
    img_rgb = _cvtColor(img_bgr, _COLOR_BGR2RGB)

    # Keep original resolution for output
    orig_h, orig_w = img_rgb.shape[:2]

    # MoGe inference resolution (matches existing GEN3C helper code)
    infer_h, infer_w = 720, 1280
    img_rgb_infer = _resize(img_rgb, (infer_w, infer_h))
    img_chw_0_1 = torch.tensor(img_rgb_infer / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)

    moge_model = MoGeModel.from_pretrained(checkpoint_id).to(device).eval()
    with torch.no_grad():
        out = moge_model.infer(img_chw_0_1)
        depth_hw_full = out["depth"]  # (infer_h, infer_w)
        mask_hw_full = out["mask"].to(torch.float32)  # (infer_h, infer_w), 0/1

    # Resize back to original resolution with sparse-aware averaging:
    # depth_resized = interp(depth*mask) / interp(mask), then apply resized mask.
    depth_11hw = depth_hw_full.unsqueeze(0).unsqueeze(0)
    mask_11hw = mask_hw_full.unsqueeze(0).unsqueeze(0)

    depth_weighted = depth_11hw * mask_11hw
    depth_weighted_rs = F.interpolate(depth_weighted, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    mask_rs = F.interpolate(mask_11hw, size=(orig_h, orig_w), mode="nearest")
    depth_rs = depth_weighted_rs / torch.clamp(mask_rs, min=1e-6)

    valid_rs = (mask_rs[0, 0] > 0.5)
    depth_hw = depth_rs[0, 0]
    depth_hw = torch.where(valid_rs, depth_hw, torch.zeros_like(depth_hw))

    depth_np = depth_hw.detach().cpu().numpy().astype(np.float32)
    mask_np = valid_rs.detach().cpu().numpy().astype(np.uint8)
    return depth_np, mask_np


def estimate_depth_dav2(
    input_image_path: str,
    device: torch.device,
    checkpoint_path: str | None = None,
    encoder: str = "vitl",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate depth using DepthAnythingV2.
    
    Returns:
      depth_hw: float32 in meters, resized to original image size
      mask_hw:  uint8 {0,1}, resized to original image size
    """
    # Find repo root to locate DepthAnythingV2 module
    script_path = Path(__file__).resolve()
    repo_root = None
    for parent in [script_path.parent] + list(script_path.parents):
        if (parent / "Depth-Estimation" / "Depth-Anything-V2").exists():
            repo_root = parent
            break
    if repo_root is None:
        raise FileNotFoundError(
            "Could not find Depth-Estimation/Depth-Anything-V2 directory. "
            "Please run: pixi run import-dav2"
        )
    
    sys.path.insert(0, str(repo_root / "Depth-Estimation" / "Depth-Anything-V2" / "metric_depth"))
    from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    max_depth = 80
    
    if checkpoint_path is None:
        checkpoint_path = repo_root / 'Depth-Estimation' / 'Depth-Anything-V2' / 'checkpoints' / f'depth_anything_v2_metric_hypersim_{encoder}.pth'
    else:
        checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"DepthAnythingV2 checkpoint not found: {checkpoint_path}\n"
            "Please run: pixi run download-dav2-checkpoints"
        )
    
    # Load model
    dav2_model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    dav2_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    dav2_model.to(device).eval()
    
    # Read image (BGR format as OpenCV does)
    img_bgr = _imread(input_image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Input image not found: {input_image_path}")
    
    orig_h, orig_w = img_bgr.shape[:2]
    
    # DepthAnythingV2's infer_image method handles preprocessing internally
    with torch.no_grad():
        depth_map = dav2_model.infer_image(img_bgr)  # Returns depth in original image size
    
    # Convert to numpy and ensure correct shape
    if isinstance(depth_map, torch.Tensor):
        depth_hw = depth_map.cpu().numpy().astype(np.float32)
    else:
        depth_hw = np.array(depth_map, dtype=np.float32)
    
    # Ensure it matches original image size
    if depth_hw.shape != (orig_h, orig_w):
        depth_11hw = torch.from_numpy(depth_hw).unsqueeze(0).unsqueeze(0).float()
        depth_rs = F.interpolate(
            depth_11hw,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False
        )
        depth_hw = depth_rs[0, 0].numpy().astype(np.float32)
    
    # All pixels are valid for DepthAnythingV2 (no mask provided by the model)
    mask_hw = np.ones((orig_h, orig_w), dtype=np.uint8)
    
    return depth_hw, mask_hw


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Monocular depth estimation. Supports MoGe (default) and DepthAnythingV2.")
    p.add_argument("--input_image_path", type=str, required=True)
    p.add_argument(
        "--depth_estimator",
        type=str,
        choices=["moge", "depthanythingv2"],
        default="moge",
        help="Depth estimation model to use: moge (default) or depthanythingv2",
    )
    p.add_argument(
        "--output_depth_npy",
        type=str,
        default=None,
        help="Optional override for depth output path. Defaults next to the input image.",
    )
    p.add_argument(
        "--output_mask_npy",
        type=str,
        default=None,
        help="Optional: also save the valid-mask as a .npy next to the image (or to this path).",
    )
    p.add_argument("--output_vis_png", type=str, default=None)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--moge_checkpoint", type=str, default="Ruicheng/moge-vitl", help="MoGe checkpoint ID (only used with --depth_estimator moge)")
    p.add_argument("--dav2_checkpoint", type=str, default=None, help="Path to DepthAnythingV2 checkpoint file (optional, defaults to repo checkpoints)")
    p.add_argument("--dav2_encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl", "vitg"], help="DepthAnythingV2 encoder size (only used with --depth_estimator depthanythingv2)")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    in_path = Path(args.input_image_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input image not found: {in_path}")

    # Default outputs: next to the image, derived from its stem.
    stem = in_path.stem
    estimator_suffix = "moge" if args.depth_estimator == "moge" else "dav2"
    default_depth = in_path.with_name(f"{stem}_{estimator_suffix}_depth.npy")

    out_depth = Path(args.output_depth_npy) if args.output_depth_npy else default_depth

    # Estimate depth based on chosen estimator
    if args.depth_estimator == "moge":
    depth_hw, mask_hw = estimate_depth_moge(
        input_image_path=args.input_image_path,
        device=device,
        checkpoint_id=args.moge_checkpoint,
    )
    elif args.depth_estimator == "depthanythingv2":
        depth_hw, mask_hw = estimate_depth_dav2(
            input_image_path=args.input_image_path,
            device=device,
            checkpoint_path=args.dav2_checkpoint,
            encoder=args.dav2_encoder,
        )
    else:
        raise ValueError(f"Unknown depth estimator: {args.depth_estimator}")

    _ensure_parent_dir(out_depth)
    np.save(out_depth, depth_hw.astype(np.float32))

    if args.output_mask_npy:
        out_mask = Path(args.output_mask_npy)
        _ensure_parent_dir(out_mask)
        np.save(out_mask, mask_hw.astype(np.uint8))

    if args.output_vis_png:
        out_vis = Path(args.output_vis_png)
        _ensure_parent_dir(out_vis)
        vis_bgr = _colorize_depth(depth_hw, mask_hw)
        _imwrite(str(out_vis), vis_bgr)


if __name__ == "__main__":
    main()


