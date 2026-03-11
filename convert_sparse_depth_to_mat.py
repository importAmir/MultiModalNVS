#!/usr/bin/env python3
"""
Inverse of convert_mat_to_sparse_depth.py

Converts a "sparse depth map" .npy back into a MATLAB .mat file containing a numeric 2D array.

Rules (mirrors convert_mat_to_sparse_depth.py):
- non-finite values are treated as invalid
- non-positive values (<= 0) are treated as invalid
- invalid pixels are written as --invalid_value_out (default: 0.0) in the output .mat

Notes
- This writes a classic MATLAB v5/v7 .mat via scipy.io.savemat (NOT v7.3/HDF5).
- Default variable name is "prediction_mean_pixel" (override via --var_name).

Examples
  python3 convert_sparse_depth_to_mat.py \
    --npy_path Samples/00750/00750_moge_depth.npy

  python3 convert_sparse_depth_to_mat.py \
    --npy_path Samples/00750/00750_moge_depth.npy \
    --out_path Samples/00750/00750_moge_depth.mat \
    --var_name prediction_mean_pixel
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np


def _squeeze_depth_array(a: np.ndarray) -> np.ndarray:
    """
    Accept common shapes:
    - (H, W)
    - (1, H, W) -> (H, W)
    """
    if a.ndim == 3 and a.shape[0] == 1:
        return a[0]
    return a


def convert_sparse_depth_to_mat_array(depth: np.ndarray, invalid_value_out: float) -> np.ndarray:
    """
    Convert arbitrary depth-like ndarray into a 2D float32 array suitable for saving to .mat.
    """
    d = np.asarray(depth)
    if np.iscomplexobj(d):
        d = np.abs(d)
    d = d.astype(np.float32, copy=False)
    d = _squeeze_depth_array(d)
    if d.ndim != 2:
        raise ValueError(f"Expected a 2D depth map (H,W) (or (1,H,W)), got shape={d.shape}")

    invalid_mask = (~np.isfinite(d)) | (d <= 0)
    out = d.copy()
    out[invalid_mask] = np.float32(invalid_value_out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy_path", required=True, help="Path to input .npy depth map")
    ap.add_argument(
        "--out_path",
        default=None,
        help="Optional explicit output .mat path. Default: same basename as input with .mat extension.",
    )
    ap.add_argument(
        "--var_name",
        default="prediction_mean_pixel",
        help='Variable name to store in the .mat file (default: "prediction_mean_pixel").',
    )
    ap.add_argument(
        "--invalid_value_out",
        type=float,
        default=0.0,
        help="Value to write for invalid pixels in the output .mat (default: 0).",
    )
    args = ap.parse_args()

    npy_path = os.path.expanduser(args.npy_path)
    if not os.path.exists(npy_path):
        print(f"ERROR: file not found: {npy_path}")
        return 2

    out_path = args.out_path
    if out_path is None:
        base, _ext = os.path.splitext(npy_path)
        out_path = base + ".mat"
    out_path = os.path.expanduser(out_path)

    try:
        import scipy.io  # type: ignore
    except ImportError:
        print("ERROR: scipy is required to write .mat files. Install it and retry:\n  pip install scipy")
        return 3

    depth = np.load(npy_path)
    mat_arr = convert_sparse_depth_to_mat_array(depth, invalid_value_out=args.invalid_value_out)

    payload: dict[str, Any] = {str(args.var_name): mat_arr}
    scipy.io.savemat(out_path, payload, do_compression=True)

    # Print quick sanity stats
    finite = np.isfinite(mat_arr)
    valid = finite & (mat_arr > 0)
    invalid = ~valid
    print(f"Loaded: {npy_path}")
    print(f"Output: {out_path}")
    print(f"Variable name: {args.var_name}")
    print(f"Output shape: {mat_arr.shape} dtype: {mat_arr.dtype}")
    print(f"valid_frac (depth>0): {float(np.mean(valid)):.6f}")
    print(f"invalid_frac: {float(np.mean(invalid)):.6f}  (invalid_value_out={args.invalid_value_out})")
    if np.any(valid):
        vals = mat_arr[valid].astype(np.float64, copy=False)
        qs = np.quantile(vals, [0.01, 0.1, 0.5, 0.9, 0.99])
        print(
            "valid depth stats: "
            f"min={float(np.min(vals)):.6g} p01={float(qs[0]):.6g} p10={float(qs[1]):.6g} "
            f"median={float(qs[2]):.6g} p90={float(qs[3]):.6g} p99={float(qs[4]):.6g} max={float(np.max(vals)):.6g}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


