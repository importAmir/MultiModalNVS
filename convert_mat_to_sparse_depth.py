#!/usr/bin/env python3
"""
Convert a MATLAB .mat array into a "sparse depth map" .npy for GEN3C-style pipelines.

What it does
- Loads a numeric 2D array from a .mat file (classic v5/v7 via scipy; v7.3 via h5py).
- Treats non-positive values as invalid depth (<= 0).
- Writes a float32 .npy with the SAME basename as the .mat (extension replaced by .npy).
- Keeps invalid pixels set to -1 by default (configurable).

Examples
  python3 convert_mat_to_sparse_depth.py \
    --mat_path Samples/prediction_mean_pixel_RadarTxNum5_VarTH50.mat

  # If you want invalid pixels as 0 (often used when mask is (depth > 0)):
  python3 convert_mat_to_sparse_depth.py \
    --mat_path Samples/prediction_mean_pixel_RadarTxNum5_VarTH50.mat \
    --invalid_value 0

  # If the .mat contains multiple arrays, specify the variable name:
  python3 convert_mat_to_sparse_depth.py \
    --mat_path Samples/prediction_mean_pixel_RadarTxNum5_VarTH50.mat \
    --var_name prediction_mean_pixel
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _load_v5_mat(mat_path: str) -> Optional[Dict[str, Any]]:
    try:
        import scipy.io  # type: ignore
    except ImportError:
        return None

    try:
        return scipy.io.loadmat(mat_path)
    except NotImplementedError:
        # Likely MATLAB v7.3 (HDF5)
        return None
    except (OSError, ValueError, TypeError):
        return None


def _load_v73_hdf5(mat_path: str) -> Optional[Tuple["h5py.File", Dict[str, Any]]]:
    try:
        import h5py  # type: ignore
    except ImportError:
        return None

    try:
        f = h5py.File(mat_path, "r")
    except OSError:
        return None

    def _to_dict(g) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k in g.keys():
            obj = g[k]
            if hasattr(obj, "keys"):
                out[k] = _to_dict(obj)
            else:
                out[k] = obj
        return out

    return f, _to_dict(f)


def _iter_named_arrays_from_loadmat(d: Dict[str, Any]) -> Iterable[Tuple[str, np.ndarray]]:
    for k, v in d.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray):
            yield k, v


def _iter_named_arrays_from_h5(root: Dict[str, Any], prefix: str = "") -> Iterable[Tuple[str, np.ndarray]]:
    for k, v in root.items():
        name = f"{prefix}{k}" if prefix == "" else f"{prefix}/{k}"
        if isinstance(v, dict):
            yield from _iter_named_arrays_from_h5(v, prefix=name)
        else:
            try:
                arr = np.array(v)
            except (TypeError, ValueError):
                continue
            yield name, arr


def _is_numeric_ndarray(x: Any) -> bool:
    return isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number)


def _choose_default_var(candidates: List[Tuple[str, np.ndarray]]) -> Tuple[str, np.ndarray]:
    """
    Heuristic: prefer 2D numeric arrays, then largest by element count.
    """
    numeric = [(n, a) for (n, a) in candidates if _is_numeric_ndarray(a)]
    if not numeric:
        raise ValueError("No numeric arrays found in .mat file.")
    two_d = [(n, a) for (n, a) in numeric if a.ndim == 2]
    pool = two_d if two_d else numeric
    pool.sort(key=lambda na: int(np.prod(na[1].shape)), reverse=True)
    return pool[0]


def _find_by_var_name(candidates: List[Tuple[str, np.ndarray]], var_name: str) -> Tuple[str, np.ndarray]:
    for n, a in candidates:
        if n == var_name:
            return n, a
    raise ValueError(f"Variable '{var_name}' not found. Available: {[n for n, _ in candidates]}")


def convert_to_sparse_depth(arr: np.ndarray, invalid_value: float) -> np.ndarray:
    """
    Rule: non-positive values (<= 0) are invalid.
    Output: float32, same shape as input.
    """
    a = np.asarray(arr)
    if np.iscomplexobj(a):
        a = np.abs(a)
    a = a.astype(np.float32, copy=False)

    # Treat NaN/inf as invalid too
    invalid_mask = (~np.isfinite(a)) | (a <= 0)
    out = a.copy()
    out[invalid_mask] = np.float32(invalid_value)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat_path", required=True, help="Path to input .mat file")
    ap.add_argument(
        "--var_name",
        default=None,
        help="MATLAB variable name to extract (optional; otherwise auto-detect a good candidate).",
    )
    ap.add_argument(
        "--invalid_value",
        type=float,
        default=-1.0,
        help="Value to write for invalid pixels (non-positive, NaN, inf). Default: -1.",
    )
    ap.add_argument(
        "--out_path",
        default=None,
        help="Optional explicit output .npy path. Default: same basename as input with .npy extension.",
    )
    args = ap.parse_args()

    mat_path = os.path.expanduser(args.mat_path)
    if not os.path.exists(mat_path):
        print(f"ERROR: file not found: {mat_path}")
        return 2

    out_path = args.out_path
    if out_path is None:
        base, _ext = os.path.splitext(mat_path)
        out_path = base + ".npy"
    out_path = os.path.expanduser(out_path)

    d = _load_v5_mat(mat_path)
    h5_file = None
    candidates: List[Tuple[str, np.ndarray]]
    if d is not None:
        candidates = list(_iter_named_arrays_from_loadmat(d))
    else:
        loaded = _load_v73_hdf5(mat_path)
        if loaded is None:
            print(
                "ERROR: Could not load .mat file.\n"
                "Install dependencies and retry:\n"
                "  pip install scipy h5py"
            )
            return 3
        h5_file, root = loaded
        candidates = list(_iter_named_arrays_from_h5(root))

    try:
        if args.var_name:
            name, arr = _find_by_var_name(candidates, args.var_name)
        else:
            name, arr = _choose_default_var(candidates)
    except ValueError as e:
        print(f"ERROR selecting array: {e}")
        if h5_file is not None:
            try:
                h5_file.close()
            except OSError:
                pass
        return 4

    if not _is_numeric_ndarray(arr):
        print(f"ERROR: selected '{name}' but it is not a numeric ndarray (dtype={getattr(arr, 'dtype', None)})")
        if h5_file is not None:
            try:
                h5_file.close()
            except OSError:
                pass
        return 5

    depth = convert_to_sparse_depth(arr, invalid_value=args.invalid_value)
    np.save(out_path, depth)

    # Print quick sanity stats
    finite = np.isfinite(depth)
    valid = finite & (depth > 0)
    invalid = ~valid
    print(f"Loaded: {mat_path}")
    print(f"Selected variable: {name}")
    print(f"Input shape: {arr.shape} dtype: {arr.dtype}")
    print(f"Output: {out_path}")
    print(f"Output shape: {depth.shape} dtype: {depth.dtype}")
    print(f"valid_frac (depth>0): {float(np.mean(valid)):.6f}")
    print(f"invalid_frac: {float(np.mean(invalid)):.6f}  (invalid_value={args.invalid_value})")
    if np.any(valid):
        vals = depth[valid].astype(np.float64, copy=False)
        qs = np.quantile(vals, [0.01, 0.1, 0.5, 0.9, 0.99])
        print(
            "valid depth stats: "
            f"min={float(np.min(vals)):.6g} p01={float(qs[0]):.6g} p10={float(qs[1]):.6g} "
            f"median={float(qs[2]):.6g} p90={float(qs[3]):.6g} p99={float(qs[4]):.6g} max={float(np.max(vals)):.6g}"
        )

    if h5_file is not None:
        try:
            h5_file.close()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


