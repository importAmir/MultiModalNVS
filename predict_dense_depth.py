"""
Predict a dense depth map from a sparse 3D point set (lidar, radar, or similar)
using angle-space locality + Gaussian kernel regression.

This is a Python port of `radarPointCloud_prediction_pixel_circle_TrVeloToCam_Zvalue.m`,
but generalized so the sparse input is NOT assumed to be radar.

Workflow:
- Load sparse points (x,y,z,...) from a binary file
- Transform points to camera frame using an extrinsic matrix (expects `Tr_velo_to_cam:` in calib)
- Convert sparse points to (azimuth, elevation) angles in degrees
- Convert each image pixel to its (azimuth, elevation) ray direction using camera intrinsics
- For each pixel, aggregate nearby sparse samples within a circular radius in angle-space
  using a Gaussian kernel (fast, binned, parallel, optionally GPU via PyTorch)

Outputs:
- dense depth map (default: camera Z depth)
- estimated per-pixel variance (weighted second central moment)
"""

from __future__ import annotations

import argparse
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[misc,assignment]

Pathish = Union[str, os.PathLike]


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def _make_bin_key(bx, by):
    bx64 = np.int64(bx)
    by64 = np.int64(by) & np.int64(0xFFFFFFFF)
    return (bx64 << np.int64(32)) | by64


def _decode_bin_key(key: int) -> Tuple[np.int32, np.int32]:
    bx = np.int32(np.int64(key) >> np.int64(32))
    by = np.int32(np.int64(key) & np.int64(0xFFFFFFFF))
    return bx, by


def load_sparse_points_bin(points_bin_path: Pathish, *, points_format: str = "vod7") -> np.ndarray:
    """
    Load sparse points from a binary file of float32.

    Supported formats:
    - vod7: VoD radar-style (N,7) floats: x,y,z,RCS,v_r,v_r_comp,t_id
    - xyz3: generic (N,3) floats: x,y,z
    """
    points_bin_path = Path(points_bin_path)
    data = np.fromfile(points_bin_path, dtype=np.float32)
    fmt = points_format.lower().strip()
    if fmt == "vod7":
        if data.size % 7 != 0:
            raise ValueError(f"File {points_bin_path} has {data.size} floats; expected multiple of 7 for vod7.")
        return data.reshape((-1, 7))
    if fmt == "xyz3":
        if data.size % 3 != 0:
            raise ValueError(f"File {points_bin_path} has {data.size} floats; expected multiple of 3 for xyz3.")
        return data.reshape((-1, 3))
    raise ValueError("points_format must be one of: vod7, xyz3")


def load_tr_velo_to_cam(calib_txt_path: Pathish) -> np.ndarray:
    """
    Parse `Tr_velo_to_cam:` from a KITTI-style calibration txt file.
    Returns a (3, 4) float64 matrix.

    Note: the key name is kept for compatibility with datasets, but the transform
    can represent any sensor-to-camera extrinsic.
    """
    calib_txt_path = Path(calib_txt_path)
    txt = calib_txt_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"Tr_velo_to_cam:\s*([-\d.eE+\s]+)", txt)
    if not m:
        raise ValueError(f"Could not find 'Tr_velo_to_cam:' in {calib_txt_path}")
    nums = [float(x) for x in m.group(1).strip().split()]
    if len(nums) < 12:
        raise ValueError(f"'Tr_velo_to_cam' in {calib_txt_path} has {len(nums)} numbers; expected 12.")
    return np.array(nums[:12], dtype=np.float64).reshape((3, 4))


def load_intrinsics_from_calib(
    calib_txt_path: Pathish,
    *,
    keys: Sequence[str] = ("P2", "K_02", "K", "camera_matrix"),
) -> Intrinsics:
    """
    Try to read camera intrinsics from a calib txt file.

    Supported formats:
    - 12 numbers (3x4 projection matrix): fx=P[0,0], fy=P[1,1], cx=P[0,2], cy=P[1,2]
    - 9 numbers  (3x3 intrinsic matrix): fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2]
    """
    calib_txt_path = Path(calib_txt_path)
    txt = calib_txt_path.read_text(encoding="utf-8", errors="ignore")

    for key in keys:
        m = re.search(rf"^{re.escape(key)}:\s*([-\d.eE+\s]+)$", txt, flags=re.MULTILINE)
        if not m:
            continue
        nums = [float(x) for x in m.group(1).strip().split()]
        if len(nums) >= 12:
            P = np.array(nums[:12], dtype=np.float64).reshape((3, 4))
            return Intrinsics(fx=float(P[0, 0]), fy=float(P[1, 1]), cx=float(P[0, 2]), cy=float(P[1, 2]))
        if len(nums) >= 9:
            K = np.array(nums[:9], dtype=np.float64).reshape((3, 3))
            return Intrinsics(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]))

    raise ValueError(f"Could not find intrinsics in {calib_txt_path} using keys={list(keys)}")


def transform_points_to_cam(xyz: np.ndarray, Tr_sensor_to_cam: np.ndarray) -> np.ndarray:
    """
    xyz: (N, 3) in sensor frame. Returns (N, 3) in camera frame.
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N,3), got {xyz.shape}")
    if Tr_sensor_to_cam.shape != (3, 4):
        raise ValueError(f"Tr_sensor_to_cam must be (3,4), got {Tr_sensor_to_cam.shape}")
    xyz1 = np.concatenate([xyz.astype(np.float64, copy=False), np.ones((xyz.shape[0], 1), dtype=np.float64)], axis=1)
    return (Tr_sensor_to_cam @ xyz1.T).T


def angles_and_depth_from_cam_xyz(
    xyz_cam: np.ndarray,
    *,
    depth_mode: str = "z",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute (az_deg, el_deg, depth) from camera-frame xyz.

    Convention used here:
    - az = atan2(x_c, z_c)
    - el = atan2(-y_c, sqrt(x_c^2 + z_c^2))
    """
    x_c = xyz_cam[:, 0]
    y_c = xyz_cam[:, 1]
    z_c = xyz_cam[:, 2]

    az_deg = np.degrees(np.arctan2(x_c, z_c))
    el_deg = np.degrees(np.arctan2(-y_c, np.sqrt(x_c * x_c + z_c * z_c)))

    if depth_mode != "z":
        raise ValueError("depth_mode must be 'z'")
    depth = z_c

    return az_deg.astype(np.float32), el_deg.astype(np.float32), depth.astype(np.float32)


def pixel_angles_from_intrinsics(
    image_hw: Tuple[int, int],
    K: Intrinsics,
    *,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Convert each pixel to (azimuth_deg, elevation_deg) using the same convention as the point-angle conversion.
    Returns flattened arrays and the effective (h, w) used (after stride).
    """
    h, w = image_hw
    if stride < 1:
        raise ValueError("stride must be >= 1")

    u = np.arange(0, w, stride, dtype=np.float32)
    v = np.arange(0, h, stride, dtype=np.float32)
    uGrid, vGrid = np.meshgrid(u, v)

    x_n = (uGrid - K.cx) / K.fx
    y_n = (K.cy - vGrid) / K.fy

    dx = x_n
    dy = y_n
    dz = np.ones_like(dx)

    norms = np.sqrt(dx * dx + dy * dy + dz * dz)
    dx /= norms
    dy /= norms
    dz /= norms

    azimuth = np.arctan2(dx, dz)
    elevation = np.arctan2(dy, np.sqrt(dx * dx + dz * dz))

    az_deg = np.degrees(azimuth).astype(np.float32).reshape(-1)
    el_deg = np.degrees(elevation).astype(np.float32).reshape(-1)

    return az_deg, el_deg, (uGrid.shape[0], uGrid.shape[1])


def _build_key_to_span(sorted_keys: np.ndarray) -> dict[int, Tuple[int, int]]:
    uniq, start, counts = np.unique(sorted_keys, return_index=True, return_counts=True)
    spans: dict[int, Tuple[int, int]] = {}
    for k, s, c in zip(uniq.tolist(), start.tolist(), counts.tolist()):
        spans[int(k)] = (int(s), int(s + c))
    return spans


def _dense_from_sparse(
    *,
    num_targets: int,
    indices: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    invalid_value: float,
    var_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    out_mean = np.full((num_targets,), np.float32(invalid_value), dtype=np.float32)
    out_var = np.full((num_targets,), np.float32("inf"), dtype=np.float32)
    if indices.size:
        indices = indices.astype(np.int64, copy=False)
        better = var < out_var[indices]
        idx_better = indices[better]
        out_mean[idx_better] = mean[better].astype(np.float32, copy=False)
        out_var[idx_better] = var[better].astype(np.float32, copy=False)
    valid_mask = np.isfinite(out_var) & (out_var <= np.float32(var_threshold)) & np.isfinite(out_mean)
    return out_mean, out_var, valid_mask


def _progress(iterable, *, total: Optional[int], desc: str, enable: bool):
    if not enable or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def _compute_shift_depth_value(depth: np.ndarray, spec: str) -> float:
    """
    Compute a numeric shift value from a string spec.

    Accepted specs:
    - "auto" (default): median of finite, positive depth samples
    - "median": median of finite, positive depth samples
    - "mean": mean of finite, positive depth samples
    - "none" / "off" / "0": 0.0
    - any numeric string: that constant
    """
    s = str(spec).strip().lower()
    if s in {"none", "off", "no", "false", "0"}:
        return 0.0
    if s in {"auto", "median", "mean"}:
        d = depth.astype(np.float64, copy=False)
        m = np.isfinite(d) & (d > 0)
        if not np.any(m):
            return 0.0
        d = d[m]
        if s in {"auto", "median"}:
            return float(np.median(d))
        return float(np.mean(d))
    try:
        return float(s)
    except ValueError as e:
        raise ValueError(f"Invalid --shift-depth-value '{spec}'. Use auto/median/mean/none or a number.") from e


def predict_depth_kernel_circle_binned_sparse(
    *,
    points_az_deg: np.ndarray,
    points_el_deg: np.ndarray,
    points_depth: np.ndarray,
    target_az_deg: np.ndarray,
    target_el_deg: np.ndarray,
    locality_radius_deg: float,
    kernel_length_scale_deg: float,
    shift_depth_value: float = 0.0,
    bin_size_deg: Optional[float] = None,
    num_workers: int = 0,
    backend: str = "numpy",
    device: str = "cpu",
    num_shards: int = 1,
    shard_rank: int = 0,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Fast dense depth prediction using Gaussian kernel regression on a circular neighborhood
    in angle-space (azimuth/elevation).

    Returns:
    - indices: (M,) int64 pixel indices (flattened)
    - mean: (M,) float32 predicted depth
    - var: (M,) float32 predicted variance
    - num_targets: int = len(target_az_deg)
    """
    if points_az_deg.shape != points_el_deg.shape or points_az_deg.shape != points_depth.shape:
        raise ValueError("points_az_deg, points_el_deg, points_depth must have the same shape")
    if target_az_deg.shape != target_el_deg.shape:
        raise ValueError("target_az_deg and target_el_deg must have the same shape")
    if locality_radius_deg <= 0:
        raise ValueError("locality_radius_deg must be > 0")
    if kernel_length_scale_deg <= 0:
        raise ValueError("kernel_length_scale_deg must be > 0")
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_rank < 0 or shard_rank >= num_shards:
        raise ValueError("shard_rank must be in [0, num_shards)")

    if bin_size_deg is None:
        bin_size_deg = float(locality_radius_deg)
    if bin_size_deg <= 0:
        raise ValueError("bin_size_deg must be > 0")

    points_depth_shifted = points_depth.astype(np.float32, copy=False) - np.float32(shift_depth_value)

    # Bin sparse points.
    p_bx = np.floor(points_az_deg / bin_size_deg).astype(np.int32)
    p_by = np.floor(points_el_deg / bin_size_deg).astype(np.int32)
    points_keys = _make_bin_key(p_bx, p_by).astype(np.int64)
    points_order = np.argsort(points_keys)
    points_keys_sorted = points_keys[points_order]
    points_key_to_span = _build_key_to_span(points_keys_sorted)

    # Bin targets (pixels).
    t_bx = np.floor(target_az_deg / bin_size_deg).astype(np.int32)
    t_by = np.floor(target_el_deg / bin_size_deg).astype(np.int32)
    target_keys = _make_bin_key(t_bx, t_by).astype(np.int64)
    target_order = np.argsort(target_keys)
    target_keys_sorted = target_keys[target_order]
    target_key_to_span = _build_key_to_span(target_keys_sorted)

    keys_all: List[int] = list(target_key_to_span.keys())
    keys_list = keys_all[shard_rank::num_shards] if num_shards > 1 else keys_all

    r2 = float(locality_radius_deg) * float(locality_radius_deg)
    ls2 = float(kernel_length_scale_deg) * float(kernel_length_scale_deg)
    max_bin_offset = int(math.ceil(locality_radius_deg / bin_size_deg))

    use_torch = backend.lower() == "torch"
    torch = None
    torch_device = None
    if use_torch:
        try:
            import torch as _torch  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("backend='torch' requested but PyTorch is not available") from e
        torch = _torch
        torch_device = torch.device(device)

    def process_keys(keys: Iterable[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx_chunks: List[np.ndarray] = []
        mean_chunks: List[np.ndarray] = []
        var_chunks: List[np.ndarray] = []

        for key in keys:
            s_t, e_t = target_key_to_span[int(key)]
            pix_idx = target_order[s_t:e_t]

            bx, by = _decode_bin_key(int(key))
            cand_slices = []
            for dx in range(-max_bin_offset, max_bin_offset + 1):
                for dy in range(-max_bin_offset, max_bin_offset + 1):
                    nkey = int(_make_bin_key(int(bx) + dx, int(by) + dy))
                    span = points_key_to_span.get(nkey)
                    if span is not None:
                        cand_slices.append(points_order[span[0] : span[1]])

            if not cand_slices:
                continue

            cand_idx = np.concatenate(cand_slices, axis=0)
            az_c = points_az_deg[cand_idx].astype(np.float32, copy=False)
            el_c = points_el_deg[cand_idx].astype(np.float32, copy=False)
            d_c = points_depth_shifted[cand_idx].astype(np.float32, copy=False)

            az_t = target_az_deg[pix_idx].astype(np.float32, copy=False)
            el_t = target_el_deg[pix_idx].astype(np.float32, copy=False)

            if use_torch:
                az_c_t = torch.as_tensor(az_c, device=torch_device)
                el_c_t = torch.as_tensor(el_c, device=torch_device)
                d_c_t = torch.as_tensor(d_c, device=torch_device)
                az_t_t = torch.as_tensor(az_t, device=torch_device)
                el_t_t = torch.as_tensor(el_t, device=torch_device)

                da = az_t_t[:, None] - az_c_t[None, :]
                de = el_t_t[:, None] - el_c_t[None, :]
                d2 = da * da + de * de
                within = d2 <= r2
                if not bool(within.any().item()):
                    continue

                w = torch.exp(-0.5 * d2 / ls2) * within.to(dtype=torch.float32)
                sumw = w.sum(dim=1)
                good = sumw > 0
                if not bool(good.any().item()):
                    continue

                w_good = w[good]
                sumw_good = sumw[good]
                mean_shifted = (w_good @ d_c_t) / sumw_good
                second_moment = (w_good @ (d_c_t * d_c_t)) / sumw_good
                var = torch.clamp(second_moment - mean_shifted * mean_shifted, min=0.0)

                idx_np = pix_idx[good.cpu().numpy()].astype(np.int64, copy=False)
                mean_np = (mean_shifted + float(shift_depth_value)).detach().cpu().numpy().astype(np.float32, copy=False)
                var_np = var.detach().cpu().numpy().astype(np.float32, copy=False)
            else:
                da = az_t[:, None] - az_c[None, :]
                de = el_t[:, None] - el_c[None, :]
                d2 = da * da + de * de
                within = d2 <= r2
                if not np.any(within):
                    continue

                w = np.exp(-0.5 * d2 / ls2, dtype=np.float32) * within.astype(np.float32)
                sumw = w.sum(axis=1)
                good = sumw > 0
                if not np.any(good):
                    continue

                w_good = w[good]
                sumw_good = sumw[good]
                mean_shifted = (w_good @ d_c) / sumw_good
                second_moment = (w_good @ (d_c * d_c)) / sumw_good
                var_np = np.maximum(second_moment - mean_shifted * mean_shifted, 0.0).astype(np.float32)
                mean_np = (mean_shifted + np.float32(shift_depth_value)).astype(np.float32)
                idx_np = pix_idx[good].astype(np.int64, copy=False)

            idx_chunks.append(idx_np)
            mean_chunks.append(mean_np)
            var_chunks.append(var_np)

        if not idx_chunks:
            empty = np.empty((0,), dtype=np.int64)
            return empty, empty.astype(np.float32), empty.astype(np.float32)
        return (
            np.concatenate(idx_chunks, axis=0),
            np.concatenate(mean_chunks, axis=0).astype(np.float32, copy=False),
            np.concatenate(var_chunks, axis=0).astype(np.float32, copy=False),
        )

    if num_workers is None or num_workers <= 0:
        idx, mean, var = process_keys(_progress(keys_list, total=len(keys_list), desc="Kernel keys", enable=show_progress))
    else:
        num_workers = int(num_workers)
        print(f"[parallel] Kernel: using {num_workers} worker threads for {len(keys_list)} keys")
        chunk = max(1, len(keys_list) // (num_workers * 8))
        chunks = [keys_list[i : i + chunk] for i in range(0, len(keys_list), chunk)]
        pbar = tqdm(total=len(keys_list), desc="Kernel keys") if (show_progress and tqdm is not None) else None
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(process_keys, c) for c in chunks]
            results = []
            for fut, c in zip(futures, chunks):
                r = fut.result()
                results.append(r)
                if pbar is not None:
                    pbar.update(len(c))
        if pbar is not None:
            pbar.close()
        idx = np.concatenate([r[0] for r in results if r[0].size], axis=0) if results else np.empty((0,), np.int64)
        mean = (
            np.concatenate([r[1] for r in results if r[1].size], axis=0).astype(np.float32, copy=False)
            if results
            else np.empty((0,), np.float32)
        )
        var = (
            np.concatenate([r[2] for r in results if r[2].size], axis=0).astype(np.float32, copy=False)
            if results
            else np.empty((0,), np.float32)
        )

    return idx, mean, var, int(target_az_deg.shape[0])


def _rbf_cov_2d(a: float, length_scale: float, az: np.ndarray, el: np.ndarray) -> np.ndarray:
    az = az.astype(np.float64, copy=False)
    el = el.astype(np.float64, copy=False)
    da = az[:, None] - az[None, :]
    de = el[:, None] - el[None, :]
    d2 = da * da + de * de
    return (float(a) * float(a)) * np.exp(-0.5 * d2 / (float(length_scale) * float(length_scale)))


def _rbf_cross_2d(a: float, length_scale: float, az_local: np.ndarray, el_local: np.ndarray, az_t: float, el_t: float) -> np.ndarray:
    az_local = az_local.astype(np.float64, copy=False)
    el_local = el_local.astype(np.float64, copy=False)
    da = az_local - float(az_t)
    de = el_local - float(el_t)
    d2 = da * da + de * de
    return (float(a) * float(a)) * np.exp(-0.5 * d2 / (float(length_scale) * float(length_scale)))


def _golden_section_minimize(
    f,
    a: float,
    b: float,
    *,
    tol: float = 1e-4,
    max_iter: int = 64,
) -> float:
    """
    Bounded 1D minimization (fminbnd-like) using golden-section search.
    """
    gr = (math.sqrt(5.0) + 1.0) / 2.0
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    fc = f(c)
    fd = f(d)
    for _ in range(max_iter):
        if abs(b - a) <= tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) / gr
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) / gr
            fd = f(d)
    return float(c if fc < fd else d)


def _gp_neg_log_likelihood(
    *,
    a: float,
    length_scale: float,
    az_local: np.ndarray,
    el_local: np.ndarray,
    y: np.ndarray,
    sigma_mismatch: float,
) -> float:
    """
    Negative log likelihood for GP hyperparameter search (numerically stable, log space).
    """
    n = int(y.shape[0])
    K = _rbf_cov_2d(a, length_scale, az_local, el_local)
    K.flat[:: n + 1] += float(sigma_mismatch) * float(sigma_mismatch)
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return float("inf")
    y64 = y.astype(np.float64, copy=False)
    v = np.linalg.solve(L, y64)
    alpha = np.linalg.solve(L.T, v)
    quad = float(y64.T @ alpha)
    logdet = 2.0 * float(np.log(np.diag(L)).sum())
    return 0.5 * quad + 0.5 * logdet + 0.5 * n * math.log(2.0 * math.pi)


def predict_depth_local_gp_mle_binned_sparse(
    *,
    points_az_deg: np.ndarray,
    points_el_deg: np.ndarray,
    points_depth: np.ndarray,
    target_az_deg: np.ndarray,
    target_el_deg: np.ndarray,
    locality_radius_deg: float,
    gaussian_std: float,
    sigma_mismatch: float,
    length_scale_min: float = 0.1,
    length_scale_max: float = 10.0,
    opt_tol: float = 1e-3,
    opt_max_iter: int = 64,
    shift_depth_value: float = 0.0,
    bin_size_deg: Optional[float] = None,
    num_workers: int = 0,
    num_shards: int = 1,
    shard_rank: int = 0,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Local GP prediction with per-target length-scale optimized by marginal likelihood:
    - For each target pixel: pick local samples within radius
    - Optimize length scale with bounded 1D search
    - Compute conditional Gaussian mean/variance

    Returns indices/mean/var for ALL pixels covered by this shard.
    """
    if points_az_deg.shape != points_el_deg.shape or points_az_deg.shape != points_depth.shape:
        raise ValueError("points_az_deg, points_el_deg, points_depth must have the same shape")
    if target_az_deg.shape != target_el_deg.shape:
        raise ValueError("target_az_deg and target_el_deg must have the same shape")
    if locality_radius_deg <= 0:
        raise ValueError("locality_radius_deg must be > 0")
    if gaussian_std <= 0:
        raise ValueError("gaussian_std must be > 0")
    if sigma_mismatch <= 0:
        raise ValueError("sigma_mismatch must be > 0")
    if length_scale_min <= 0 or length_scale_max <= length_scale_min:
        raise ValueError("length_scale_min/max must define a valid interval")
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_rank < 0 or shard_rank >= num_shards:
        raise ValueError("shard_rank must be in [0, num_shards)")

    if bin_size_deg is None:
        bin_size_deg = float(locality_radius_deg)
    if bin_size_deg <= 0:
        raise ValueError("bin_size_deg must be > 0")

    a = float(gaussian_std)
    sigma2 = float(sigma_mismatch) * float(sigma_mismatch)
    r2 = float(locality_radius_deg) * float(locality_radius_deg)
    max_bin_offset = int(math.ceil(locality_radius_deg / bin_size_deg))

    points_depth_shifted = points_depth.astype(np.float32, copy=False) - np.float32(shift_depth_value)

    # Bin sparse points.
    p_bx = np.floor(points_az_deg / bin_size_deg).astype(np.int32)
    p_by = np.floor(points_el_deg / bin_size_deg).astype(np.int32)
    points_keys = _make_bin_key(p_bx, p_by).astype(np.int64)
    points_order = np.argsort(points_keys)
    points_keys_sorted = points_keys[points_order]
    points_key_to_span = _build_key_to_span(points_keys_sorted)

    # Bin targets.
    t_bx = np.floor(target_az_deg / bin_size_deg).astype(np.int32)
    t_by = np.floor(target_el_deg / bin_size_deg).astype(np.int32)
    target_keys = _make_bin_key(t_bx, t_by).astype(np.int64)
    target_order = np.argsort(target_keys)
    target_keys_sorted = target_keys[target_order]
    target_key_to_span = _build_key_to_span(target_keys_sorted)

    keys_all: List[int] = list(target_key_to_span.keys())
    keys_list = keys_all[shard_rank::num_shards] if num_shards > 1 else keys_all

    def process_keys(keys: Iterable[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx_chunks: List[np.ndarray] = []
        mean_chunks: List[np.ndarray] = []
        var_chunks: List[np.ndarray] = []

        for key in keys:
            s_t, e_t = target_key_to_span[int(key)]
            pix_idx = target_order[s_t:e_t].astype(np.int64, copy=False)

            bx, by = _decode_bin_key(int(key))
            cand_slices = []
            for dx in range(-max_bin_offset, max_bin_offset + 1):
                for dy in range(-max_bin_offset, max_bin_offset + 1):
                    nkey = int(_make_bin_key(int(bx) + dx, int(by) + dy))
                    span = points_key_to_span.get(nkey)
                    if span is not None:
                        cand_slices.append(points_order[span[0] : span[1]])

            # Default if there are no candidate samples at all: mean=shift, var=a^2.
            if not cand_slices:
                idx_chunks.append(pix_idx)
                mean_chunks.append(np.full((pix_idx.shape[0],), np.float32(shift_depth_value), dtype=np.float32))
                var_chunks.append(np.full((pix_idx.shape[0],), np.float32(a * a), dtype=np.float32))
                continue

            cand_idx = np.concatenate(cand_slices, axis=0)
            az_c = points_az_deg[cand_idx].astype(np.float32, copy=False)
            el_c = points_el_deg[cand_idx].astype(np.float32, copy=False)
            y_c = points_depth_shifted[cand_idx].astype(np.float32, copy=False)

            az_t_all = target_az_deg[pix_idx].astype(np.float32, copy=False)
            el_t_all = target_el_deg[pix_idx].astype(np.float32, copy=False)

            out_mean = np.empty((pix_idx.shape[0],), dtype=np.float32)
            out_var = np.empty((pix_idx.shape[0],), dtype=np.float32)

            # Process pixels in chunks to limit (M x C) temporary allocations.
            chunk_sz = 512
            for start in range(0, pix_idx.shape[0], chunk_sz):
                end = min(pix_idx.shape[0], start + chunk_sz)
                az_t = az_t_all[start:end]
                el_t = el_t_all[start:end]

                da = az_t[:, None] - az_c[None, :]
                de = el_t[:, None] - el_c[None, :]
                d2 = da * da + de * de
                within = d2 <= r2

                for i in range(end - start):
                    m = within[i]
                    if not np.any(m):
                        out_mean[start + i] = np.float32(shift_depth_value)
                        out_var[start + i] = np.float32(a * a)
                        continue

                    az_l = az_c[m]
                    el_l = el_c[m]
                    y_l = y_c[m]
                    n_local = int(y_l.shape[0])

                    def obj(lval: float) -> float:
                        return _gp_neg_log_likelihood(
                            a=a,
                            length_scale=float(lval),
                            az_local=az_l,
                            el_local=el_l,
                            y=y_l,
                            sigma_mismatch=float(sigma_mismatch),
                        )

                    opt_l = _golden_section_minimize(
                        obj,
                        float(length_scale_min),
                        float(length_scale_max),
                        tol=float(opt_tol),
                        max_iter=int(opt_max_iter),
                    )

                    # GP prediction (conditional mean/variance).
                    Kuu = _rbf_cov_2d(a, opt_l, az_l, el_l)
                    Kuu.flat[:: n_local + 1] += sigma2
                    try:
                        L = np.linalg.cholesky(Kuu)
                    except np.linalg.LinAlgError:
                        out_mean[start + i] = np.float32(shift_depth_value)
                        out_var[start + i] = np.float32(a * a)
                        continue

                    y64 = y_l.astype(np.float64, copy=False)
                    v = np.linalg.solve(L, y64)
                    alpha = np.linalg.solve(L.T, v)

                    kuv = _rbf_cross_2d(a, opt_l, az_l, el_l, float(az_t[i]), float(el_t[i])).astype(np.float64, copy=False)
                    mean_shifted = float(kuv.T @ alpha)

                    w = np.linalg.solve(L, kuv)
                    var = float((a * a + sigma2) - (w.T @ w))
                    if var < 0:
                        var = 0.0

                    out_mean[start + i] = np.float32(mean_shifted + float(shift_depth_value))
                    out_var[start + i] = np.float32(var)

            idx_chunks.append(pix_idx)
            mean_chunks.append(out_mean)
            var_chunks.append(out_var)

        if not idx_chunks:
            empty = np.empty((0,), dtype=np.int64)
            return empty, empty.astype(np.float32), empty.astype(np.float32)

        return (
            np.concatenate(idx_chunks, axis=0),
            np.concatenate(mean_chunks, axis=0).astype(np.float32, copy=False),
            np.concatenate(var_chunks, axis=0).astype(np.float32, copy=False),
        )

    if num_workers is None or num_workers <= 0:
        idx, mean, var = process_keys(_progress(keys_list, total=len(keys_list), desc="GP keys", enable=show_progress))
    else:
        num_workers = int(num_workers)
        print(f"[parallel] GP MLE: using {num_workers} worker threads for {len(keys_list)} keys")
        chunk = max(1, len(keys_list) // (num_workers * 8))
        chunks = [keys_list[i : i + chunk] for i in range(0, len(keys_list), chunk)]
        pbar = tqdm(total=len(keys_list), desc="GP keys") if (show_progress and tqdm is not None) else None
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = {ex.submit(process_keys, c): len(c) for c in chunks}
            results = []
            for fut in as_completed(futures):
                results.append(fut.result())
                if pbar is not None:
                    pbar.update(futures[fut])
        if pbar is not None:
            pbar.close()

        idx = np.concatenate([r[0] for r in results if r[0].size], axis=0) if results else np.empty((0,), np.int64)
        mean = (
            np.concatenate([r[1] for r in results if r[1].size], axis=0).astype(np.float32, copy=False)
            if results
            else np.empty((0,), np.float32)
        )
        var = (
            np.concatenate([r[2] for r in results if r[2].size], axis=0).astype(np.float32, copy=False)
            if results
            else np.empty((0,), np.float32)
        )

    return idx, mean, var, int(target_az_deg.shape[0])


def merge_shards(
    shard_files: Sequence[Pathish],
    *,
    output: Pathish,
    invalid_value: float = -1.0,
    var_threshold: float = float("inf"),
) -> int:
    shard_files_p = [Path(p) for p in shard_files]
    if not shard_files_p:
        raise SystemExit("merge_shards: provide at least one shard file.")

    first = np.load(shard_files_p[0], allow_pickle=False)
    hw_eff = tuple(int(x) for x in first["hw_eff"].tolist())
    num_targets = int(hw_eff[0] * hw_eff[1])

    out_mean = np.full((num_targets,), np.float32(invalid_value), dtype=np.float32)
    out_var = np.full((num_targets,), np.float32("inf"), dtype=np.float32)

    for sf in shard_files_p:
        d = np.load(sf, allow_pickle=False)
        if tuple(int(x) for x in d["hw_eff"].tolist()) != hw_eff:
            raise ValueError(f"Shard {sf} has different hw_eff.")
        idx = d["indices"].astype(np.int64, copy=False)
        mean = d["mean"].astype(np.float32, copy=False)
        var = d["var"].astype(np.float32, copy=False)
        if idx.size:
            better = var < out_var[idx]
            idx_better = idx[better]
            out_mean[idx_better] = mean[better]
            out_var[idx_better] = var[better]

    valid = np.isfinite(out_var) & (out_var <= np.float32(var_threshold)) & np.isfinite(out_mean)
    depth_map = out_mean.reshape(hw_eff)
    var_map = out_var.reshape(hw_eff)
    valid_map = valid.reshape(hw_eff)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, depth=depth_map, variance=var_map, valid=valid_map, hw_eff=np.array(hw_eff, np.int32))

    print(f"Merged: {out_path}")
    print(f"Output shape: {depth_map.shape}")
    print(f"Valid pixels: {int(valid_map.sum())} / {valid_map.size}")
    return 0


def _sidecar_output_paths(depth_output_npy: Path) -> tuple[Path, Path, Path]:
    """
    Given a depth output path (typically *_depth.npy), derive sidecar paths.
    """
    if depth_output_npy.suffix.lower() != ".npy":
        depth_output_npy = depth_output_npy.with_suffix(".npy")
    var_path = depth_output_npy.with_name(depth_output_npy.stem + "_variance.npy")
    valid_path = depth_output_npy.with_name(depth_output_npy.stem + "_valid.npy")
    return depth_output_npy, var_path, valid_path


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Predict dense depth from a sparse point cloud (fast, sharded, optional GPU).")

    # Either provide explicit file paths...
    p.add_argument("--points-bin", type=str, default=None, dest="points_bin", help="Path to points .bin.")
    p.add_argument("--calib-txt", type=str, default=None, help="Path to calib .txt containing Tr_velo_to_cam.")
    p.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        required=True,
        metavar=("H", "W"),
        help="Target image size (height width).",
    )

    # Input format.
    p.add_argument("--points-format", choices=["vod7", "xyz3"], default="vod7", help="Binary format for points .bin.")

    # Camera intrinsics.
    p.add_argument(
        "--intrinsics-source",
        choices=["auto", "calib", "args"],
        default="auto",
        help="Where to get fx/fy/cx/cy: auto (prefer calib), calib (required), or args.",
    )
    p.add_argument(
        "--intrinsics-keys",
        type=str,
        default="P2,K_02,K,camera_matrix",
        help="Comma-separated calib keys to try for intrinsics (e.g. P2,K_02,K).",
    )
    p.add_argument("--fx", type=float, default=None, help="Override fx (used when --intrinsics-source args, or as fallback).")
    p.add_argument("--fy", type=float, default=None, help="Override fy (used when --intrinsics-source args, or as fallback).")
    p.add_argument("--cx", type=float, default=None, help="Override cx (used when --intrinsics-source args, or as fallback).")
    p.add_argument("--cy", type=float, default=None, help="Override cy (used when --intrinsics-source args, or as fallback).")

    # Prediction parameters.
    p.add_argument(
        "--prediction-model",
        type=str,
        default="local_gp_mle",
        help=(
            "Prediction model. "
            "local_gp_mle = local Gaussian Process with per-target length-scale optimized by marginal likelihood; "
            "kernel = fast Gaussian kernel regression."
        ),
    )
    p.add_argument("--depth-mode", choices=["z"], default="z", help="Depth is camera Z.")
    p.add_argument("--locality-radius-deg", type=float, default=2.0)
    p.add_argument("--kernel-length-scale-deg", type=float, default=0.75)
    p.add_argument("--bin-size-deg", type=float, default=None, help="Angle bin size (default: locality radius).")
    p.add_argument(
        "--shift-depth-value",
        type=str,
        default="auto",
        help="Depth centering shift: auto/median/mean/none or a numeric constant.",
    )
    p.add_argument("--var-threshold", type=float, default=float("inf"))
    p.add_argument("--invalid-value", type=float, default=-1.0)
    p.add_argument("--stride", type=int, default=1, help="Stride for pixel grid (>=1).")

    # Local GP MLE parameters (used when --prediction-model local_gp_mle).
    p.add_argument("--gaussian-std", type=float, default=25.0, help="GP amplitude parameter a.")
    p.add_argument("--sigma-mismatch", type=float, default=0.5, help="Observation noise std (added to diagonal).")
    p.add_argument("--length-scale-min", type=float, default=0.1, help="Lower bound for length scale optimization.")
    p.add_argument("--length-scale-max", type=float, default=10.0, help="Upper bound for length scale optimization.")
    p.add_argument("--gp-opt-tol", type=float, default=1e-3, help="Tolerance for bounded length-scale optimizer.")
    p.add_argument("--gp-opt-max-iter", type=int, default=64, help="Max iterations for bounded length-scale optimizer.")

    # Cluster sharding / backend.
    p.add_argument("--num-shards", type=int, default=1, help="Total number of shards/jobs.")
    p.add_argument("--shard-rank", type=int, default=0, help="This job's shard rank in [0, num_shards).")
    p.add_argument("--backend", choices=["numpy", "torch"], default="numpy", help="Compute backend.")
    p.add_argument("--device", type=str, default="cpu", help="Device for torch backend (e.g. cuda, cuda:0).")

    # Parallelization within a shard.
    default_workers = max(1, (os.cpu_count() or 8) - 1)
    p.add_argument("--num-workers", type=int, default=default_workers, help="Worker threads per shard (0 = single-thread).")

    # Output.
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output depth .npy path (variance/valid are saved alongside as *_variance.npy and *_valid.npy). "
            "For sharded runs (--num-shards > 1), this is a shard .npz path."
        ),
    )
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")

    # Merge mode.
    p.add_argument("--merge-shards", action="store_true", help="Merge shard .npz files into one dense output.")
    p.add_argument("--shard-files", nargs="+", default=None, help="Shard .npz files to merge.")
    p.add_argument("--merged-output", type=str, default=None, help="Output path for merged dense .npz.")

    args = p.parse_args(list(argv) if argv is not None else None)
    show_progress = not bool(args.no_progress)

    # Normalize/validate prediction model.
    args.prediction_model = str(args.prediction_model).strip()
    if args.prediction_model not in {"local_gp_mle", "kernel"}:
        raise SystemExit("--prediction-model must be one of: local_gp_mle, kernel.")

    if args.backend == "torch":
        dev = str(args.device)
        if dev == "cuda":
            local_rank = os.environ.get("LOCAL_RANK") or os.environ.get("SLURM_LOCALID")
            if local_rank is not None:
                args.device = f"cuda:{int(local_rank)}"

    if args.merge_shards:
        if not args.shard_files or not args.merged_output:
            raise SystemExit("--merge-shards requires --shard-files ... and --merged-output.")
        return merge_shards(args.shard_files, output=args.merged_output, invalid_value=args.invalid_value, var_threshold=args.var_threshold)

    if not args.points_bin or not args.calib_txt:
        raise SystemExit("Both --points-bin and --calib-txt are required.")

    points_path = Path(args.points_bin)
    calib_path = Path(args.calib_txt)

    # points_path and calib_path are required above.

    if args.output is None:
        stem = points_path.stem
        shard_suffix = f".shard{int(args.shard_rank):03d}of{int(args.num_shards):03d}" if int(args.num_shards) > 1 else ""
        if int(args.num_shards) > 1:
            out_ext = f"{shard_suffix}.npz"
        else:
            out_ext = "_depth.npy"
        if args.prediction_model == "local_gp_mle":
            out_name = (
                f"depth_pred_{stem}_model-local_gp_mle_mode-{args.depth_mode}"
                f"_r{args.locality_radius_deg:g}_a{args.gaussian_std:g}_sig{args.sigma_mismatch:g}"
                f"_stride{args.stride}{out_ext}"
            )
        else:
            out_name = (
                f"depth_pred_{stem}_model-kernel_mode-{args.depth_mode}"
                f"_r{args.locality_radius_deg:g}_ls{args.kernel_length_scale_deg:g}"
                f"_stride{args.stride}{out_ext}"
            )
        out_path = points_path.parent / out_name
    else:
        out_path = Path(args.output)
        if int(args.num_shards) > 1:
            # Sharded outputs stay as .npz with shard suffix.
            if out_path.suffix.lower() != ".npz":
                out_path = out_path.with_suffix(".npz")
            shard_suffix = f".shard{int(args.shard_rank):03d}of{int(args.num_shards):03d}"
            if shard_suffix not in out_path.name:
                out_path = out_path.with_name(out_path.stem + shard_suffix + out_path.suffix)
        else:
            # Non-sharded dense outputs are .npy (depth), with sidecars.
            if out_path.suffix.lower() != ".npy":
                out_path = out_path.with_suffix(".npy")

    pts = load_sparse_points_bin(points_path, points_format=args.points_format)
    Tr = load_tr_velo_to_cam(calib_path)

    xyz = pts[:, 0:3]
    xyz_cam = transform_points_to_cam(xyz, Tr)
    pts_az, pts_el, pts_depth = angles_and_depth_from_cam_xyz(xyz_cam, depth_mode=args.depth_mode)
    shift_depth_value = _compute_shift_depth_value(pts_depth, args.shift_depth_value)

    # Intrinsics: optionally read from calib.
    intr_keys = tuple(k.strip() for k in str(args.intrinsics_keys).split(",") if k.strip())
    if args.intrinsics_source in {"auto", "calib"}:
        try:
            K = load_intrinsics_from_calib(calib_path, keys=intr_keys)
        except Exception:
            if args.intrinsics_source == "calib":
                raise
            if args.fx is None or args.fy is None or args.cx is None or args.cy is None:
                raise SystemExit(
                    "Intrinsics were not found in calib. Provide --fx/--fy/--cx/--cy or use --intrinsics-source calib with a valid calib file."
                )
            K = Intrinsics(fx=float(args.fx), fy=float(args.fy), cx=float(args.cx), cy=float(args.cy))
    else:
        if args.fx is None or args.fy is None or args.cx is None or args.cy is None:
            raise SystemExit("--intrinsics-source args requires --fx --fy --cx --cy.")
        K = Intrinsics(fx=float(args.fx), fy=float(args.fy), cx=float(args.cx), cy=float(args.cy))

    # Target image size.
    h, w = int(args.image_size[0]), int(args.image_size[1])
    if h <= 0 or w <= 0:
        raise SystemExit("--image-size must be positive: H W")
    target_az, target_el, hw_eff = pixel_angles_from_intrinsics((h, w), K, stride=args.stride)

    cpu_count = os.cpu_count() or 8
    nw = int(args.num_workers)
    if nw <= 0:
        print(f"[parallel] CPUs available: {cpu_count} | workers: 0 (single-threaded)")
    else:
        print(f"[parallel] CPUs available: {cpu_count} | workers: {nw} (parallel)")

    if args.prediction_model == "local_gp_mle":
        if args.backend == "torch":
            # Local GP MLE mode is implemented with NumPy linear algebra.
            args.backend = "numpy"
            args.device = "cpu"
        indices, mean, var, num_targets = predict_depth_local_gp_mle_binned_sparse(
            points_az_deg=pts_az,
            points_el_deg=pts_el,
            points_depth=pts_depth,
            target_az_deg=target_az,
            target_el_deg=target_el,
            locality_radius_deg=args.locality_radius_deg,
            gaussian_std=args.gaussian_std,
            sigma_mismatch=args.sigma_mismatch,
            length_scale_min=args.length_scale_min,
            length_scale_max=args.length_scale_max,
            opt_tol=args.gp_opt_tol,
            opt_max_iter=args.gp_opt_max_iter,
            shift_depth_value=shift_depth_value,
            bin_size_deg=args.bin_size_deg,
            num_workers=args.num_workers,
            num_shards=int(args.num_shards),
            shard_rank=int(args.shard_rank),
            show_progress=show_progress,
        )
    else:
        indices, mean, var, num_targets = predict_depth_kernel_circle_binned_sparse(
            points_az_deg=pts_az,
            points_el_deg=pts_el,
            points_depth=pts_depth,
            target_az_deg=target_az,
            target_el_deg=target_el,
            locality_radius_deg=args.locality_radius_deg,
            kernel_length_scale_deg=args.kernel_length_scale_deg,
            shift_depth_value=shift_depth_value,
            bin_size_deg=args.bin_size_deg,
            num_workers=args.num_workers,
            backend=args.backend,
            device=args.device,
            num_shards=int(args.num_shards),
            shard_rank=int(args.shard_rank),
            show_progress=show_progress,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if int(args.num_shards) > 1:
        np.savez_compressed(
            out_path,
            indices=indices.astype(np.int64, copy=False),
            mean=mean.astype(np.float32, copy=False),
            var=var.astype(np.float32, copy=False),
            hw_eff=np.array(hw_eff, dtype=np.int32),
            intrinsics=np.array([K.fx, K.fy, K.cx, K.cy], dtype=np.float32),
            Tr_sensor_to_cam=Tr.astype(np.float64),
            prediction_model=np.array(str(args.prediction_model)),
            params=np.array([args.locality_radius_deg, shift_depth_value, float(args.stride)], dtype=np.float32),
            kernel_params=np.array([args.kernel_length_scale_deg], dtype=np.float32),
            gp_params=np.array(
                [
                    args.gaussian_std,
                    args.sigma_mismatch,
                    args.length_scale_min,
                    args.length_scale_max,
                    args.gp_opt_tol,
                    float(args.gp_opt_max_iter),
                ],
                dtype=np.float32,
            ),
            shard=np.array([int(args.shard_rank), int(args.num_shards)], dtype=np.int32),
            backend=np.array(str(args.backend)),
            device=np.array(str(args.device)),
            source_points=np.array(str(points_path)),
            source_calib=np.array(str(calib_path)),
            points_format=np.array(str(args.points_format)),
        )
        print(f"Saved shard: {out_path}")
        print(f"Shard predictions: {indices.size} pixels (out of {num_targets})")
        return 0

    pred_mean, pred_var, valid = _dense_from_sparse(
        num_targets=num_targets,
        indices=indices,
        mean=mean,
        var=var,
        invalid_value=args.invalid_value,
        var_threshold=args.var_threshold,
    )

    depth_map = pred_mean.reshape(hw_eff)
    var_map = pred_var.reshape(hw_eff)
    valid_map = valid.reshape(hw_eff)

    depth_path, var_path, valid_path = _sidecar_output_paths(out_path)
    np.save(depth_path, depth_map.astype(np.float32, copy=False))
    np.save(var_path, var_map.astype(np.float32, copy=False))
    # Save valid as uint8 for compactness/compatibility (0/1).
    np.save(valid_path, valid_map.astype(np.uint8, copy=False))

    print(f"Saved depth: {depth_path}")
    print(f"Saved variance: {var_path}")
    print(f"Saved valid: {valid_path}")
    print(f"Output shape: {depth_map.shape} (stride={args.stride}, original image={h}x{w})")
    print(f"Valid pixels: {int(valid_map.sum())} / {valid_map.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())