"""
Video Quality Metrics Module
Computes PSNR, SSIM, LPIPS, tLPIPS, FID, FVD, KVD, IS metrics for video evaluation.
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ms_ssim
import torch
import lpips
import json
import os
from typing import List, Tuple, Dict, Optional
import cv2
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

# Model caches
_LPIPS_MODEL_CACHE: Dict[str, torch.nn.Module] = {}


def calculate_psnr(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Calculate PSNR between two frames (HxWx3 uint8 RGB)."""
    diff = frame1.astype(np.float32) - frame2.astype(np.float32)
    mse = np.mean(diff * diff, dtype=np.float64)
    if mse <= 1e-12:
        return 100.0
    return 20.0 * np.log10(255.0) - 10.0 * np.log10(mse)


def rgb_to_y_bt601_full(frame_rgb: np.ndarray) -> np.ndarray:
    """RGB (uint8, gamma-corrected) → BT.601 luma Y′ in [0,255] (full range)."""
    f = frame_rgb.astype(np.float32)
    y = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    return np.clip(y, 0.0, 255.0).astype(np.float32)


def calculate_psnr_y_bt601_full(a_rgb: np.ndarray, b_rgb: np.ndarray) -> float:
    """Calculate PSNR on BT.601 luma channel."""
    y1 = rgb_to_y_bt601_full(a_rgb)
    y2 = rgb_to_y_bt601_full(b_rgb)
    mse = np.mean((y1 - y2) ** 2, dtype=np.float64)
    if mse <= 1e-12:
        return 100.0
    return 20.0 * np.log10(255.0) - 10.0 * np.log10(mse)


def calculate_ssim(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Calculate SSIM between two frames (HxWx3 uint8 RGB)."""
    return float(
        ssim(
            frame1, frame2,
            data_range=255,
            channel_axis=-1,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            win_size=11
        )
    )


def calculate_ms_ssim(frame1: np.ndarray, frame2: np.ndarray, device: torch.device) -> float:
    """Calculate MS-SSIM between two frames (HxWx3 uint8 RGB)."""
    x = torch.from_numpy(frame1).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    y = torch.from_numpy(frame2).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    with torch.no_grad():
        score = ms_ssim(x, y, data_range=1.0, size_average=True)
    return float(score.item())


def get_lpips_model(device: torch.device, net: str = 'alex') -> torch.nn.Module:
    """Get or create LPIPS model (cached)."""
    cache_key = f"{device}|{net}"
    if cache_key not in _LPIPS_MODEL_CACHE:
        model = lpips.LPIPS(net=net).to(device).eval()
        _LPIPS_MODEL_CACHE[cache_key] = model
    return _LPIPS_MODEL_CACHE[cache_key]


def calculate_lpips(frame1: np.ndarray, frame2: np.ndarray, lpips_model: torch.nn.Module, device: torch.device) -> float:
    """Computes the LPIPS between two frames (HxWx3 uint8 RGB)."""
    def np_to_lpips_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
        t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)  # 1x3xHxW
        t = t / 255.0 * 2.0 - 1.0  # [0,1] -> [-1,1]
        return t.to(device)
    
    t1 = np_to_lpips_tensor(frame1, device)
    t2 = np_to_lpips_tensor(frame2, device)
    with torch.no_grad():
        score = lpips_model(t1, t2)
    return float(score.item())


def calculate_tlpips(
    real_frames: List[np.ndarray],
    fake_frames: List[np.ndarray],
    lpips_model: torch.nn.Module,
    device: torch.device,
) -> Tuple[List[float], float]:
    """
    Temporal LPIPS (tLPIPS): average over t of | LPIPS(g_{t-1}, g_t) - LPIPS(xhat_{t-1}, xhat_t) |.
    Returns (per_pair_values_over_t, average). Lower is better.
    """
    assert len(real_frames) == len(fake_frames) and len(real_frames) >= 2, \
        "Need same-length real/fake and at least 2 frames"

    diffs: List[float] = []
    for t in range(1, len(real_frames)):
        lp_real = calculate_lpips(real_frames[t-1], real_frames[t], lpips_model, device)
        lp_fake = calculate_lpips(fake_frames[t-1], fake_frames[t], lpips_model, device)
        diffs.append(abs(lp_real - lp_fake))
    return diffs, float(np.mean(diffs))


def stack_frames_uint8(frames_np_list: List[np.ndarray]) -> torch.Tensor:
    """Stack list of HxWx3 uint8 RGB frames into a uint8 tensor (N,3,H,W)."""
    arr = np.stack(frames_np_list, axis=0)  # (N,H,W,3)
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # (N,3,H,W)
    return t


def calculate_fid(
    all_real: torch.Tensor,
    all_fake: torch.Tensor,
    device: torch.device,
    normalize: bool = False,
    batch_size: int = 64,
) -> float:
    """Compute Frechet Inception Distance for two image sets.
    all_real/all_fake: uint8 tensors (N,3,H,W) in [0,255], or float [0,1] if normalize=True.
    """
    fid = FrechetInceptionDistance(feature=2048, normalize=normalize).to(device).eval()
    with torch.no_grad():
        for start in range(0, all_real.shape[0], batch_size):
            fid.update(all_real[start:start+batch_size].to(device), real=True)
        for start in range(0, all_fake.shape[0], batch_size):
            fid.update(all_fake[start:start+batch_size].to(device), real=False)
        score = fid.compute()
    return float(score.item())


def calculate_fid_from_frames(
    real_frames: List[np.ndarray],
    fake_frames: List[np.ndarray],
    device: torch.device,
    sample_every: int = 1,
    max_frames: Optional[int] = None,
    batch_size: int = 64,
) -> float:
    """Convenience wrapper to compute FID from lists of HxWx3 uint8 RGB frames."""
    assert len(real_frames) > 0 and len(fake_frames) > 0, "Empty frame lists"
    idx = np.arange(0, min(len(real_frames), len(fake_frames)), sample_every)
    if max_frames is not None:
        idx = idx[:max_frames]
    real = [real_frames[i] for i in idx]
    fake = [fake_frames[i] for i in idx]
    real_tensor = stack_frames_uint8(real)
    fake_tensor = stack_frames_uint8(fake)
    return calculate_fid(real_tensor, fake_tensor, device=device, normalize=False, batch_size=batch_size)


def _resize_frames(frames: List[np.ndarray], size: Optional[Tuple[int, int]]) -> List[np.ndarray]:
    """Resize frames to (H, W) with bilinear interpolation. If size=None, no resize."""
    if size is None:
        return frames
    H, W = size
    return [cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR) for f in frames]


def _make_clips(
    frames: List[np.ndarray],
    clip_len: int = 16,
    stride: Optional[int] = None,
    resize_hw: Optional[Tuple[int, int]] = (224, 224),
    pad_tail: bool = True
) -> torch.Tensor:
    """
    Turn a list of HxWx3 uint8 RGB frames into (N, 3, T, H, W) uint8 clips.
    """
    if len(frames) == 0:
        raise ValueError("No frames provided")
    frames = _resize_frames(frames, resize_hw)
    if stride is None:
        stride = clip_len
    N = len(frames)
    clips: List[np.ndarray] = []
    for start in range(0, max(1, N - clip_len + 1), stride):
        end = start + clip_len
        if end <= N:
            seq = frames[start:end]
        else:
            if not pad_tail:
                break
            seq = frames[start:N] + [frames[-1]] * (end - N)
        arr = np.stack(seq, axis=0)  # (T, H, W, 3) uint8
        clips.append(arr)
    if not clips:
        seq = frames + [frames[-1]] * (clip_len - len(frames))
        clips = [np.stack(seq, axis=0)]
    clips_np = np.stack(clips, axis=0)  # (N, T, H, W, 3)
    clips_t = torch.from_numpy(clips_np).permute(0, 4, 1, 2, 3).contiguous()  # (N, 3, T, H, W)
    return clips_t


def videos_to_fvd_clips(
    real_frames: List[np.ndarray],
    fake_frames: List[np.ndarray],
    clip_len: int = 16,
    stride: Optional[int] = None,
    resize_hw: Optional[Tuple[int, int]] = (224, 224),
    pad_tail: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build aligned clip tensors (N,3,T,H,W) uint8 for real & fake, trimmed to the same number of clips."""
    real = _make_clips(real_frames, clip_len=clip_len, stride=stride, resize_hw=resize_hw, pad_tail=pad_tail)
    fake = _make_clips(fake_frames, clip_len=clip_len, stride=stride, resize_hw=resize_hw, pad_tail=pad_tail)
    n = min(real.shape[0], fake.shape[0])
    return real[:n], fake[:n]


@torch.no_grad()
def calculate_fvd_from_frames(
    real_frames: List[np.ndarray],
    fake_frames: List[np.ndarray],
    device: torch.device,
    clip_len: int = 16,
    stride: Optional[int] = None,
    resize_hw: Optional[Tuple[int, int]] = (224, 224),
    num_samples: Optional[int] = None,
    backbone: str = "videomae",
) -> float:
    """
    Compute FVD via cd-fvd on NumPy uint8 clips shaped (B, T, H, W, C).
    Returns a single float (lower is better).
    """
    real_u8, fake_u8 = videos_to_fvd_clips(
        real_frames, fake_frames, clip_len=clip_len, stride=stride, resize_hw=resize_hw, pad_tail=True
    )
    real_np = real_u8.permute(0, 2, 3, 4, 1).contiguous().cpu().numpy()  # (N, T, H, W, C), uint8
    fake_np = fake_u8.permute(0, 2, 3, 4, 1).contiguous().cpu().numpy()

    n = real_np.shape[0]
    if num_samples is not None and n > num_samples:
        idx = np.linspace(0, n - 1, num=num_samples).round().astype(np.int64)
        real_np = real_np[idx]
        fake_np = fake_np[idx]

    model_name = backbone.lower()
    if model_name.startswith("i3d"):
        model_name = "i3d"
    elif model_name.startswith("video"):
        model_name = "videomae"
    elif model_name not in {"i3d", "videomae"}:
        model_name = "videomae"

    try:
        from cdfvd import fvd
        evaluator = fvd.cdfvd(model_name, device=str(device), half_precision=False)
        fvd_score = evaluator.compute_fvd(real_np, fake_np)
        return float(fvd_score)
    except ImportError:
        print("Warning: cdfvd not available, skipping FVD")
        return float('nan')


def calculate_kid(
    all_real: torch.Tensor,
    all_fake: torch.Tensor,
    device: torch.device,
    subsets: int = 50,
    seed: int | None = None
) -> Tuple[float, float]:
    """
    Computes Kernel Inception Distance (KID) between two image sets.
    all_real/all_fake: uint8 tensors (N,3,H,W) in [0,255].
    Returns: (kid_mean, kid_std)
    """
    if seed is not None:
        try:
            np.random.seed(seed)
        except Exception:
            pass
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    num_samples = int(min(all_real.shape[0], all_fake.shape[0], 1000))
    kid = KernelInceptionDistance(
        subsets=subsets,
        subset_size=max(num_samples, 1),
        normalize=False
    ).to(device)
    with torch.no_grad():
        kid.update(all_real.to(device), real=True)
        kid.update(all_fake.to(device), real=False)
        kid_mean, kid_std = kid.compute()
    return float(kid_mean.item()), float(kid_std.item())


def calculate_kid_from_frames(
    real_frames: List[np.ndarray],
    fake_frames: List[np.ndarray],
    device: torch.device,
    subsets: int = 50,
    seed: int | None = None
) -> Tuple[float, float]:
    """Convenience wrapper to compute KID from lists of HxWx3 uint8 RGB frames."""
    assert len(real_frames) > 0 and len(fake_frames) > 0, "Empty frame lists"
    n = min(len(real_frames), len(fake_frames))
    real_stack = np.stack(real_frames[:n], axis=0)  # N,H,W,3
    fake_stack = np.stack(fake_frames[:n], axis=0)
    real_tensor = torch.from_numpy(real_stack).permute(0, 3, 1, 2).contiguous()  # N,3,H,W
    fake_tensor = torch.from_numpy(fake_stack).permute(0, 3, 1, 2).contiguous()
    return calculate_kid(real_tensor, fake_tensor, device=device, subsets=subsets, seed=seed)


def calculate_is_from_frames(
    fake_frames: List[np.ndarray],
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[float, float]:
    """
    Compute Inception Score (IS) from generated frames.
    Returns: (is_mean, is_std)
    """
    fake_tensor = stack_frames_uint8(fake_frames)
    is_metric = InceptionScore(normalize=False).to(device).eval()
    with torch.no_grad():
        for start in range(0, fake_tensor.shape[0], batch_size):
            is_metric.update(fake_tensor[start:start+batch_size].to(device))
        is_mean, is_std = is_metric.compute()
    return float(is_mean.item()), float(is_std.item())


def compute_all_video_metrics(
    original_frames: List[np.ndarray],
    generated_frames: List[np.ndarray],
    device: torch.device,
    output_dir: str,
    name_suffix: Optional[str] = None,
) -> Dict:
    """
    Computes all metrics for a video and returns results as a dictionary.
    
    Args:
        original_frames: List of HxWx3 uint8 RGB frames (ground truth)
        generated_frames: List of HxWx3 uint8 RGB frames (generated)
        device: torch device
        output_dir: Directory to save results
        name_suffix: Optional suffix for output files
    
    Returns:
        Dictionary with all computed metrics
    """
    assert len(original_frames) == len(generated_frames), "Videos must have same number of frames"
    
    print("Initializing metric models...")
    lpips_model = get_lpips_model(device, net='alex')
    
    metrics = {
        'psnr': {'per_frame': [], 'average': 0.0},
        'psnr_y': {'per_frame': [], 'average': 0.0},
        'ssim': {'per_frame': [], 'average': 0.0},
        'ms_ssim': {'per_frame': [], 'average': 0.0},
        'lpips': {'per_frame': [], 'average': 0.0},
        'tlpips': {'per_frame': [], 'average': 0.0},
    }
    
    print("Computing per-frame metrics...")
    for k in range(len(original_frames)):
        orig = original_frames[k]
        gen = generated_frames[k]
        
        metrics['psnr']['per_frame'].append(calculate_psnr(orig, gen))
        metrics['psnr_y']['per_frame'].append(calculate_psnr_y_bt601_full(orig, gen))
        metrics['ssim']['per_frame'].append(calculate_ssim(orig, gen))
        metrics['ms_ssim']['per_frame'].append(calculate_ms_ssim(orig, gen, device))
        metrics['lpips']['per_frame'].append(calculate_lpips(orig, gen, lpips_model, device))
    
    # Compute temporal LPIPS
    print("Computing temporal LPIPS...")
    tlpips_pf, tlpips_avg = calculate_tlpips(original_frames, generated_frames, lpips_model, device)
    metrics['tlpips']['per_frame'] = tlpips_pf
    metrics['tlpips']['average'] = tlpips_avg

    # Calculate averages for per-frame metrics
    for metric in metrics:
        if 'per_frame' in metrics[metric] and isinstance(metrics[metric]['per_frame'], list):
            metrics[metric]['average'] = float(np.mean(metrics[metric]['per_frame']))

    # Compute dataset-level metrics
    print("Computing FID...")
    try:
        fid_value = calculate_fid_from_frames(original_frames, generated_frames, device, batch_size=64)
        metrics['fid'] = {'per_frame': None, 'average': fid_value}
    except Exception as e:
        print(f"Warning: FID computation failed: {e}")
        metrics['fid'] = {'per_frame': None, 'average': float('nan')}

    print("Computing FVD...")
    try:
        fvd_value = calculate_fvd_from_frames(
            original_frames, generated_frames, device=device,
            clip_len=16, stride=None, resize_hw=(224, 224),
            num_samples=None, backbone="i3d"
        )
        metrics['fvd'] = {'per_frame': None, 'average': fvd_value}
    except Exception as e:
        print(f"Warning: FVD computation failed: {e}")
        metrics['fvd'] = {'per_frame': None, 'average': float('nan')}

    print("Computing KVD (KID)...")
    try:
        kid_mean, kid_std = calculate_kid_from_frames(original_frames, generated_frames, device, subsets=50, seed=42)
        metrics['kvd'] = {'per_frame': None, 'average': kid_mean, 'std': kid_std}
    except Exception as e:
        print(f"Warning: KVD (KID) computation failed: {e}")
        metrics['kvd'] = {'per_frame': None, 'average': float('nan'), 'std': float('nan')}

    print("Computing IS...")
    try:
        is_mean, is_std = calculate_is_from_frames(generated_frames, device, batch_size=64)
        metrics['is'] = {'per_frame': None, 'average': is_mean, 'std': is_std}
    except Exception as e:
        print(f"Warning: IS computation failed: {e}")
        metrics['is'] = {'per_frame': None, 'average': float('nan'), 'std': float('nan')}

    # Save JSON results
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{name_suffix}" if name_suffix else ""
    json_path = os.path.join(output_dir, f"metrics{suffix}.json")
    
    to_save = {}
    for k, v in metrics.items():
        if k == 'kvd' or k == 'is':
            to_save[k] = {'mean': v['average'], 'std': v.get('std', 0.0)}
        else:
            to_save[k] = v['average']
    
    with open(json_path, 'w') as f:
        json.dump(to_save, f, indent=4)
    
    print(f"\nMetrics summary (averages):")
    print(json.dumps(to_save, indent=4))
    print(f"\nMetrics saved to: {json_path}")
    
    return metrics

