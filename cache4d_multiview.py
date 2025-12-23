import torch
from einops import rearrange

from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_Base

class Cache4D_BufferSelector(Cache3D_Base):
    def __init__(
        self,
        frame_buffer_max: int = 1,
        mask_full_threshold: float = 0.9,
        mask_for_max_buffer_model: bool = True,
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.frame_buffer_max = max(int(frame_buffer_max), 1)
        self.mask_for_max_buffer_model = bool(mask_for_max_buffer_model)
        self.mask_full_threshold = float(mask_full_threshold)
    
    def update_cache(self, *args, **kwargs):
        raise NotImplementedError("Cache4D_BufferSelector does not support update_cache")

    def render_cache(
        self,
        target_w2cs: torch.Tensor,
        target_intrinsics: torch.Tensor,
        render_depth: bool = False,
        start_frame_idx: int = 0,
    ):
        output_device = target_w2cs.device
        target_w2cs = target_w2cs.to(self.weight_dtype).to(self.device)
        target_intrinsics = target_intrinsics.to(self.weight_dtype).to(self.device)

        pixels_all, masks_all = super().render_cache(
            target_w2cs, target_intrinsics, render_depth, start_frame_idx
        )
        print(f"pixels_all.shape: {pixels_all.shape}")
        print(f"masks_all.shape: {masks_all.shape}")

        B, F_t, N = masks_all.shape[0], masks_all.shape[1], masks_all.shape[2]
        K = min(self.frame_buffer_max, N)

        if N <= K:
            # Nothing to select; already <= K videos
            return pixels_all.to(output_device), masks_all.to(output_device)
        else:
            overlap_scores = masks_all.sum(dim=(1, 3, 4, 5))
            topk_indices = overlap_scores.topk(k=K, dim=1, largest=True, sorted=True).indices
            selected_pixels, selected_masks = [], []
            for b in range(B):
                idx_b = topk_indices[b]
                selected_pixels.append(pixels_all[b : b + 1, :, idx_b])
                selected_masks.append(masks_all[b : b + 1, :, idx_b])
            pixels_sel = torch.cat(selected_pixels, dim=0)
            masks_sel = torch.cat(selected_masks, dim=0)
        
        if self.mask_for_max_buffer_model and not render_depth:
            _masks = masks_sel.mean(dim=[3, 4, 5])  # [B, F_t, K]
            Bm, Fm, Km = _masks.shape
            _flat = rearrange(_masks, "b f k -> (b f) k")

            result_mask = torch.zeros_like(_flat)
            near_full = _flat >= self.mask_full_threshold
            has_near_full = near_full.any(dim=1)

            first_indices = near_full.float().argmax(dim=1)
            rows_with = torch.arange(_flat.size(0), device=_flat.device)[has_near_full]
            first_keep = first_indices[has_near_full]
            result_mask[rows_with, first_keep] = 1

            rows_without = torch.arange(_flat.size(0), device=_flat.device)[~has_near_full]
            if rows_without.numel() > 0:
                result_mask[rows_without] = 1

            result_mask = rearrange(result_mask, "(b f) k -> b f k", b=Bm, f=Fm)
            gate = result_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, F, K, 1, 1, 1]
            if pixels_sel.dim() == 6:
                # RGB path: [B, F, K, C, H, W]
                pixels_sel = (pixels_sel + 1) * gate - 1
            else:
                # Depth path: [B, F, K, H, W] (no C)
                # For depth, we only gate masks (keep pixels_sel unchanged)
                pass
            masks_sel = masks_sel * gate

        return pixels_sel.to(output_device), masks_sel.to(output_device)


        
