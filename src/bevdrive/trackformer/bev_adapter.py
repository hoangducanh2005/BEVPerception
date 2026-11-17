"""BEVFusion fused BEV -> UniAD TrackFormer value map.

Interface gap that this module closes (see reports/phase_04_query_tracking.md):

=================  ===========================  ===========================
                   BEVFusion (pts_neck output)  UniAD TrackFormer value
=================  ===========================  ===========================
channels           512                          256
grid               180 x 180, 0.6 m             200 x 200, 0.512 m
extent             [-54, 54] m                  [-51.2, 51.2] m
tensor layout      [C, x, y]                    [C, y, x] (flattened H*W)
positional info    none (LiDAR/camera content)  learned row/col embedding
temporal context   none                         (BEVFormer TSA history)
=================  ===========================  ===========================

The current map and the motion-gated, ego-aligned history context (Phase 2)
are concatenated, projected with a small conv stack on the native grid and
then resampled *metrically* onto the UniAD grid (no learned warping).
"""

from __future__ import annotations

import torch
from torch import nn

from bevdrive.temporal.bev_alignment import BEVFUSION_GRID, UNIAD_GRID, BEVGrid, resample


class LearnedPositionalEncoding(nn.Module):
    """mmdet ``LearnedPositionalEncoding`` (row/col embeddings, concatenated)."""

    def __init__(self, num_feats: int = 128, rows: int = 200, cols: int = 200) -> None:
        super().__init__()
        self.row_embed = nn.Embedding(rows, num_feats)
        self.col_embed = nn.Embedding(cols, num_feats)
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, h: int, w: int) -> torch.Tensor:
        device = self.row_embed.weight.device
        x_embed = self.col_embed(torch.arange(w, device=device))
        y_embed = self.row_embed(torch.arange(h, device=device))
        pos = torch.cat([x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(1, w, 1)], dim=-1)
        return pos.permute(2, 0, 1).unsqueeze(0)


class BEVAdapter(nn.Module):
    def __init__(
        self,
        in_channels: int = 512,
        use_temporal: bool = True,
        embed_dims: int = 256,
        src_grid: BEVGrid = BEVFUSION_GRID,
        dst_grid: BEVGrid = UNIAD_GRID,
        add_positional: bool = True,
    ) -> None:
        super().__init__()
        self.use_temporal, self.src_grid, self.dst_grid, self.add_positional = use_temporal, src_grid, dst_grid, add_positional
        cin = in_channels * (2 if use_temporal else 1)
        self.proj = nn.Sequential(
            nn.GroupNorm(32, cin),
            nn.Conv2d(cin, embed_dims, 1, bias=False),
            nn.GroupNorm(32, embed_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims, embed_dims, 3, padding=1, bias=False),
            nn.GroupNorm(32, embed_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims, embed_dims, 1),
        )
        rows, cols = dst_grid.shape
        self.positional_encoding = LearnedPositionalEncoding(embed_dims // 2, rows, cols)

    def forward(self, bev: torch.Tensor, history: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """bev/history [1, 512, 180, 180] -> (value [H*W, 1, C], map [1, C, H, W]) on the UniAD grid."""

        if self.use_temporal:
            history = torch.zeros_like(bev) if history is None else history
            bev = torch.cat([bev, history], dim=1)
        x = self.proj(bev.float())
        x, _ = resample(x, self.src_grid, self.dst_grid)
        if self.add_positional:
            x = x + self.positional_encoding(*self.dst_grid.shape)
        return x.flatten(2).permute(2, 0, 1), x
