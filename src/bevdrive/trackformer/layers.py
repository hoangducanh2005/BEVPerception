"""TrackFormer decoder building blocks, weight-compatible with UniAD stage-1.

Ported from UniAD (``modules/decoder.py``: ``DetectionTransformerDecoder``,
``CustomMSDeformableAttention``; mmcv ``DetrTransformerDecoderLayer`` with
``operation_order=(self_attn, norm, cross_attn, norm, ffn, norm)``) as plain
PyTorch, so it runs in the mmcv-2 environment of the BEVFusion detector.
Parameter names mirror the UniAD state dict:

    layers.{i}.attentions.0.attn.*            nn.MultiheadAttention (query self-attn)
    layers.{i}.attentions.1.sampling_offsets  deformable cross-attn on the BEV
    layers.{i}.attentions.1.attention_weights
    layers.{i}.attentions.1.value_proj / output_proj
    layers.{i}.ffns.0.layers.0.0 / layers.1   FFN
    layers.{i}.norms.0..2
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def deformable_attn_single_level(value: torch.Tensor, shape_hw: tuple[int, int], sampling_locations: torch.Tensor, attention_weights: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch single-level MS-deformable attention (== mmcv multi_scale_deformable_attn_pytorch).

    value [bs, H*W, heads, dh]; sampling_locations [bs, nq, heads, P, 2] in [0,1] (x->W, y->H);
    attention_weights [bs, nq, heads, P]. Returns [bs, nq, heads*dh].
    """

    bs, _, heads, dh = value.shape
    _, nq, _, points, _ = sampling_locations.shape
    h, w = shape_hw
    value_l = value.permute(0, 2, 3, 1).reshape(bs * heads, dh, h, w)
    grid = (2 * sampling_locations - 1).permute(0, 2, 1, 3, 4).reshape(bs * heads, nq, points, 2)
    sampled = F.grid_sample(value_l, grid, mode="bilinear", padding_mode="zeros", align_corners=False)  # [bs*heads, dh, nq, P]
    weights = attention_weights.permute(0, 2, 1, 3).reshape(bs * heads, 1, nq, points)
    out = (sampled * weights).sum(-1).view(bs, heads * dh, nq)
    return out.transpose(1, 2).contiguous()


class CustomMSDeformableAttention(nn.Module):
    def __init__(self, embed_dims: int = 256, num_heads: int = 8, num_levels: int = 1, num_points: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        assert num_levels == 1, "BEV decoder uses a single level"
        self.embed_dims, self.num_heads, self.num_levels, self.num_points = embed_dims, num_heads, num_levels, num_points
        self.dropout = nn.Dropout(dropout)
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid = (grid / grid.abs().max(-1, keepdim=True)[0]).view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid.view(-1)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, query: torch.Tensor, value: torch.Tensor, query_pos: torch.Tensor, reference_points: torch.Tensor, shape_hw: tuple[int, int]) -> torch.Tensor:
        """query/query_pos [nq, bs, C]; value [H*W, bs, C]; reference_points [bs, nq, 2] in [0,1]."""

        identity = query
        q = (query + query_pos).permute(1, 0, 2)
        v = value.permute(1, 0, 2)
        bs, nq, _ = q.shape
        v = self.value_proj(v).view(bs, v.shape[1], self.num_heads, -1)
        offsets = self.sampling_offsets(q).view(bs, nq, self.num_heads, self.num_points, 2)
        weights = self.attention_weights(q).view(bs, nq, self.num_heads, self.num_points).softmax(-1)
        h, w = shape_hw
        normalizer = torch.tensor([w, h], dtype=q.dtype, device=q.device)
        locations = reference_points[:, :, None, None, :] + offsets / normalizer
        out = deformable_attn_single_level(v, shape_hw, locations, weights)
        out = self.output_proj(out).permute(1, 0, 2)
        return self.dropout(out) + identity


class _SelfAttention(nn.Module):
    """mmcv ``MultiheadAttention`` wrapper semantics: q = k = x + pos, v = x, residual."""

    def __init__(self, embed_dims: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout)
        self.proj_drop = nn.Dropout(dropout)  # mmcv dropout_layer(drop_prob=dropout)

    def forward(self, query: torch.Tensor, query_pos: torch.Tensor) -> torch.Tensor:
        qk = query + query_pos
        return query + self.proj_drop(self.attn(qk, qk, query)[0])


class _FFN(nn.Module):
    def __init__(self, embed_dims: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Sequential(nn.Linear(embed_dims, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout)), nn.Linear(hidden, embed_dims), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class TrackDecoderLayer(nn.Module):
    def __init__(self, embed_dims: int = 256, num_heads: int = 8, ffn_dims: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.attentions = nn.ModuleList([_SelfAttention(embed_dims, num_heads, dropout), CustomMSDeformableAttention(embed_dims, num_heads, 1, 4, dropout)])
        self.ffns = nn.ModuleList([_FFN(embed_dims, ffn_dims, dropout)])
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(3)])

    def forward(self, query: torch.Tensor, query_pos: torch.Tensor, value: torch.Tensor, reference_points: torch.Tensor, shape_hw: tuple[int, int]) -> torch.Tensor:
        query = self.norms[0](self.attentions[0](query, query_pos))
        query = self.norms[1](self.attentions[1](query, value, query_pos, reference_points, shape_hw))
        return self.norms[2](self.ffns[0](query))


class TrackDecoder(nn.Module):
    """DETR3D-style decoder with iterative reference refinement (``with_box_refine=True``)."""

    def __init__(self, num_layers: int = 6, embed_dims: int = 256) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TrackDecoderLayer(embed_dims) for _ in range(num_layers)])

    def forward(self, query, query_pos, value, reference_points, shape_hw, reg_branches):
        """query/query_pos [nq, bs, C]; reference_points [bs, nq, 3] (sigmoid space, normalized pc_range)."""

        states, refs = [], []
        output = query
        for lid, layer in enumerate(self.layers):
            output = layer(output, query_pos, value, reference_points[..., :2], shape_hw)
            tmp = reg_branches[lid](output.permute(1, 0, 2))
            new_ref = torch.zeros_like(reference_points)
            new_ref[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
            new_ref[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
            reference_points = new_ref.sigmoid().detach()
            states.append(output)
            refs.append(reference_points)
        return torch.stack(states), torch.stack(refs)


def make_cls_branch(embed_dims: int = 256, num_classes: int = 10, num_reg_fcs: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    for _ in range(num_reg_fcs):
        layers += [nn.Linear(embed_dims, embed_dims), nn.LayerNorm(embed_dims), nn.ReLU(inplace=True)]
    layers.append(nn.Linear(embed_dims, num_classes))
    return nn.Sequential(*layers)


def make_reg_branch(embed_dims: int = 256, code_size: int = 10, num_reg_fcs: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    for _ in range(num_reg_fcs):
        layers += [nn.Linear(embed_dims, embed_dims), nn.ReLU()]
    layers.append(nn.Linear(embed_dims, code_size))
    return nn.Sequential(*layers)
