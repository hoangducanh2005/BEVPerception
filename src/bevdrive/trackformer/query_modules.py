"""Track-query temporal modules ported from UniAD ``track_head_plugin``.

* ``QueryInteractionModule`` (QIM, from MOTR): updates the query of every
  surviving track with its decoder output embedding (self-attn + FFNs);
  during training it randomly drops tracks and injects false-positive tracks.
* ``MemoryBank``: per-track FIFO of past embeddings + temporal attention
  (the learned track embedding history).
* ``RuntimeTrackerBase``: query lifecycle at inference - birth of new ids for
  confident detection queries, sleeping/termination after ``miss_tolerance``.

Parameter names match the UniAD stage-1 checkpoint (``query_interact.*``,
``memory_bank.*``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .instances import Instances


class MemoryBank(nn.Module):
    def __init__(self, dim: int = 256, memory_len: int = 4, save_thresh: float = 0.0, save_period: int = 3) -> None:
        super().__init__()
        self.save_thresh, self.save_period, self.max_his_length = save_thresh, save_period, memory_len
        self.save_proj = nn.Linear(dim, dim)
        self.temporal_attn = nn.MultiheadAttention(dim, 8, dropout=0)
        self.temporal_fc1 = nn.Linear(dim, dim)
        self.temporal_fc2 = nn.Linear(dim, dim)
        self.temporal_norm1 = nn.LayerNorm(dim)
        self.temporal_norm2 = nn.LayerNorm(dim)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def update(self, inst: Instances) -> None:
        embed = inst.output_embedding[:, None]
        scores, mask, save_period = inst.scores, inst.mem_padding_mask, inst.save_period
        if self.training:
            saved = scores > 0
        else:
            saved = (save_period == 0) & (scores > self.save_thresh)
            save_period[save_period > 0] -= 1
            save_period[saved] = self.save_period
        if saved.any():
            save_embed = self.save_proj(embed[saved])
            mask[saved] = torch.cat([mask[saved, 1:], torch.zeros((int(saved.sum()), 1), dtype=torch.bool, device=embed.device)], dim=1)
            bank = inst.mem_bank.clone()
            bank[saved] = torch.cat([inst.mem_bank[saved][:, 1:], save_embed], dim=1)
            inst.mem_bank = bank

    def temporal_attention(self, inst: Instances) -> Instances:
        if len(inst) == 0:
            return inst
        valid = inst.mem_padding_mask[:, -1] == 0
        if valid.any():
            embed = inst.output_embedding[valid]
            prev = inst.mem_bank[valid]
            attn = self.temporal_attn(embed[None], prev.transpose(0, 1), prev.transpose(0, 1), key_padding_mask=inst.mem_padding_mask[valid])[0][0]
            embed = self.temporal_norm1(embed + attn)
            embed = self.temporal_norm2(embed + self.temporal_fc2(F.relu(self.temporal_fc1(embed))))
            out = inst.output_embedding.clone()
            out[valid] = embed
            inst.output_embedding = out
        return inst

    def forward(self, inst: Instances) -> Instances:
        inst = self.temporal_attention(inst)
        self.update(inst)
        return inst


class QueryInteractionModule(nn.Module):
    def __init__(self, dim: int = 256, dropout: float = 0.0, update_query_pos: bool = True, fp_ratio: float = 0.3, random_drop: float = 0.1) -> None:
        super().__init__()
        self.update_query_pos, self.fp_ratio, self.random_drop = update_query_pos, fp_ratio, random_drop
        self.self_attn = nn.MultiheadAttention(dim, 8, dropout)
        self.linear1, self.linear2 = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        if update_query_pos:
            self.linear_pos1, self.linear_pos2 = nn.Linear(dim, dim), nn.Linear(dim, dim)
            self.dropout_pos1, self.dropout_pos2 = nn.Dropout(dropout), nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim)
        self.linear_feat1, self.linear_feat2 = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.dropout_feat1, self.dropout_feat2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _update_track_embedding(self, inst: Instances) -> Instances:
        if len(inst) == 0:
            return inst
        dim = inst.query.shape[1]
        out_embed = inst.output_embedding
        query_pos, query_feat = inst.query[:, : dim // 2], inst.query[:, dim // 2 :]
        q = k = query_pos + out_embed
        tgt = self.norm1(out_embed + self.dropout1(self.self_attn(q[:, None], k[:, None], value=out_embed[:, None])[0][:, 0]))
        tgt = self.norm2(tgt + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(tgt))))))
        if self.update_query_pos:
            query_pos = self.norm_pos(query_pos + self.dropout_pos2(self.linear_pos2(self.dropout_pos1(F.relu(self.linear_pos1(tgt))))))
        query_feat = self.norm_feat(query_feat + self.dropout_feat2(self.linear_feat2(self.dropout_feat1(F.relu(self.linear_feat1(tgt))))))
        inst.query = torch.cat([query_pos, query_feat], dim=1)
        return inst

    def _select_active_tracks(self, inst: Instances) -> Instances:
        if not self.training:
            return inst[inst.obj_idxes >= 0]
        active = inst[(inst.obj_idxes >= 0) & (inst.iou > 0.5)]
        if self.random_drop > 0 and len(active):
            active = active[torch.rand_like(active.scores) > self.random_drop]
        if self.fp_ratio > 0:
            inactive = inst[inst.obj_idxes < 0]
            num_fp = int(torch.bernoulli(torch.full((len(active),), self.fp_ratio, device=inst.scores.device)).sum()) if len(active) else 0
            if num_fp and len(inactive):
                fp = inactive[torch.argsort(inactive.scores)[-num_fp:]] if num_fp < len(inactive) else inactive
                active = Instances.cat([active, fp])
        return active

    def forward(self, inst: Instances) -> Instances:
        """Return the updated surviving track queries (detection queries are added by the caller)."""

        return self._update_track_embedding(self._select_active_tracks(inst))


class RuntimeTrackerBase:
    def __init__(self, score_thresh: float = 0.4, filter_score_thresh: float = 0.35, miss_tolerance: int = 5) -> None:
        self.score_thresh, self.filter_score_thresh, self.miss_tolerance = score_thresh, filter_score_thresh, miss_tolerance
        self.max_obj_id = 0

    def clear(self) -> None:
        self.max_obj_id = 0

    def update(self, inst: Instances) -> None:
        inst.disappear_time[inst.scores >= self.score_thresh] = 0
        new = (inst.obj_idxes == -1) & (inst.scores >= self.score_thresh)
        n_new = int(new.sum())
        if n_new:
            inst.obj_idxes[new] = torch.arange(self.max_obj_id, self.max_obj_id + n_new, device=inst.obj_idxes.device)
            self.max_obj_id += n_new
        sleeping = (inst.obj_idxes >= 0) & (inst.scores < self.filter_score_thresh) & ~new
        inst.disappear_time[sleeping] += 1
        dead = sleeping & (inst.disappear_time >= self.miss_tolerance)
        inst.obj_idxes[dead] = -1
