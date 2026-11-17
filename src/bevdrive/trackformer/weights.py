"""Initialise BEVFusion-TrackFormer from the UniAD stage-1 (track+map) checkpoint.

Only modules whose *interface* is unchanged are transferred: the decoder,
class/regression branches, QIM, memory bank, reference-point head, the
learned BEV positional embedding and (optionally) the 900 learned queries.
Everything that touches BEVFusion features (adapter, proposal encoders) is
new and must be trained. The image backbone / BEVFormer encoder / map head of
UniAD are not used.
"""

from __future__ import annotations

from pathlib import Path

import torch

UNIAD_STAGE1 = Path("/home/anhhd/GIT/UniAD/ckpts/uniad_base_track_map.pth")

PREFIX_MAP = (
    ("pts_bbox_head.transformer.decoder.", "decoder."),
    ("pts_bbox_head.cls_branches.", "cls_branches."),
    ("pts_bbox_head.reg_branches.", "reg_branches."),
    ("pts_bbox_head.positional_encoding.", "bev_adapter.positional_encoding."),
    ("query_interact.", "query_interact."),
    ("memory_bank.", "memory_bank."),
    ("reference_points.", "reference_points."),
    ("query_embedding.", "query_embedding."),
)


def load_uniad_track_weights(model: torch.nn.Module, checkpoint: Path | str = UNIAD_STAGE1) -> dict[str, list[str]]:
    state = torch.load(str(checkpoint), map_location="cpu")
    state = state.get("state_dict", state)
    own = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    skipped_shape: list[str] = []
    for key, value in state.items():
        for src, dst in PREFIX_MAP:
            if key.startswith(src):
                target = dst + key[len(src):]
                if target not in own:
                    break
                if target == "query_embedding.weight" and value.shape[0] == own[target].shape[0] + 1:
                    value = value[:-1]  # drop UniAD's ego (SDC) query
                if own[target].shape != value.shape:
                    skipped_shape.append(f"{target}: ckpt {tuple(value.shape)} vs model {tuple(own[target].shape)}")
                    break
                mapped[target] = value
                break
    missing = sorted(set(own) - set(mapped))
    model.load_state_dict(mapped, strict=False)
    return {"loaded": sorted(mapped), "not_initialised": missing, "shape_mismatch": skipped_shape}
