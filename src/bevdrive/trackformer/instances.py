"""Minimal ``Instances`` container (port of UniAD/detectron2 ``track_instance.py``).

Every field is a tensor (or list) of equal length; indexing applies to all
fields. Track queries, their boxes, ids and memory live in one object.
"""

from __future__ import annotations

from typing import Any

import torch


class Instances:
    def __init__(self, **fields: Any) -> None:
        object.__setattr__(self, "_fields", {})
        for key, value in fields.items():
            self.set(key, value)

    def __setattr__(self, name: str, value: Any) -> None:
        self.set(name, value)

    def __getattr__(self, name: str) -> Any:
        fields = object.__getattribute__(self, "_fields")
        if name not in fields:
            raise AttributeError(f"Instances has no field '{name}'")
        return fields[name]

    def set(self, name: str, value: Any) -> None:
        if self._fields:
            assert len(value) == len(self), f"field {name}: length {len(value)} != {len(self)}"
        self._fields[name] = value

    def has(self, name: str) -> bool:
        return name in self._fields

    def fields(self) -> dict[str, Any]:
        return self._fields

    def __len__(self) -> int:
        for value in self._fields.values():
            return len(value)
        return 0

    def __getitem__(self, item: Any) -> "Instances":
        if isinstance(item, int):
            item = slice(item, item + 1)
        out = Instances()
        for key, value in self._fields.items():
            if isinstance(value, list):
                if isinstance(item, torch.Tensor):
                    idx = item.nonzero().flatten().tolist() if item.dtype == torch.bool else item.tolist()
                    out.set(key, [value[i] for i in idx])
                else:
                    out.set(key, value[item])
            else:
                out.set(key, value[item])
        return out

    def to(self, device: Any) -> "Instances":
        out = Instances()
        for key, value in self._fields.items():
            out.set(key, value.to(device) if hasattr(value, "to") else value)
        return out

    def detach_clone(self) -> "Instances":
        out = Instances()
        for key, value in self._fields.items():
            out.set(key, value.detach().clone() if isinstance(value, torch.Tensor) else list(value))
        return out

    @staticmethod
    def cat(items: list["Instances"]) -> "Instances":
        items = [i for i in items if len(i._fields)]
        if len(items) == 1:
            return items[0]
        out = Instances()
        for key in items[0]._fields:
            values = [i._fields[key] for i in items]
            if isinstance(values[0], torch.Tensor):
                out.set(key, torch.cat(values, dim=0))
            else:
                out.set(key, [v for vs in values for v in vs])
        return out
