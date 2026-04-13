"""Shared HLS configuration space helpers."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np


ParamSpec = Tuple[str, List[int], float, float]


def iter_param_specs(param_ranges: Dict[str, List[int]]) -> Iterable[ParamSpec]:
    for name, values in param_ranges.items():
        if not values:
            raise ValueError(f"Parameter range '{name}' is empty")
        low = float(min(values))
        high = float(max(values))
        scale = high - low
        if scale <= 0.0:
            raise ValueError(f"Parameter range '{name}' must span more than one value")
        yield name, list(values), low, scale


def action_space_size(param_ranges: Dict[str, List[int]]) -> int:
    total = 1
    for values in param_ranges.values():
        total *= len(values)
    return total


def decode_action_to_config(action_idx: int, param_ranges: Dict[str, List[int]]) -> Dict:
    """Decode an action index using the JSON-defined parameter order."""
    config: Dict = {}
    idx = action_idx
    for param_name, values, _, _ in reversed(list(iter_param_specs(param_ranges))):
        config[param_name] = values[idx % len(values)]
        idx //= len(values)
    return config


def normalise_config(config: Dict, param_ranges: Dict[str, List[int]]) -> np.ndarray:
    """Convert a raw HLS config dict to the model's 10-dim normalised vector."""
    specs = list(iter_param_specs(param_ranges))
    values = np.empty(len(specs), dtype=np.float32)
    for index, (name, _, offset, scale) in enumerate(specs):
        values[index] = (config.get(name, offset) - offset) / scale
    np.clip(values, 0.0, 1.0, out=values)
    return values