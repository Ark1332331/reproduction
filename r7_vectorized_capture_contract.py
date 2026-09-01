"""Pure naming and layout contracts for vectorized R7 IsaacLab collection.

This module deliberately has no Isaac Lab imports.  It gives the batch collector
stable, inspectable identities before any GPU simulation starts.
"""

from __future__ import annotations

from pathlib import Path


CAMERA_DIRECTIONS = ("front", "back", "left", "right")


def layout_seed_for_environment(base_seed: int, environment_index: int) -> int:
    """Give each parallel terrain tile a deterministic, distinct layout seed."""
    if base_seed < 0 or environment_index < 0:
        raise ValueError("base_seed and environment_index must be non-negative")
    return base_seed + environment_index


def output_path_for_environment(output_dir: Path, terrain: str, base_seed: int, environment_index: int) -> Path:
    """Keep every parallel trajectory separate so it remains one R7 NPZ sample."""
    layout_seed = layout_seed_for_environment(base_seed, environment_index)
    return output_dir / f"isaac_anymal_{terrain}_s{layout_seed}_e{environment_index}.npz"


def required_camera_count(num_environments: int) -> int:
    """Each environment owns one camera for every paper capture direction."""
    if num_environments <= 0:
        raise ValueError("num_environments must be positive")
    return num_environments * len(CAMERA_DIRECTIONS)
