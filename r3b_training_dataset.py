"""R3-b1: build one structured, partially observed occupancy-grid sample."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StructuredCompletionSample:
    """One supervised input/target pair for the R3-b toy completion task."""

    complete_grid: np.ndarray
    current_grid: np.ndarray
    previous_grid: np.ndarray
    input_grid: np.ndarray


@dataclass(frozen=True)
class StructuredCompletionDataset:
    """Fixed-shape train/test arrays built from many structured completion samples."""

    train_inputs: np.ndarray
    train_targets: np.ndarray
    train_blind_xs: np.ndarray
    test_inputs: np.ndarray
    test_targets: np.ndarray
    test_blind_xs: np.ndarray


def make_structured_completion_sample(
    width: int,
    platform_start: int,
    platform_end: int,
    blind_x: int,
) -> StructuredCompletionSample:
    """Build a two-layer terrain with one real high cell hidden from both sources."""
    if width < 4:
        raise ValueError("width must be at least 4")
    if not 0 <= platform_start < platform_end < width:
        raise ValueError("platform bounds must satisfy 0 <= start < end < width")
    if not platform_start < blind_x < platform_end:
        raise ValueError("blind_x must be an internal platform cell")

    complete_grid = np.zeros((2, width), dtype=np.int8)
    complete_grid[0, :] = 1
    complete_grid[1, platform_start : platform_end + 1] = 1

    current_grid = np.zeros_like(complete_grid)
    current_grid[0, :platform_start] = 1

    previous_grid = np.zeros_like(complete_grid)
    previous_grid[0, platform_start:] = 1
    previous_grid[1, platform_start : platform_end + 1] = 1
    previous_grid[1, blind_x] = 0

    input_grid = np.stack([current_grid, previous_grid])
    return StructuredCompletionSample(
        complete_grid=complete_grid,
        current_grid=current_grid,
        previous_grid=previous_grid,
        input_grid=input_grid,
    )


def make_structured_completion_dataset(
    width: int = 8,
) -> StructuredCompletionDataset:
    """Enumerate structured samples and reserve one blind pattern for testing."""
    if width < 5:
        raise ValueError("width must be at least 5 for nonempty train and test splits")

    train_inputs: list[np.ndarray] = []
    train_targets: list[np.ndarray] = []
    train_blind_xs: list[int] = []
    test_inputs: list[np.ndarray] = []
    test_targets: list[np.ndarray] = []
    test_blind_xs: list[int] = []

    for platform_start in range(1, width - 2):
        for platform_end in range(platform_start + 2, width):
            for blind_x in range(platform_start + 1, platform_end):
                sample = make_structured_completion_sample(
                    width=width,
                    platform_start=platform_start,
                    platform_end=platform_end,
                    blind_x=blind_x,
                )
                if blind_x == platform_start + 1:
                    test_inputs.append(sample.input_grid)
                    test_targets.append(sample.complete_grid)
                    test_blind_xs.append(blind_x)
                else:
                    train_inputs.append(sample.input_grid)
                    train_targets.append(sample.complete_grid)
                    train_blind_xs.append(blind_x)

    return StructuredCompletionDataset(
        train_inputs=np.stack(train_inputs),
        train_targets=np.stack(train_targets),
        train_blind_xs=np.asarray(train_blind_xs, dtype=int),
        test_inputs=np.stack(test_inputs),
        test_targets=np.stack(test_targets),
        test_blind_xs=np.asarray(test_blind_xs, dtype=int),
    )
