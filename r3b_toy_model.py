"""R3-b4: minimal PyTorch model for structured toy-grid completion."""

import torch
from torch import nn


class ToyCompletionMLP(nn.Module):
    """Map current/history occupancy grids to one complete-scene score grid."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.network = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(4 * width, 32),
            nn.ReLU(),
            nn.Linear(32, 2 * width),
        )

    def forward(self, input_grids: torch.Tensor) -> torch.Tensor:
        score_vectors = self.network(input_grids)
        return score_vectors.reshape(-1, 2, self.width)
