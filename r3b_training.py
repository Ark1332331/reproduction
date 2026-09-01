"""R3-b4: train and apply the minimal structured toy completion model."""

import numpy as np
import torch
from torch import nn

from r3b_toy_model import ToyCompletionMLP


def train_completion_model(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    width: int,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> tuple[ToyCompletionMLP, list[float]]:
    """Fit one seeded MLP to the toy input-grid/complete-grid pairs."""
    torch.manual_seed(seed)
    model = ToyCompletionMLP(width=width)
    inputs = torch.as_tensor(train_inputs, dtype=torch.float32)
    targets = torch.as_tensor(train_targets, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
    losses: list[float] = []

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model(inputs), targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return model, losses


def predict_occupancy_grids(
    model: ToyCompletionMLP,
    input_grids: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Convert the model's occupancy logits into binary occupancy grids."""
    inputs = torch.as_tensor(input_grids, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(inputs))
    return (probabilities >= threshold).to(torch.int8).cpu().numpy()
    
