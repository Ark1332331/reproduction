"""R5: train the current minimal 4D sparse model on a batched 3D terrain set.

This file does not claim paper-level results. It verifies the first full loop:
    many current/history point-cloud observations
    -> batched 4D SparseTensor
    -> minimal candidate model
    -> paper-style losses
    -> Adam updates that lower the training loss
"""

from dataclasses import dataclass

import torch

from r5_sparse_loss import CompletionLoss, completion_loss
from r5_sparse_loss import downsample_occupancy_target, multiscale_likelihood_loss
from r5_sparse_model import FourLevel4DCompletionModel, Minimal4DCompletionModel
from r5_terrain_dataset import PreparedSparseDataset


@dataclass(frozen=True)
class SparseTrainingRun:
    """Inspectable result of one deterministic minimal training run."""

    model: Minimal4DCompletionModel
    initial_loss: float
    final_loss: float
    loss_history: tuple[float, ...]
    learning_rate_history: tuple[float, ...] = ()


def train_minimal_sparse_model(
    dataset: PreparedSparseDataset,
    hidden_channels: int = 8,
    steps: int = 80,
    learning_rate: float = 0.03,
    position_weight: float = 1.0,
    seed: int = 0,
) -> SparseTrainingRun:
    """Train the current one-level model and return its directly observed losses."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    torch.manual_seed(seed)
    model = Minimal4DCompletionModel(hidden_channels=hidden_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[float] = []

    for _ in range(steps):
        optimizer.zero_grad()
        predictions = model(dataset.input_tensor)
        losses = _losses_for_predictions(predictions, dataset, position_weight)
        losses.total.backward()
        optimizer.step()
        history.append(float(losses.total.detach()))

    return SparseTrainingRun(
        model=model,
        initial_loss=history[0],
        final_loss=history[-1],
        loss_history=tuple(history),
    )


def evaluate_minimal_sparse_model(
    model: Minimal4DCompletionModel,
    dataset: PreparedSparseDataset,
    position_weight: float = 1.0,
) -> CompletionLoss:
    """Measure the same paper-style loss without updating any model parameter."""
    model.eval()
    with torch.no_grad():
        predictions = model(dataset.input_tensor)
        return _losses_for_predictions(predictions, dataset, position_weight)


def _losses_for_predictions(
    predictions,
    dataset: PreparedSparseDataset,
    position_weight: float,
) -> CompletionLoss:
    return completion_loss(
        candidate_coordinates=predictions.candidates.C,
        occupancy_logits=predictions.occupancy_logits.F,
        position_offsets=predictions.position_offsets.F,
        target=dataset.target_batch,
        position_weight=position_weight,
    )


def four_level_training_loss(predictions, dataset: PreparedSparseDataset) -> torch.Tensor:
    """Combine final reconstruction loss with the paper's four likelihood BCE terms."""
    final = _losses_for_predictions(predictions, dataset, position_weight=1.0)
    return final.total + multiscale_likelihood_loss(predictions.decoder_likelihoods, dataset.target_batch)


def train_four_level_sparse_model(dataset: PreparedSparseDataset, steps: int = 40, learning_rate: float = 0.01, seed: int = 0) -> SparseTrainingRun:
    """Train with target-guarded pruning and the paper's exponential LR schedule."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    torch.manual_seed(seed)
    # SparseTensor features and module parameters must live on the same device.
    model = FourLevel4DCompletionModel().to(dataset.input_tensor.F.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    final_learning_rate = 0.0001
    gamma = 1.0 if steps == 1 else (final_learning_rate / learning_rate) ** (1 / (steps - 1))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    targets = tuple(torch.from_numpy(downsample_occupancy_target(dataset.target_batch, factor).coordinates) for factor in (8, 4, 2, 1))
    history=[]; learning_rate_history=[]
    for _ in range(steps):
        learning_rate_history.append(float(optimizer.param_groups[0]["lr"]))
        optimizer.zero_grad()
        prediction = model(dataset.input_tensor, alpha=0.5, training_target_coordinates=targets)
        loss = four_level_training_loss(prediction, dataset)
        loss.backward(); optimizer.step(); scheduler.step(); history.append(float(loss.detach()))
    return SparseTrainingRun(model, history[0], history[-1], tuple(history), tuple(learning_rate_history))
