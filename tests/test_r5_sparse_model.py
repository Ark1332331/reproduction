"""R5 tests: the smallest real 4D encoder-decoder/pruning slice."""

import unittest

import torch

from models.r5_sparse_input import SparseVoxelBatch, make_sparse_tensor
from models.r5_sparse_model import prune_candidates
from tests.sparse_model_fixtures import Minimal4DCompletionModel


class R5SparseModelTests(unittest.TestCase):
    """Verify real sparse operations before building the full paper network."""

    def setUp(self):
        sparse_batch = SparseVoxelBatch(
            coordinates=torch.tensor(
                [[0, 2, 2, 2, 0], [0, 3, 2, 2, 1]],
                dtype=torch.int32,
            ).numpy(),
            features=torch.tensor(
                [[0.4, 0.0, 0.0], [0.1, 0.2, 0.3]],
                dtype=torch.float32,
            ).numpy(),
        )
        self.input_tensor = make_sparse_tensor(sparse_batch)

    def test_generates_spatial_candidates_and_two_paper_outputs(self):
        model = Minimal4DCompletionModel(hidden_channels=4)

        predictions = model(self.input_tensor)

        self.assertEqual(predictions.candidates.D, 4)
        self.assertEqual(predictions.candidates.tensor_stride, [1, 1, 1, 1])
        self.assertGreater(len(predictions.candidates.C), len(self.input_tensor.C))
        self.assertEqual(tuple(predictions.occupancy_logits.F.shape)[1:], (1,))
        self.assertEqual(tuple(predictions.position_offsets.F.shape)[1:], (3,))
        self.assertTrue(torch.equal(predictions.candidates.C, predictions.occupancy_logits.C))
        self.assertTrue(torch.equal(predictions.candidates.C, predictions.position_offsets.C))

    def test_pruning_uses_sigmoid_likelihood_and_keeps_only_passing_candidates(self):
        model = Minimal4DCompletionModel(hidden_channels=4)
        predictions = model(self.input_tensor)
        logits = torch.full_like(predictions.occupancy_logits.F, -2.0)
        logits[0, 0] = 2.0
        logits[1, 0] = 0.0

        kept = prune_candidates(predictions.candidates, logits, alpha=0.5)

        self.assertEqual(len(kept.C), 2)
        self.assertEqual(
            {tuple(row.tolist()) for row in kept.C},
            {tuple(row.tolist()) for row in predictions.candidates.C[:2]},
        )

    def test_target_guard_keeps_ground_truth_candidate_during_training(self):
        model = Minimal4DCompletionModel(hidden_channels=4)
        predictions = model(self.input_tensor)
        logits = torch.full_like(predictions.occupancy_logits.F, -10.0)
        target = predictions.candidates.C[[0]]

        kept = prune_candidates(predictions.candidates, logits, alpha=0.5, target_coordinates=target)

        self.assertEqual(len(kept.C), 1)
        self.assertTrue(torch.equal(kept.C, target))

    def test_prediction_heads_can_backpropagate_to_model_parameters(self):
        model = Minimal4DCompletionModel(hidden_channels=4)

        predictions = model(self.input_tensor)
        loss = predictions.occupancy_logits.F.square().mean()
        loss = loss + predictions.position_offsets.F.square().mean()
        loss.backward()

        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
