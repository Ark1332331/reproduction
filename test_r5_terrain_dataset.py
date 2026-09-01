"""R5 tests: one hand-checkable 3D terrain sample through the real sparse path."""

import unittest

import numpy as np
import torch

from r5_sparse_loss import completion_loss
from r5_sparse_model import Minimal4DCompletionModel
from r5_terrain_dataset import make_structured_terrain_sample, prepare_sparse_sample


class R5TerrainDatasetTests(unittest.TestCase):
    """Verify the first non-grid input/output chain for R5."""

    def test_builds_complete_current_and_history_views_of_one_3d_step(self):
        sample = make_structured_terrain_sample()

        self.assertEqual(sample.complete_points.shape, (8, 3))
        self.assertEqual(sample.current_points.shape, (6, 3))
        self.assertEqual(sample.previous_points.shape, (4, 3))
        self.assertTrue(np.all(sample.complete_points[sample.complete_points[:, 0] < 1.0, 2] == 0.1))
        self.assertTrue(np.all(sample.complete_points[sample.complete_points[:, 0] >= 1.0, 2] == 0.6))
        self.assertFalse(np.any(np.isclose(sample.current_points[:, 0], 1.6)))

    def test_prepares_two_time_input_but_current_only_target(self):
        prepared = prepare_sparse_sample(make_structured_terrain_sample())

        self.assertEqual(prepared.input_representation.coords.shape, (10, 4))
        self.assertEqual(prepared.target_representation.coords.shape, (8, 4))
        self.assertEqual(prepared.input_batch.coordinates.shape, (10, 5))
        self.assertEqual(prepared.target_batch.coordinates.shape, (8, 5))
        self.assertEqual(set(prepared.input_batch.coordinates[:, 4]), {0, 1})
        self.assertEqual(set(prepared.target_batch.coordinates[:, 4]), {0})
        self.assertEqual(set(prepared.input_batch.coordinates[:, 0]), {0})
        self.assertEqual(set(prepared.target_batch.coordinates[:, 0]), {0})

    def test_runs_one_3d_sample_from_observations_to_finite_paper_losses(self):
        prepared = prepare_sparse_sample(make_structured_terrain_sample())
        model = Minimal4DCompletionModel(hidden_channels=4)

        predictions = model(prepared.input_tensor)
        result = completion_loss(
            candidate_coordinates=predictions.candidates.C,
            occupancy_logits=predictions.occupancy_logits.F,
            position_offsets=predictions.position_offsets.F,
            target=prepared.target_batch,
        )

        self.assertGreater(result.positive_count, 0)
        self.assertTrue(torch.isfinite(result.total))
        result.total.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
