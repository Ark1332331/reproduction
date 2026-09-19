"""R5 tests: evaluate current-frame sparse voxel occupancy like the paper."""

import unittest

import numpy as np
import torch

from evaluation.r5_sparse_evaluation import (
    current_frame_merge_baseline,
    current_frame_merge_baseline_with_offsets,
    current_frame_prediction_coordinates,
    current_frame_prediction_with_offsets,
    height_metrics,
    occupancy_metrics,
)
from models.r5_sparse_input import SparseVoxelBatch


class R5SparseEvaluationTests(unittest.TestCase):
    """Keep baseline, predicted output, and target in the same coordinate format."""

    def setUp(self):
        self.target = SparseVoxelBatch(
            coordinates=np.array(
                [[0, 1, 1, 1, 0], [0, 2, 1, 1, 0]],
                dtype=np.int32,
            ),
            features=np.zeros((2, 3), dtype=np.float32),
        )

    def test_merge_baseline_relabels_history_as_current_and_removes_duplicates(self):
        observed_input = SparseVoxelBatch(
            coordinates=np.array(
                [
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1],
                    [0, 2, 1, 1, 1],
                ],
                dtype=np.int32,
            ),
            features=np.zeros((3, 3), dtype=np.float32),
        )

        baseline = current_frame_merge_baseline(observed_input)

        np.testing.assert_array_equal(
            baseline,
            np.array([[0, 1, 1, 1, 0], [0, 2, 1, 1, 0]], dtype=np.int32),
        )

    def test_prediction_keeps_only_current_candidates_above_threshold(self):
        coordinates = torch.tensor(
            [[0, 1, 1, 1, 0], [0, 2, 1, 1, 0], [0, 2, 1, 1, 1]],
            dtype=torch.int32,
        )
        logits = torch.tensor([[2.0], [-2.0], [3.0]])

        predicted = current_frame_prediction_coordinates(coordinates, logits, alpha=0.5)

        np.testing.assert_array_equal(predicted, np.array([[0, 1, 1, 1, 0]], dtype=np.int32))

    def test_reports_precision_recall_and_f1_from_coordinate_sets(self):
        predicted = np.array(
            [[0, 1, 1, 1, 0], [0, 3, 1, 1, 0]],
            dtype=np.int32,
        )

        metrics = occupancy_metrics(predicted, self.target)

        self.assertEqual(metrics.true_positive, 1)
        self.assertEqual(metrics.false_positive, 1)
        self.assertEqual(metrics.false_negative, 1)
        self.assertEqual(metrics.precision, 0.5)
        self.assertEqual(metrics.recall, 0.5)
        self.assertEqual(metrics.f1, 0.5)

    def test_keeps_offsets_then_measures_surface_height_mae(self):
        observed_input = SparseVoxelBatch(
            coordinates=np.array([[0, 1, 1, 1, 0], [0, 2, 1, 1, 1]], dtype=np.int32),
            features=np.zeros((2, 3), dtype=np.float32),
        )
        baseline_coordinates, baseline_offsets = current_frame_merge_baseline_with_offsets(observed_input)
        baseline_height = height_metrics(baseline_coordinates, baseline_offsets, self.target, voxel_size=0.5)
        self.assertEqual(baseline_height.matched_cell_count, 2)
        self.assertEqual(baseline_height.target_cell_count, 2)
        self.assertEqual(baseline_height.mean_absolute_error, 0.0)

        candidates = torch.tensor([[0, 1, 1, 1, 0], [0, 2, 1, 1, 0]], dtype=torch.int32)
        logits = torch.tensor([[8.0], [8.0]])
        offsets = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.4]])
        predicted_coordinates, predicted_offsets = current_frame_prediction_with_offsets(
            candidates, logits, offsets, alpha=0.5
        )
        model_height = height_metrics(predicted_coordinates, predicted_offsets, self.target, voxel_size=0.5)
        self.assertEqual(model_height.matched_cell_count, 2)
        self.assertAlmostEqual(model_height.mean_absolute_error, 0.1)


if __name__ == "__main__":
    unittest.main()
