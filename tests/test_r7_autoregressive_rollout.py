"""R7 tests: a previous model estimate, not an old measurement, becomes k=1."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from rollout.r7_autoregressive_rollout import (
    TemporalTerrainStep,
    make_sparse_model_predictor,
    make_autoregressive_voxels,
    evaluate_detached_rollout,
    load_isaaclab_temporal_trajectory,
    train_detached_rollout,
    rollout_without_temporal_gradients,
    sparse_prediction_to_current_points,
    _summarize_metrics,
)
from evaluation.r5_sparse_evaluation import SparseHeightMetrics, SparseOccupancyMetrics


class R7AutoregressiveRolloutTests(unittest.TestCase):
    def test_aligns_previous_network_estimate_and_marks_it_as_time_one(self):
        representation = make_autoregressive_voxels(
            current_measurement=np.array([[1.1, 0.1, 0.1]]),
            previous_estimate=np.array([[1.5, 0.1, 0.1]]),
            previous_to_current_translation=(0.5, 0.0, 0.0),
            previous_to_current_yaw=0.0,
            voxel_size=1.0,
            grid_size=4,
        )

        self.assertEqual(set(map(tuple, representation.coords)), {(1, 0, 0, 0), (1, 0, 0, 1)})

    def test_default_rotation_center_is_the_middle_of_the_voxel_map(self):
        representation = make_autoregressive_voxels(
            current_measurement=np.empty((0, 3)),
            previous_estimate=np.array([[2.6, 1.6, 0.45]]),
            previous_to_current_translation=(0.0, 0.0, 0.0),
            previous_to_current_yaw=np.pi / 2,
            voxel_size=.1,
            grid_size=32,
        )

        self.assertTrue(np.any(np.all(representation.coords == np.array([16, 6, 4, 1]), axis=1)))

    def test_rollout_feeds_predictor_output_into_the_next_step_not_old_measurement(self):
        calls = []

        def predict(representation):
            calls.append(representation)
            return np.array([[2.1, 0.1, 0.1]])

        steps = (
            TemporalTerrainStep(
                current_measurement=np.array([[0.1, 0.1, 0.1]]),
                target_points=np.array([[0.1, 0.1, 0.1]]),
                previous_to_current_translation=(0.0, 0.0, 0.0),
                previous_to_current_yaw=0.0,
            ),
            TemporalTerrainStep(
                current_measurement=np.array([[0.1, 0.1, 0.1]]),
                target_points=np.array([[0.1, 0.1, 0.1]]),
                previous_to_current_translation=(0.0, 0.0, 0.0),
                previous_to_current_yaw=0.0,
            ),
        )

        rollout = rollout_without_temporal_gradients(steps, predict, voxel_size=1.0, grid_size=4)

        self.assertEqual(len(rollout.inputs), 2)
        self.assertTrue(np.any(np.all(calls[1].coords == np.array([2, 0, 0, 1]), axis=1)))
        self.assertEqual(len(rollout.estimates), 2)

    def test_decodes_only_current_time_predictions_that_pass_the_likelihood_threshold(self):
        points = sparse_prediction_to_current_points(
            coordinates=torch.tensor([[0, 1, 2, 3, 0], [0, 2, 2, 2, 1], [0, 3, 3, 3, 0]]),
            likelihood_logits=torch.tensor([[0.0], [4.0], [-4.0]]),
            offsets=torch.tensor([[.5, .25, .0], [.1, .1, .1], [.5, .5, .5]]),
            alpha=.5,
            voxel_size=.2,
        )

        np.testing.assert_allclose(points, [[.3, .45, .6]])

    def test_real_sparse_model_can_cross_the_detached_point_cloud_boundary(self):
        from models.r5_sparse_model import FourLevel4DCompletionModel

        representation = make_autoregressive_voxels(
            current_measurement=np.array([[.1, .1, .1], [.6, .1, .1]]),
            previous_estimate=np.empty((0, 3)),
            previous_to_current_translation=(0.0, 0.0, 0.0),
            previous_to_current_yaw=0.0,
            voxel_size=.5,
            grid_size=8,
        )
        points = make_sparse_model_predictor(FourLevel4DCompletionModel(), alpha=0.0, voxel_size=.5)(representation)

        self.assertEqual(points.ndim, 2)
        self.assertEqual(points.shape[1], 3)
        self.assertTrue(np.isfinite(points).all())

    def test_trains_a_two_step_rollout_without_backpropagating_through_feedback(self):
        from models.r5_sparse_model import FourLevel4DCompletionModel

        trajectory = tuple(
            TemporalTerrainStep(
                current_measurement=np.array([[.1, .1, .1], [.6, .1, .1]]),
                target_points=np.array([[.1, .1, .1], [.6, .1, .1], [1.1, .1, .1]]),
                previous_to_current_translation=(0.0, 0.0, 0.0),
                previous_to_current_yaw=0.0,
            )
            for _ in range(2)
        )
        run = train_detached_rollout(
            FourLevel4DCompletionModel(), [trajectory], steps=2, voxel_size=.5, grid_size=8
        )

        self.assertEqual(len(run.loss_history), 2)
        self.assertEqual(len(run.learning_rate_history), 2)
        self.assertGreater(run.learning_rate_history[0], run.learning_rate_history[-1])
        self.assertTrue(np.isfinite(run.final_loss))

    def test_loads_explicit_robot_centric_trajectory_contract(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.npz"
            np.savez(
                path,
                metadata_json='{"coordinate_frame":"robot_centric_local_map","trajectory_steps":2}',
                measurement_00=np.array([[.1, .1, .1]]),
                target_00=np.array([[.1, .1, .1]]),
                translation_00=np.array([0., 0., 0.]),
                measurement_01=np.array([[.2, .1, .1]]),
                target_01=np.array([[.2, .1, .1]]),
                translation_01=np.array([.1, 0., 0.]),
                yaw_01=np.array(.25),
            )

            steps = load_isaaclab_temporal_trajectory(str(path))

        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[1].previous_to_current_translation, (.1, 0., 0.))
        self.assertEqual(steps[1].previous_to_current_yaw, .25)

    def test_evaluation_reports_baseline_and_autoregressive_metrics_even_if_pruning_is_empty(self):
        from models.r5_sparse_model import FourLevel4DCompletionModel

        trajectory = [
            TemporalTerrainStep(
                current_measurement=np.array([[.1, .1, .1]]),
                target_points=np.array([[.1, .1, .1], [.6, .1, .1]]),
                previous_to_current_translation=(0.0, 0.0, 0.0),
                previous_to_current_yaw=0.0,
            )
        ]
        report = evaluate_detached_rollout(FourLevel4DCompletionModel(), trajectory, voxel_size=.5, grid_size=8)

        self.assertEqual(report["frame_count"], 1)
        self.assertIn("f1", report["current_measurement_baseline"]["occupancy"])
        self.assertIn("mean_absolute_error", report["autoregressive_model"]["height"])

    def test_evaluation_exposes_paper_macro_and_legacy_micro_aggregates(self):
        occupancy = (
            SparseOccupancyMetrics(1, 1, 0, 0.5, 1.0, 2.0 / 3.0),
            SparseOccupancyMetrics(1, 0, 1, 1.0, 0.5, 2.0 / 3.0),
        )
        heights = (
            SparseHeightMetrics(1, 2, 0.1),
            SparseHeightMetrics(2, 2, 0.3),
        )

        summary = _summarize_metrics(occupancy, heights)

        self.assertEqual(summary["aggregation"]["paper_primary"], "macro_per_frame")
        self.assertAlmostEqual(summary["macro"]["occupancy"]["precision"], 0.75)
        self.assertAlmostEqual(summary["macro"]["height"]["mean_absolute_error"], 0.2)
        self.assertAlmostEqual(summary["micro"]["occupancy"]["precision"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["micro"]["height"]["coverage"], 0.75)


if __name__ == "__main__":
    unittest.main()
