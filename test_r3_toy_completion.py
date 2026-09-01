"""R3 tests: verify the raw point-cloud completion baseline.

Data flow:
    R2 ToySceneDataset -> R3 merge/evaluation -> unittest assertions.
complete_points is the evaluation reference, not an input to the merge.
"""

# Standard library: discover and run the test methods.
import unittest

# NumPy: construct expected points and compare evaluation arrays.
import numpy as np

# R2 data source: creates the complete/current/previous toy clouds.
from r2_toy_dataset import make_step_terrain_dataset
# R3 implementation under test: merge now, evaluate after compare_completion exists.
from r3_toy_completion import compare_completion, merge_point_clouds


class R3ToyCompletionTests(unittest.TestCase):
    """Group tests for R3-a merging and completion evaluation."""

    def test_merges_current_and_previous_points_without_duplicates(self):
        """Current/history observations merge into unique raw points."""
        dataset = make_step_terrain_dataset(
            x_values=[0.0, 0.5, 1.0],
            y_values=[0.0],
            current_x_max=0.5,
            previous_x_min=0.5,
            previous_x_max=1.0,
            step_x=1.0,
        )

        merged = merge_point_clouds(
            dataset.current_points,
            dataset.previous_points,
        )

        np.testing.assert_allclose(
            merged,
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                    [1.0, 0.0, 0.3],
                ],
                dtype=float,
            ),
        )

    def test_reports_full_coverage_when_merge_matches_complete_scene(self):
        """A perfect merge has no missing/extra points and scores 1.0."""
        dataset = make_step_terrain_dataset(
            x_values=[0.0, 0.5, 1.0],
            y_values=[0.0],
            current_x_max=0.5,
            previous_x_min=0.5,
            previous_x_max=1.0,
            step_x=1.0,
        )
        merged = merge_point_clouds(
            dataset.current_points,
            dataset.previous_points,
        )

        evaluation = compare_completion(merged, dataset.complete_points)

        self.assertEqual(evaluation.missing_points.shape, (0, 3))
        self.assertEqual(evaluation.extra_points.shape, (0, 3))
        self.assertEqual(evaluation.coverage, 1.0)
        self.assertEqual(evaluation.precision, 1.0)

    def test_reports_missing_points(self):
        """Evaluation reports points present in complete but absent in merged."""
        complete = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ],
            dtype=float,
        )
        merged = complete[:1]

        evaluation = compare_completion(merged, complete)

        np.testing.assert_allclose(
            evaluation.missing_points,
            np.array([[0.5, 0.0, 0.0]], dtype=float),
        )
        self.assertEqual(evaluation.coverage, 0.5)

    def test_reports_extra_points(self):
        """Evaluation reports points present in merged but absent in complete."""
        complete = np.array([[0.0, 0.0, 0.0]], dtype=float)
        merged = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        evaluation = compare_completion(merged, complete)

        np.testing.assert_allclose(
            evaluation.extra_points,
            np.array([[1.0, 0.0, 0.0]], dtype=float),
        )
        self.assertEqual(evaluation.precision, 0.5)

    def test_rejects_non_n_by_3_point_cloud(self):
        """R3 rejects inputs that are not N x 3 point clouds."""
        with self.assertRaises(ValueError):
            merge_point_clouds(
                current_points=[[0.0, 0.0]],
                previous_points=np.empty((0, 3), dtype=float),
            )


if __name__ == "__main__":
    unittest.main()
