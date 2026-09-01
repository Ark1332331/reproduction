"""R2 tests: verify the toy complete/current/previous data contract.

Data flow:
    generated complete terrain -> partial observations -> R1 integration test.
"""

# Standard library: discover and run the test methods.
import unittest

# NumPy: construct expected toy point clouds and inspect coordinates.
import numpy as np

# R1 handoff used by the final R2 integration test.
from r1_data_representation import points_to_voxel_representation_with_aligned_previous
# R2 implementation under test: build complete/current/previous clouds.
from r2_toy_dataset import (
    generate_step_terrain,
    make_partial_observation,
    make_step_terrain_dataset,
)


class R2ToyDatasetTests(unittest.TestCase):
    """Group tests for toy terrain construction and its R1 handoff."""

    def test_generates_complete_step_terrain(self):
        """The artificial ground-plus-step point cloud is correct."""
        points = generate_step_terrain(
            x_values=[0.0, 1.0],
            y_values=[0.0, 1.0],
            step_x=1.0,
            low_z=0.0,
            high_z=0.3,
        )

        np.testing.assert_allclose(
            points,
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.3],
                    [1.0, 1.0, 0.3],
                ],
                dtype=float,
            ),
        )

    def test_makes_partial_observation_by_x_range(self):
        """An x range selects the expected local observation."""
        complete_points = generate_step_terrain(
            x_values=[0.0, 0.5, 1.0, 1.5],
            y_values=[0.0],
            step_x=1.0,
        )

        observed = make_partial_observation(
            complete_points,
            x_min=0.5,
            x_max=1.0,
        )

        np.testing.assert_allclose(
            observed,
            np.array(
                [
                    [0.5, 0.0, 0.0],
                    [1.0, 0.0, 0.3],
                ],
                dtype=float,
            ),
        )

    def test_builds_current_previous_and_complete_points(self):
        """The dataset contains complete, current, and previous clouds."""
        dataset = make_step_terrain_dataset(
            x_values=[0.0, 0.5, 1.0, 1.5],
            y_values=[0.0],
            current_x_max=1.0,
            previous_x_min=0.5,
            previous_x_max=1.5,
            step_x=1.0,
        )

        self.assertEqual(len(dataset.complete_points), 4)
        np.testing.assert_allclose(
            dataset.current_points,
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                    [1.0, 0.0, 0.3],
                ],
                dtype=float,
            ),
        )
        np.testing.assert_allclose(
            dataset.previous_points,
            np.array(
                [
                    [0.5, 0.0, 0.0],
                    [1.0, 0.0, 0.3],
                    [1.5, 0.0, 0.3],
                ],
                dtype=float,
            ),
        )

    def test_toy_dataset_can_enter_aligned_voxelization(self):
        """R2 output can enter R1 alignment and voxelization."""
        dataset = make_step_terrain_dataset(
            x_values=[0.0, 0.5, 1.0],
            y_values=[0.0],
            current_x_max=0.5,
            previous_x_min=0.5,
            previous_x_max=1.0,
            step_x=1.0,
        )

        result = points_to_voxel_representation_with_aligned_previous(
            current_points=dataset.current_points,
            previous_points=dataset.previous_points,
            translation=(0.0, 0.0, 0.0),
            yaw=0.0,
            voxel_size=0.5,
            grid_size=8,
            origin=(0.0, 0.0, 0.0),
        )

        self.assertTrue(np.any(np.all(result.coords == np.array([0, 0, 0, 0]), axis=1)))
        self.assertTrue(np.any(np.all(result.coords == np.array([1, 0, 0, 0]), axis=1)))
        self.assertTrue(np.any(np.all(result.coords == np.array([2, 0, 0, 1]), axis=1)))


if __name__ == "__main__":
    unittest.main()
