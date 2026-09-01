"""R1 tests: verify alignment, voxelization, c_i, f_i, k, and filtering.

Data flow:
    hand-written N x 3 points -> R1 functions -> VoxelRepresentation -> asserts.
The tests are the executable contract for reproduction/r1_data_representation.py.
"""

# Standard library: discover and run the test methods.
import unittest

# NumPy: construct expected point/voxel arrays and compare results.
import numpy as np

# R1 implementation under test: alignment and voxel representation.
from r1_data_representation import (
    align_previous_points_to_current,
    points_to_voxel_representation,
    points_to_voxel_representation_with_aligned_previous,
)


class R1DataRepresentationTests(unittest.TestCase):
    """Group tests for the R1 data-representation pipeline."""

    def test_aligns_previous_points_before_voxelization(self):
        """Alignment happens before the previous cloud is voxelized."""
        current_points = np.empty((0, 3), dtype=float)
        previous_points = np.array([[2.0, 0.0, 0.0]], dtype=float)

        result = points_to_voxel_representation_with_aligned_previous(
            current_points=current_points,
            previous_points=previous_points,
            translation=(1.0, 0.0, 0.0),
            yaw=0.0,
            voxel_size=0.5,
            grid_size=8,
            origin=(0.0, 0.0, 0.0),
        )

        np.testing.assert_array_equal(result.coords, np.array([[2, 0, 0, 1]]))
        np.testing.assert_allclose(result.features, np.array([[0.0, 0.0, 0.0]]))
        self.assertEqual(result.dropped_count, 0)

    def test_aligns_previous_points_with_translation_only(self):
        """A translation changes the previous cloud into the current frame."""
        previous_points = np.array([[2.0, 0.0, 0.0]], dtype=float)

        aligned = align_previous_points_to_current(
            previous_points=previous_points,
            translation=(1.0, 0.0, 0.0),
            yaw=0.0,
        )

        np.testing.assert_allclose(aligned, np.array([[1.0, 0.0, 0.0]]), atol=1e-9)

    def test_aligns_previous_points_with_yaw_only(self):
        """A yaw rotation changes the previous cloud into the current frame."""
        previous_points = np.array([[2.0, 0.0, 0.0]], dtype=float)

        aligned = align_previous_points_to_current(
            previous_points=previous_points,
            translation=(0.0, 0.0, 0.0),
            yaw=np.pi / 2,
        )

        np.testing.assert_allclose(aligned, np.array([[0.0, -2.0, 0.0]]), atol=1e-9)

    def test_aligns_yaw_about_the_robot_not_the_bottom_left_map_corner(self):
        """A 0..3.2 local map stores the robot at its centre, not at (0, 0)."""
        aligned = align_previous_points_to_current(
            previous_points=np.array([[2.6, 1.6, 0.4]], dtype=float),
            translation=(0.0, 0.0, 0.0),
            yaw=np.pi / 2,
            rotation_center=(1.6, 1.6, 1.6),
        )

        np.testing.assert_allclose(aligned, np.array([[1.6, 0.6, 0.4]]), atol=1e-9)

    def test_converts_current_and_previous_points_to_coords_and_features(self):
        """Voxelization produces c_i/f_i and drops out-of-grid points."""
        current_points = np.array(
            [
                [0.12, 0.10, 0.10],
                [0.80, 0.50, 0.20],
            ],
            dtype=float,
        )
        previous_points = np.array(
            [
                [0.00, 0.05, 0.10],
                [3.25, 0.10, 0.10],
            ],
            dtype=float,
        )

        result = points_to_voxel_representation(
            current_points=current_points,
            previous_points=previous_points,
            voxel_size=0.05,
            grid_size=64,
            origin=(0.0, 0.0, 0.0),
        )

        np.testing.assert_array_equal(
            result.coords,
            np.array(
                [
                    [2, 2, 2, 0],
                    [16, 10, 4, 0],
                    [0, 1, 2, 1],
                ],
                dtype=int,
            ),
        )
        np.testing.assert_allclose(
            result.features,
            np.array(
                [
                    [0.4, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=float,
            ),
            atol=1e-9,
        )
        self.assertEqual(result.dropped_count, 1)

    def test_rejects_points_with_wrong_shape(self):
        """Inputs must be point clouds with shape N x 3."""
        with self.assertRaisesRegex(ValueError, "N x 3"):
            points_to_voxel_representation(
                current_points=np.array([0.1, 0.2, 0.3]),
                previous_points=np.empty((0, 3)),
            )

    def test_uses_centroid_when_multiple_points_fall_in_same_voxel(self):
        """Multiple points in one voxel become one voxel feature."""
        current_points = np.array(
            [
                [0.11, 0.10, 0.10],
                [0.13, 0.10, 0.10],
            ],
            dtype=float,
        )

        result = points_to_voxel_representation(
            current_points=current_points,
            previous_points=np.empty((0, 3)),
            voxel_size=0.05,
            grid_size=64,
            origin=(0.0, 0.0, 0.0),
        )

        np.testing.assert_array_equal(result.coords, np.array([[2, 2, 2, 0]]))
        np.testing.assert_allclose(result.features, np.array([[0.4, 0.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
