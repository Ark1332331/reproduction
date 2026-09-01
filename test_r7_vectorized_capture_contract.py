"""Tests for the GPU-independent contracts of vectorized R7 collection."""

import unittest
from pathlib import Path

from r7_vectorized_capture_contract import (
    CAMERA_DIRECTIONS,
    layout_seed_for_environment,
    output_path_for_environment,
    required_camera_count,
)


class VectorizedCaptureContractTests(unittest.TestCase):
    def test_parallel_environments_receive_distinct_stable_layout_seeds(self):
        self.assertEqual([layout_seed_for_environment(40, index) for index in range(3)], [40, 41, 42])

    def test_each_parallel_trajectory_gets_its_own_r7_output_file(self):
        self.assertEqual(
            output_path_for_environment(Path("data"), "stairs", 40, 1),
            Path("data/isaac_anymal_stairs_s41_e1.npz"),
        )

    def test_camera_budget_is_four_cameras_per_environment(self):
        self.assertEqual(CAMERA_DIRECTIONS, ("front", "back", "left", "right"))
        self.assertEqual(required_camera_count(2), 8)
        with self.assertRaises(ValueError):
            required_camera_count(0)


if __name__ == "__main__":
    unittest.main()
