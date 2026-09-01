"""R5 contract test: simulator captures can cross into the training environment."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from r5_terrain_dataset import load_isaaclab_depth_sample


class IsaacLabCaptureContractTests(unittest.TestCase):
    def test_loads_three_world_aligned_point_clouds_from_npz(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.npz"
            np.savez_compressed(
                path,
                complete_points=np.array([[0., 0., 0.], [1., 0., .2]]),
                current_points=np.array([[0., 0., 0.]]),
                previous_points=np.array([[1., 0., .2]]),
                metadata_json='{"coordinate_frame": "world"}',
            )

            sample = load_isaaclab_depth_sample(str(path))

        self.assertEqual(sample.complete_points.shape, (2, 3))
        self.assertEqual(sample.current_points.shape, (1, 3))
        self.assertEqual(sample.previous_points.shape, (1, 3))

    def test_rejects_a_capture_missing_a_required_cloud(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incomplete.npz"
            np.savez_compressed(path, complete_points=np.zeros((1, 3)), current_points=np.zeros((1, 3)))

            with self.assertRaisesRegex(ValueError, "previous_points"):
                load_isaaclab_depth_sample(str(path))
