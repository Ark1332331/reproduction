"""Regression test for a fully dropped scan followed by outlier injection."""
import unittest

import numpy as np

from r5_terrain_dataset import augment_measurement_points


class AugmentationEdgeCaseTests(unittest.TestCase):
    def test_outliers_can_be_added_after_all_measurements_are_dropped(self):
        points = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        augmented = augment_measurement_points(
            points,
            np.random.default_rng(4),
            position_noise=0,
            tilt_degrees=0,
            drop_probability=1,
            outlier_count=2,
        )

        self.assertEqual(augmented.shape, (2, 3))
        self.assertTrue(np.isfinite(augmented).all())
