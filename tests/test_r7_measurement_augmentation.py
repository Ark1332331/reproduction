"""Contracts for paper-style measurement corruption."""

import unittest

import numpy as np

from training.r7_measurement_augmentation import augment_measurement_points


class MeasurementAugmentationTests(unittest.TestCase):
    def test_augmentation_is_seeded_and_preserves_point_shape(self):
        points = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=float)
        first = augment_measurement_points(
            points,
            np.random.default_rng(7),
            position_noise=0,
            tilt_degrees=0,
            outlier_count=2,
            outlier_cluster_size=3,
        )
        second = augment_measurement_points(
            points,
            np.random.default_rng(7),
            position_noise=0,
            tilt_degrees=0,
            outlier_count=2,
            outlier_cluster_size=3,
        )
        np.testing.assert_allclose(first, second)
        self.assertEqual(first.shape, (8, 3))

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


if __name__ == "__main__":
    unittest.main()
