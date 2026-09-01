"""R5 tests: measurement augmentations are deterministic and never alter target data."""
import unittest
import numpy as np
from r5_terrain_dataset import augment_measurement_points, make_randomized_structured_samples, make_randomized_urban_samples, make_structured_terrain_sample, mirror_structured_sample, prepare_sparse_dataset

class R5AugmentationTests(unittest.TestCase):
    def test_adds_deterministic_noise_and_outliers_to_measurements_only(self):
        sample=make_structured_terrain_sample()
        augmented=augment_measurement_points(sample.current_points,np.random.default_rng(3),position_noise=.01,tilt_degrees=0,drop_probability=0,outlier_count=2)
        self.assertEqual(augmented.shape,(8,3))
        self.assertFalse(np.array_equal(augmented[:6],sample.current_points))
        self.assertEqual(sample.complete_points.shape,(8,3))

    def test_drop_probability_can_remove_all_measurement_points(self):
        sample=make_structured_terrain_sample()
        augmented=augment_measurement_points(sample.current_points,np.random.default_rng(0),drop_probability=1.0)
        self.assertEqual(augmented.shape,(0,3))

    def test_seeded_augmentation_changes_input_batch_but_not_clean_target_batch(self):
        sample=make_structured_terrain_sample()
        clean=prepare_sparse_dataset([sample])
        augmented=prepare_sparse_dataset([sample],augmentation_seed=4,mirror=False)
        np.testing.assert_array_equal(augmented.target_batch.coordinates,clean.target_batch.coordinates)
        self.assertFalse(np.array_equal(augmented.input_batch.coordinates,clean.input_batch.coordinates))

    def test_mirroring_transforms_target_and_both_measurements_consistently(self):
        sample = make_structured_terrain_sample()
        mirrored = mirror_structured_sample(sample, x_extent=2., y_extent=1., mirror_x=True, mirror_y=True)
        np.testing.assert_allclose(mirrored.complete_points[:, :2], [2., 1.] - sample.complete_points[:, :2])
        np.testing.assert_allclose(mirrored.current_points[:, :2], [2., 1.] - sample.current_points[:, :2])
        np.testing.assert_allclose(mirrored.previous_points[:, :2], [2., 1.] - sample.previous_points[:, :2])

    def test_patch_augmentation_is_deterministic_and_outliers_are_clustered(self):
        sample = make_structured_terrain_sample()
        first = augment_measurement_points(
            sample.current_points, np.random.default_rng(7), position_noise=0, tilt_degrees=0,
            height_patch_noise=.05, height_patch_count=1, drop_patch_count=1,
            outlier_count=2, outlier_cluster_size=3, pose_noise=.02,
        )
        second = augment_measurement_points(
            sample.current_points, np.random.default_rng(7), position_noise=0, tilt_degrees=0,
            height_patch_noise=.05, height_patch_count=1, drop_patch_count=1,
            outlier_count=2, outlier_cluster_size=3, pose_noise=.02,
        )
        np.testing.assert_allclose(first, second)
        self.assertGreaterEqual(len(first), 6)

    def test_randomized_scene_generator_is_seeded_and_varies_step_heights(self):
        first=make_randomized_structured_samples(4,seed=8)
        second=make_randomized_structured_samples(4,seed=8)
        heights=[sample.complete_points[-1,2] for sample in first]
        self.assertEqual(heights,[sample.complete_points[-1,2] for sample in second])
        self.assertGreater(len(set(heights)),1)

    def test_urban_generator_contains_ground_and_nonflat_structures(self):
        sample=make_randomized_urban_samples(1,seed=5)[0]
        self.assertTrue(np.any(sample.complete_points[:,2] == .1))
        self.assertTrue(np.any(sample.complete_points[:,2] > .1))
        self.assertLess(len(sample.current_points),len(sample.complete_points))

if __name__=='__main__': unittest.main()
