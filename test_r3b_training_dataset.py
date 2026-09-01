"""R3-b1 tests: verify structured completion training samples."""

import unittest

import numpy as np

from r3b_training_dataset import (
    make_structured_completion_dataset,
    make_structured_completion_sample,
)


class R3BTrainingDatasetTests(unittest.TestCase):
    def test_builds_complete_platform_and_two_observation_channels(self):
        sample = make_structured_completion_sample(
            width=6,
            platform_start=2,
            platform_end=4,
            blind_x=3,
        )

        np.testing.assert_array_equal(
            sample.complete_grid,
            np.array(
                [
                    [1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 0],
                ],
                dtype=np.int8,
            ),
        )
        self.assertEqual(sample.current_grid.shape, (2, 6))
        self.assertEqual(sample.previous_grid.shape, (2, 6))
        self.assertEqual(sample.input_grid.shape, (2, 2, 6))
        np.testing.assert_array_equal(sample.input_grid[0], sample.current_grid)
        np.testing.assert_array_equal(sample.input_grid[1], sample.previous_grid)

    def test_hides_one_real_platform_cell_from_both_sources_but_keeps_neighbors(self):
        sample = make_structured_completion_sample(
            width=6,
            platform_start=2,
            platform_end=4,
            blind_x=3,
        )

        merged_observation = np.maximum(sample.current_grid, sample.previous_grid)

        self.assertEqual(sample.complete_grid[1, 3], 1)
        self.assertEqual(sample.current_grid[1, 3], 0)
        self.assertEqual(sample.previous_grid[1, 3], 0)
        self.assertEqual(merged_observation[1, 3], 0)
        self.assertEqual(sample.previous_grid[1, 2], 1)
        self.assertEqual(sample.previous_grid[1, 4], 1)

    def test_builds_fixed_shape_train_test_splits_with_real_blind_cells(self):
        dataset = make_structured_completion_dataset(width=8)

        self.assertGreater(len(dataset.train_inputs), 0)
        self.assertGreater(len(dataset.test_inputs), 0)
        self.assertEqual(dataset.train_inputs.shape[1:], (2, 2, 8))
        self.assertEqual(dataset.train_targets.shape[1:], (2, 8))
        self.assertEqual(dataset.test_inputs.shape[1:], (2, 2, 8))
        self.assertEqual(dataset.test_targets.shape[1:], (2, 8))
        self.assertEqual(dataset.test_blind_xs.shape, (len(dataset.test_inputs),))

        for input_grid, target_grid, blind_x in zip(
            dataset.test_inputs,
            dataset.test_targets,
            dataset.test_blind_xs,
        ):
            merged_grid = np.maximum(input_grid[0], input_grid[1])
            self.assertEqual(merged_grid[1, blind_x], 0)
            self.assertEqual(target_grid[1, blind_x], 1)


if __name__ == "__main__":
    unittest.main()
