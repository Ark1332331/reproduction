"""R3-b2 tests: verify the non-learning occupancy-grid merge baseline."""

import unittest

import numpy as np

from r3b_grid_baseline import merge_occupancy_grids
from r3b_training_dataset import make_structured_completion_sample


class R3BGridBaselineTests(unittest.TestCase):
    def test_merge_keeps_a_blind_platform_cell_missing(self):
        sample = make_structured_completion_sample(
            width=6,
            platform_start=2,
            platform_end=4,
            blind_x=3,
        )

        merged_grid = merge_occupancy_grids(
            sample.current_grid,
            sample.previous_grid,
        )

        np.testing.assert_array_equal(
            merged_grid,
            np.array(
                [
                    [1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 0, 1, 0],
                ],
                dtype=np.int8,
            ),
        )
        self.assertEqual(merged_grid[1, 3], 0)
        self.assertEqual(sample.complete_grid[1, 3], 1)


if __name__ == "__main__":
    unittest.main()
