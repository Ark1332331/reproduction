"""R3-b4 tests: verify the minimal PyTorch completion-model interface."""

import unittest

import torch

from r3b_toy_model import ToyCompletionMLP


class R3BToyModelTests(unittest.TestCase):
    def test_maps_a_batch_of_input_grids_to_occupancy_score_grids(self):
        model = ToyCompletionMLP(width=8)
        input_grids = torch.zeros((3, 2, 2, 8), dtype=torch.float32)

        score_grids = model(input_grids)

        self.assertEqual(tuple(score_grids.shape), (3, 2, 8))
        self.assertTrue(torch.is_floating_point(score_grids))


if __name__ == "__main__":
    unittest.main()
