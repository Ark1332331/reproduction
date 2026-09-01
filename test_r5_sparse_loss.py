"""R5 tests: supervise candidate voxels with current-frame ground truth."""

import math
import unittest

import numpy as np
import torch

from r5_sparse_input import SparseVoxelBatch
from r5_sparse_loss import completion_loss


class R5SparseLossTests(unittest.TestCase):
    """Check occupancy and sub-voxel losses against a hand-written target."""

    def setUp(self):
        self.candidate_coordinates = torch.tensor(
            [
                [0, 2, 2, 2, 0],
                [0, 3, 2, 2, 0],
                [0, 3, 2, 2, 1],
            ],
            dtype=torch.int32,
        )
        self.target = SparseVoxelBatch(
            coordinates=np.array(
                [[0, 2, 2, 2, 0], [0, 3, 2, 2, 0]],
                dtype=np.int32,
            ),
            features=np.array(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                dtype=np.float32,
            ),
        )

    def test_labels_current_target_voxels_but_not_same_spatial_history_voxel(self):
        result = completion_loss(
            candidate_coordinates=self.candidate_coordinates,
            occupancy_logits=torch.zeros((3, 1), requires_grad=True),
            position_offsets=torch.zeros((3, 3), requires_grad=True),
            target=self.target,
        )

        self.assertTrue(torch.equal(result.occupancy_targets.cpu(), torch.tensor([[1.0], [1.0], [0.0]])))
        self.assertEqual(result.positive_count, 2)

    def test_combines_bce_with_mean_distance_only_for_occupied_targets(self):
        occupancy_logits = torch.zeros((3, 1), requires_grad=True)
        position_offsets = torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.6, 0.5, 0.6],
                [9.0, 9.0, 9.0],
            ],
            requires_grad=True,
        )

        result = completion_loss(
            candidate_coordinates=self.candidate_coordinates,
            occupancy_logits=occupancy_logits,
            position_offsets=position_offsets,
            target=self.target,
            position_weight=2.0,
        )

        self.assertAlmostEqual(float(result.occupancy_loss.detach()), math.log(2.0), places=6)
        self.assertAlmostEqual(float(result.position_loss.detach()), 0.1, places=6)
        self.assertAlmostEqual(float(result.total.detach()), math.log(2.0) + 0.2, places=6)
        result.total.backward()
        self.assertIsNotNone(occupancy_logits.grad)
        self.assertIsNotNone(position_offsets.grad)

    def test_rejects_a_target_that_is_not_current_frame_ground_truth(self):
        history_target = SparseVoxelBatch(
            coordinates=np.array([[0, 2, 2, 2, 1]], dtype=np.int32),
            features=np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        )

        with self.assertRaisesRegex(ValueError, "k=0"):
            completion_loss(
                candidate_coordinates=self.candidate_coordinates,
                occupancy_logits=torch.zeros((3, 1)),
                position_offsets=torch.zeros((3, 3)),
                target=history_target,
            )


if __name__ == "__main__":
    unittest.main()
