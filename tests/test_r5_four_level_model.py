"""R5 tests: verify the paper-shaped four-level 4D U-Net structure."""
import unittest
import torch
from tests.r5_test_fixtures import make_structured_terrain_sample, prepare_sparse_sample
from models.r5_sparse_model import FourLevel4DCompletionModel
from losses.r5_sparse_loss import downsample_occupancy_target

class R5FourLevelModelTests(unittest.TestCase):
    def test_runs_four_spatial_scales_and_returns_current_prediction_heads(self):
        data = prepare_sparse_sample(make_structured_terrain_sample())
        model = FourLevel4DCompletionModel()
        prediction = model(data.input_tensor)
        self.assertEqual(prediction.candidates.D, 4)
        self.assertEqual(prediction.candidates.tensor_stride, [1, 1, 1, 1])
        self.assertGreater(len(prediction.candidates.C), len(data.input_tensor.C))
        self.assertEqual(prediction.occupancy_logits.F.shape[1], 1)
        self.assertTrue(torch.equal(prediction.candidates.C, prediction.occupancy_logits.C))
        self.assertEqual(prediction.position_offsets.F.shape[1], 3)
        # the final occupancy head l0 is excluded from the auxiliary likelihood
        # tuple (supervised once via completion_loss); l3/l2/l1 remain
        self.assertEqual(len(prediction.decoder_likelihoods), 3)
        self.assertEqual([item.tensor_stride for item in prediction.decoder_likelihoods], [[8, 8, 8, 1], [4, 4, 4, 1], [2, 2, 2, 1]])
        self.assertTrue(all(item.F.shape[1] == 1 for item in prediction.decoder_likelihoods))
        prediction.occupancy_logits.F.mean().backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_training_targets_guard_all_decoder_scales_against_aggressive_pruning(self):
        data = prepare_sparse_sample(make_structured_terrain_sample())
        targets = tuple(
            torch.from_numpy(downsample_occupancy_target(data.target_batch, factor).coordinates)
            for factor in (8, 4, 2, 1)
        )
        model = FourLevel4DCompletionModel()

        prediction = model(data.input_tensor, alpha=1.0, training_target_coordinates=targets)

        self.assertGreater(len(prediction.candidates.C), 0)
        self.assertEqual(prediction.candidates.tensor_stride, [1, 1, 1, 1])

if __name__ == '__main__':
    unittest.main()
