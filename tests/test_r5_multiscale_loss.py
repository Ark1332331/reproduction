"""R5 tests: the three intermediate decoder likelihood heads receive BCE supervision.

The final occupancy head l0 is excluded from the auxiliary likelihood loss: it is
supervised once by completion_loss, so giving it a second (differently weighted)
BCE would pull the same head toward two conflicting objectives.
"""
import unittest
import torch
from tests.r5_test_fixtures import make_structured_terrain_sample, prepare_sparse_sample
from models.r5_sparse_model import FourLevel4DCompletionModel
from losses.r5_sparse_loss import multiscale_likelihood_loss


class R5MultiscaleLossTests(unittest.TestCase):
    def test_averages_bce_for_three_intermediate_decoder_scales_and_backpropagates(self):
        data = prepare_sparse_sample(make_structured_terrain_sample())
        model = FourLevel4DCompletionModel()
        prediction = model(data.input_tensor)
        self.assertEqual(len(prediction.decoder_likelihoods), 3)
        loss = multiscale_likelihood_loss(prediction.decoder_likelihoods, data.target_batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.decoder_likelihood_heads.parameters()))


if __name__ == "__main__":
    unittest.main()
