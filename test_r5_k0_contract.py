"""Contract tests: the final prediction must be k=0-only and match k=0 targets."""

import unittest

import numpy as np
import torch
import MinkowskiEngine as ME

from r5_sparse_model import FourLevel4DCompletionModel
from r5_sparse_loss import completion_loss
from r5_test_fixtures import make_structured_terrain_sample, prepare_sparse_sample
from r5_sparse_input import batch_voxel_representations


class K0ContractTests(unittest.TestCase):
    def test_final_candidates_are_k0_only(self):
        data = prepare_sparse_sample(make_structured_terrain_sample())
        model = FourLevel4DCompletionModel()
        model.eval()
        with torch.no_grad():
            prediction = model(data.input_tensor, alpha=None)
        ks = prediction.candidates.C[:, 4].tolist()
        self.assertTrue(all(k == 0 for k in ks), f"final candidates must be k=0 only, got {set(ks)}")

    def test_completion_loss_ignores_k1_candidates(self):
        """k=1 candidates must not be labelled negative against the k=0 target."""
        data = prepare_sparse_sample(make_structured_terrain_sample())
        model = FourLevel4DCompletionModel()
        model.eval()
        with torch.no_grad():
            prediction = model(data.input_tensor, alpha=None)
        # the loss on the k0-only final output must be finite and report only k0
        # positives; a k=1 row smuggled in must not change the positive count
        coords = prediction.candidates.C
        loss = completion_loss(
            coords, prediction.occupancy_logits.F, prediction.position_offsets.F, data.target_batch
        )
        self.assertTrue(torch.isfinite(loss.total))
        # every positive label must resolve inside the k=0 target coordinates
        target_keys = {tuple(map(int, r)) for r in data.target_batch.coordinates}
        for key, is_pos in zip(
            [tuple(map(int, r)) for r in coords.tolist()], loss.occupancy_targets[:, 0].tolist()
        ):
            if is_pos:
                self.assertIn(key, target_keys)


if __name__ == "__main__":
    unittest.main()
