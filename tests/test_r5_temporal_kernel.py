"""Acceptance test: the 4D convolution must actually read across time.

The temporal kernel length is 2 (kernel (2,2,2,2), stride (2,2,2,1)): a k=0
voxel's output must depend on its k=1 neighbours. The test fixes the k=0 input,
changes only the k=1 layout, and requires the k=0 occupancy logits to change.
"""

import unittest

import torch
import MinkowskiEngine as ME

from models.r5_sparse_model import FourLevel4DCompletionModel


def _block_tensor(k1_z: int) -> tuple[torch.Tensor, torch.Tensor]:
    """k=0 plane at z=0 plus a k=1 plane at z=k1_z (3x3 blocks, batch 0)."""
    coords = []
    for z, k in ((0, 0), (k1_z, 1)):
        for x in (4, 5, 6):
            for y in (4, 5, 6):
                coords.append((0, x, y, z, k))
    features = torch.ones(len(coords), 3)
    return torch.tensor(coords, dtype=torch.int32), features


class TemporalKernelAcceptanceTests(unittest.TestCase):
    def test_k0_logits_change_when_only_k1_changes(self):
        model = FourLevel4DCompletionModel()
        model.eval()
        coords_a, feats_a = _block_tensor(k1_z=1)  # k=1 plane adjacent to k=0
        coords_b, feats_b = _block_tensor(k1_z=2)  # k=1 plane one cell farther
        with torch.no_grad():
            pred_a = model(ME.SparseTensor(features=feats_a, coordinates=coords_a))
            pred_b = model(ME.SparseTensor(features=feats_b, coordinates=coords_b))
        logits_a = dict(
            zip(
                [tuple(int(v) for v in row) for row in pred_a.candidates.C.tolist()],
                pred_a.occupancy_logits.F[:, 0].tolist(),
            )
        )
        logits_b = dict(
            zip(
                [tuple(int(v) for v in row) for row in pred_b.candidates.C.tolist()],
                pred_b.occupancy_logits.F[:, 0].tolist(),
            )
        )
        common = set(logits_a) & set(logits_b)
        self.assertGreaterEqual(len(common), 4, "the two outputs should share candidate voxels")
        differences = [abs(logits_a[key] - logits_b[key]) for key in common]
        self.assertGreater(
            max(differences), 1e-4,
            "k=0 occupancy logits must depend on k=1 neighbours (temporal kernel length 2)",
        )


if __name__ == "__main__":
    unittest.main()
