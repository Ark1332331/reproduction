"""R5 tests: generated decoder locations must receive correctly aligned skips."""

import unittest

import torch
import MinkowskiEngine as ME

from models.r5_sparse_model import align_skip_to_candidates


class R5SkipAlignmentTests(unittest.TestCase):
    def test_keeps_seen_skip_values_and_zeros_new_generated_locations(self):
        skip = ME.SparseTensor(
            features=torch.tensor([[3.0, 4.0]]),
            coordinates=torch.tensor([[0, 2, 2, 2, 0]], dtype=torch.int32),
        )
        candidates = ME.SparseTensor(
            features=torch.zeros((2, 1)),
            coordinates=torch.tensor(
                [[0, 2, 2, 2, 0], [0, 3, 2, 2, 0]], dtype=torch.int32),
        )

        aligned = align_skip_to_candidates(skip, candidates)

        self.assertTrue(torch.equal(aligned.C, candidates.C))
        self.assertTrue(torch.equal(aligned.F[0], torch.tensor([3.0, 4.0])))
        self.assertTrue(torch.equal(aligned.F[1], torch.tensor([0.0, 0.0])))
        self.assertEqual(tuple(ME.cat(candidates, aligned).F.shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
