"""R5 tests: empty pruning must become a controlled experiment result, not a crash."""
import unittest
import torch
from models.r5_sparse_input import SparseVoxelBatch, make_sparse_tensor
from models.r5_sparse_model import EmptyPruningError, prune_candidates

class R5EmptyPruningTests(unittest.TestCase):
    def test_raises_controlled_error_before_framework_attempts_empty_tensor(self):
        x=make_sparse_tensor(SparseVoxelBatch(torch.tensor([[0,1,1,1,0]],dtype=torch.int32).numpy(),torch.zeros((1,3)).numpy()))
        with self.assertRaises(EmptyPruningError):
            prune_candidates(x,torch.tensor([[-10.0]]),alpha=.5)

if __name__=='__main__': unittest.main()
