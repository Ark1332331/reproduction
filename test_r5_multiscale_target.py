"""R5 tests: each decoder scale needs a coarsened occupancy target."""
import unittest
import numpy as np
from r5_sparse_input import SparseVoxelBatch
from r5_sparse_loss import downsample_occupancy_target

class R5MultiscaleTargetTests(unittest.TestCase):
    def test_snaps_only_space_to_the_decoder_stride_grid_and_merges_children(self):
        target=SparseVoxelBatch(np.array([[0,2,3,4,0],[0,3,3,5,0],[1,2,3,4,0]],dtype=np.int32),np.zeros((3,3),dtype=np.float32))
        coarse=downsample_occupancy_target(target,2)
        np.testing.assert_array_equal(coarse.coordinates,np.array([[0,2,2,4,0],[1,2,2,4,0]],dtype=np.int32))
        self.assertEqual(coarse.features.shape,(2,3))

    def test_keeps_minkowski_coordinate_units_at_large_stride(self):
        target=SparseVoxelBatch(
            np.array([[0,17,9,3,0]],dtype=np.int32),
            np.zeros((1,3),dtype=np.float32),
        )
        coarse=downsample_occupancy_target(target,8)
        np.testing.assert_array_equal(
            coarse.coordinates,
            np.array([[0,16,8,0,0]],dtype=np.int32),
        )

if __name__=='__main__':
    unittest.main()
