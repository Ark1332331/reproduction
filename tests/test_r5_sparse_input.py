"""R5 tests: bridge R1 voxel results into MinkowskiEngine's 4D sparse input."""

import unittest

import numpy as np
import torch

from representation.r1_data_representation import VoxelRepresentation
from models.r5_sparse_input import batch_voxel_representations, make_sparse_tensor


class R5SparseInputTests(unittest.TestCase):
    """Check the data contract between R1 and the real sparse-network backend."""

    def test_adds_batch_column_without_changing_r1_coordinates_or_features(self):
        r1_result = VoxelRepresentation(
            coords=np.array([[2, 2, 2, 0], [2, 2, 2, 1]], dtype=int),
            features=np.array([[0.4, 0.0, 0.0], [0.1, 0.2, 0.3]], dtype=float),
            dropped_count=0,
        )

        sparse_batch = batch_voxel_representations([r1_result])

        np.testing.assert_array_equal(
            sparse_batch.coordinates,
            np.array([[0, 2, 2, 2, 0], [0, 2, 2, 2, 1]], dtype=np.int32),
        )
        np.testing.assert_allclose(sparse_batch.features, r1_result.features)
        self.assertEqual(sparse_batch.coordinates.dtype, np.dtype(np.int32))
        self.assertEqual(sparse_batch.features.dtype, np.dtype(np.float32))

    def test_keeps_two_samples_separate_using_batch_numbers(self):
        first = VoxelRepresentation(
            coords=np.array([[1, 2, 3, 0]], dtype=int),
            features=np.array([[0.1, 0.2, 0.3]], dtype=float),
            dropped_count=0,
        )
        second = VoxelRepresentation(
            coords=np.array([[1, 2, 3, 0]], dtype=int),
            features=np.array([[0.7, 0.8, 0.9]], dtype=float),
            dropped_count=0,
        )

        sparse_batch = batch_voxel_representations([first, second])

        np.testing.assert_array_equal(
            sparse_batch.coordinates,
            np.array([[0, 1, 2, 3, 0], [1, 1, 2, 3, 0]], dtype=np.int32),
        )
        np.testing.assert_allclose(
            sparse_batch.features,
            np.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]], dtype=np.float32),
        )

    def test_rejects_duplicate_4d_coordinates_inside_one_sample(self):
        invalid_r1_result = VoxelRepresentation(
            coords=np.array([[1, 2, 3, 0], [1, 2, 3, 0]], dtype=int),
            features=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=float),
            dropped_count=0,
        )

        with self.assertRaisesRegex(ValueError, "unique"):
            batch_voxel_representations([invalid_r1_result])

    def test_creates_a_real_4d_minkowski_sparse_tensor(self):
        r1_result = VoxelRepresentation(
            coords=np.array([[2, 2, 2, 0], [3, 2, 2, 1]], dtype=int),
            features=np.array([[0.4, 0.0, 0.0], [0.1, 0.2, 0.3]], dtype=float),
            dropped_count=0,
        )

        sparse_tensor = make_sparse_tensor(batch_voxel_representations([r1_result]))

        self.assertEqual(tuple(sparse_tensor.C.shape), (2, 5))
        self.assertEqual(tuple(sparse_tensor.F.shape), (2, 3))
        self.assertEqual(sparse_tensor.C.dtype, torch.int32)
        self.assertEqual(sparse_tensor.F.dtype, torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "R5 GPU environment is required")
    def test_creates_a_cuda_sparse_tensor_from_the_same_r1_contract(self):
        r1_result = VoxelRepresentation(
            coords=np.array([[2, 2, 2, 0], [3, 2, 2, 1]], dtype=int),
            features=np.array([[0.4, 0.0, 0.0], [0.1, 0.2, 0.3]], dtype=float),
            dropped_count=0,
        )

        sparse_tensor = make_sparse_tensor(
            batch_voxel_representations([r1_result]),
            device="cuda",
        )

        self.assertTrue(sparse_tensor.F.is_cuda)
        self.assertEqual(tuple(sparse_tensor.C.shape), (2, 5))


if __name__ == "__main__":
    unittest.main()
