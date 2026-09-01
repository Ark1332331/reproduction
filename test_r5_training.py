"""R5 tests: a batched 3D sparse training loop should genuinely learn something."""

import unittest

import torch

from r5_terrain_dataset import make_structured_terrain_sample, prepare_sparse_dataset
from r5_training import evaluate_minimal_sparse_model, train_minimal_sparse_model


class R5TrainingTests(unittest.TestCase):
    """Use varied step heights and observation ranges, not repeated copies of one view."""

    def setUp(self):
        train_samples = [
            make_structured_terrain_sample(0.55, 0.6, 0.6, 1.1),
            make_structured_terrain_sample(0.65, 1.1, 0.6, 1.1),
            make_structured_terrain_sample(0.75, 0.6, 0.6, 1.1),
            make_structured_terrain_sample(0.85, 1.1, 0.6, 1.1),
        ]
        validation_samples = [
            make_structured_terrain_sample(0.6, 0.6, 0.6, 1.1),
            make_structured_terrain_sample(0.8, 1.1, 0.6, 1.1),
        ]
        self.training_data = prepare_sparse_dataset(train_samples)
        self.validation_data = prepare_sparse_dataset(validation_samples)

    def test_batches_multiple_3d_samples_without_mixing_their_coordinates(self):
        self.assertEqual(self.training_data.sample_count, 4)
        self.assertEqual(set(self.training_data.input_batch.coordinates[:, 0]), {0, 1, 2, 3})
        self.assertEqual(set(self.training_data.target_batch.coordinates[:, 0]), {0, 1, 2, 3})
        self.assertTrue((self.training_data.target_batch.coordinates[:, 4] == 0).all())

    def test_training_lowers_loss_on_the_varied_structured_terrain_set(self):
        run = train_minimal_sparse_model(self.training_data, steps=80, seed=0)

        self.assertLess(run.final_loss, run.initial_loss)
        self.assertLess(run.final_loss, 0.8 * run.initial_loss)

    def test_trained_model_produces_finite_validation_loss(self):
        run = train_minimal_sparse_model(self.training_data, steps=80, seed=0)

        validation_loss = evaluate_minimal_sparse_model(run.model, self.validation_data)

        self.assertTrue(torch.isfinite(validation_loss.total))
        self.assertGreater(validation_loss.positive_count, 0)


if __name__ == "__main__":
    unittest.main()
