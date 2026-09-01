"""R3-b4/b5 tests: train the toy model and compare it with grid merging."""

import unittest

import numpy as np

from r3b_grid_evaluation import evaluate_occupancy_grid
from r3b_training import predict_occupancy_grids, train_completion_model
from r3b_training_dataset import make_structured_completion_dataset


class R3BTrainingTests(unittest.TestCase):
    def test_training_reduces_test_grid_errors_against_merge_baseline(self):
        dataset = make_structured_completion_dataset(width=8)

        model, losses = train_completion_model(
            train_inputs=dataset.train_inputs,
            train_targets=dataset.train_targets,
            width=8,
            epochs=1200,
            learning_rate=0.03,
            seed=0,
        )
        learned_grids = predict_occupancy_grids(model, dataset.test_inputs)
        baseline_grids = np.maximum(dataset.test_inputs[:, 0], dataset.test_inputs[:, 1])

        learned = evaluate_occupancy_grid(learned_grids, dataset.test_targets)
        baseline = evaluate_occupancy_grid(baseline_grids, dataset.test_targets)
        learned_blind_hits = sum(
            learned_grids[index, 1, blind_x] == 1
            for index, blind_x in enumerate(dataset.test_blind_xs)
        )

        self.assertLess(losses[-1], losses[0])
        self.assertEqual(baseline.missing_count, len(dataset.test_blind_xs))
        self.assertEqual(learned_blind_hits, len(dataset.test_blind_xs))
        self.assertLess(
            learned.missing_count + learned.extra_count,
            baseline.missing_count + baseline.extra_count,
        )


if __name__ == "__main__":
    unittest.main()
