"""R3-b5 tests: evaluate a binary occupancy-grid estimate against its target."""

import unittest

import numpy as np

from r3b_grid_evaluation import evaluate_occupancy_grid


class R3BGridEvaluationTests(unittest.TestCase):
    def test_reports_missing_extra_coverage_and_precision(self):
        estimate = np.array(
            [
                [1, 1, 0, 0],
                [0, 1, 1, 0],
            ],
            dtype=np.int8,
        )
        target = np.array(
            [
                [1, 1, 1, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.int8,
        )

        evaluation = evaluate_occupancy_grid(estimate, target)

        self.assertEqual(evaluation.missing_count, 1)
        self.assertEqual(evaluation.extra_count, 1)
        self.assertEqual(evaluation.coverage, 0.75)
        self.assertEqual(evaluation.precision, 0.75)


if __name__ == "__main__":
    unittest.main()
