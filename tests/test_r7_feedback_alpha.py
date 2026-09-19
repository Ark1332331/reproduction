"""P3-c test: output_alpha sweep must not change the history fed to later frames."""

import unittest
from pathlib import Path

import numpy as np
import torch

from models.r5_sparse_model import FourLevel4DCompletionModel
from rollout.r7_autoregressive_rollout import evaluate_detached_rollout, load_isaaclab_temporal_trajectory


class FeedbackAlphaDecouplingTests(unittest.TestCase):
    @staticmethod
    def _trajectory_path() -> str:
        return str(Path(__file__).resolve().parents[1] / "data" / "isaac_anymal_boxes_s10.npz")

    def test_output_alpha_sweep_keeps_second_frame_history_fixed(self):
        trajectory = load_isaaclab_temporal_trajectory(
            self._trajectory_path()
        )[:2]
        model = FourLevel4DCompletionModel()

        def count_history(alpha: float) -> int:
            counts = []

            def hook(module, args, kwargs):
                counts.append(int((args[0].C[:, 4] == 1).sum().item()))

            handle = model.register_forward_hook(hook)
            try:
                evaluate_detached_rollout(
                    model, trajectory, alpha=alpha, prune_internal=True, feedback_alpha=0.0
                )
            finally:
                handle.remove()
            # second frame's k=1 voxel count (history fed into the model)
            self.assertGreaterEqual(len(counts), 2, "evaluate should call the model per frame")
            return counts[1]

        c_a = count_history(0.1)
        c_b = count_history(0.3)
        self.assertEqual(c_a, c_b, "history must not change when only output_alpha changes")

    def test_history_ablation_feeds_no_k1_points_to_later_frames(self):
        trajectory = load_isaaclab_temporal_trajectory(
            self._trajectory_path()
        )[:2]
        model = FourLevel4DCompletionModel()
        history_counts = []

        def hook(module, args, kwargs):
            history_counts.append(int((args[0].C[:, 4] == 1).sum().item()))

        handle = model.register_forward_hook(hook)
        try:
            evaluate_detached_rollout(
                model, trajectory, alpha=0.1, prune_internal=False, feedback_alpha=0.0, use_history=False
            )
        finally:
            handle.remove()

        self.assertGreaterEqual(len(history_counts), 2)
        self.assertEqual(history_counts[1], 0)


if __name__ == "__main__":
    unittest.main()
