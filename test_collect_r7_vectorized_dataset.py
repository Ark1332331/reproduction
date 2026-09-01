"""Contracts for the recoverable vectorized R7 collection launcher."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from collect_r7_vectorized_dataset import build_jobs, completed_job, job_outputs


class VectorizedCollectionLauncherTests(unittest.TestCase):
    def test_batches_have_non_overlapping_seed_ranges(self):
        jobs = build_jobs(Path("data"), Path("logs"), ("boxes",), seed_start=40, batch_count=2, num_envs=4)
        self.assertEqual([(job.terrain, job.base_seed) for job in jobs], [("boxes", 40), ("boxes", 44)])
        self.assertEqual(
            job_outputs(jobs[0], 4),
            tuple(Path(f"data/isaac_anymal_boxes_s{seed}_e{seed - 40}.npz") for seed in range(40, 44)),
        )

    def test_job_is_not_complete_until_all_parallel_outputs_meet_r7_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            job = build_jobs(data_dir, data_dir / "logs", ("walls",), seed_start=7, batch_count=1, num_envs=2)[0]
            for index, path in enumerate(job_outputs(job, 2)):
                np.savez_compressed(
                    path,
                    metadata_json='{"coordinate_frame": "robot_centric_local_map", "trajectory_steps": 2}',
                    measurement_00=np.zeros((1, 3)), target_00=np.zeros((1, 3)), translation_00=np.zeros(3), yaw_00=np.array(0.0),
                    measurement_01=np.zeros((1, 3)), target_01=np.zeros((1, 3)), translation_01=np.zeros(3), yaw_01=np.array(0.0),
                )
                if index == 0:
                    self.assertFalse(completed_job(job, 2))
            self.assertTrue(completed_job(job, 2))


if __name__ == "__main__":
    unittest.main()
