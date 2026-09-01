"""Contracts for the recoverable R7 IsaacLab collection launcher."""

import tempfile
import unittest
from pathlib import Path

from collect_r7_dataset import build_jobs, capture_summary, command_for, completed_capture
import json
import numpy as np


class R7CollectionLauncherTests(unittest.TestCase):
    def test_builds_one_stable_job_per_terrain_and_seed(self):
        jobs = build_jobs(Path("data"), Path("logs"), ("boxes", "walls"), seed_start=4, seed_count=2)
        self.assertEqual([(job.terrain, job.seed) for job in jobs], [("boxes", 4), ("walls", 4), ("boxes", 5), ("walls", 5)])
        self.assertEqual(jobs[0].output, Path("data/isaac_anymal_boxes_s4.npz"))

    def test_completed_capture_requires_a_nonempty_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.npz"
            self.assertFalse(completed_capture(path))
            path.touch()
            self.assertFalse(completed_capture(path))
            path.write_bytes(b"npz")
            self.assertFalse(completed_capture(path))
            np.savez_compressed(
                path,
                metadata_json=json.dumps({"coordinate_frame": "robot_centric_local_map", "trajectory_steps": 2}),
                measurement_00=np.zeros((1, 3)), target_00=np.zeros((1, 3)), translation_00=np.zeros(3), yaw_00=np.array(0.0),
                measurement_01=np.zeros((1, 3)), target_01=np.zeros((1, 3)), translation_01=np.zeros(3), yaw_01=np.array(0.0),
            )
            self.assertTrue(completed_capture(path))
            self.assertEqual(capture_summary(path)["captured_frames"], 2)

    def test_command_preserves_terrain_seed_and_output_contract(self):
        job = build_jobs(Path("data"), Path("logs"), ("poles",), seed_start=7, seed_count=1)[0]
        command = command_for(Path("/isaac"), Path("collector.py"), job, trajectory_steps=12, points_per_camera=1500)
        self.assertEqual(command[:6], ["conda", "run", "--no-capture-output", "-n", "isaaclab", "/isaac/isaaclab.sh"])
        self.assertEqual(command[6], "-p")
        self.assertTrue(command[7].endswith("/collector.py"))
        self.assertIn("poles", command)
        self.assertIn("7", command)
        self.assertTrue(command[-1].endswith("/data/isaac_anymal_poles_s7.npz"))


if __name__ == "__main__":
    unittest.main()
