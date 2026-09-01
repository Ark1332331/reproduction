"""Contract tests for r6.isaac_height_scan against the verified ANYmal-C rough convention.

The expectations here were verified empirically against Isaac Lab
``Isaac-Velocity-Rough-Anymal-C-Direct-v0`` on 2026-08-21: 187 rays (17 x 11),
values (base_z - hit_z - 0.5).clip(-1, 1), flat ground ~= +0.1, row-major
iy * 17 + ix flattening with x forward and y left.
"""

import unittest

import numpy as np

from reproduction.r6_controller_interface import HeightMap, isaac_height_scan


class TestIsaacHeightScan(unittest.TestCase):
    def _flat_map(self, z: float = 0.0) -> HeightMap:
        # 40 x 40 cells of 0.1 m covers the 1.6 m x 1.0 m window at any yaw
        heights = np.full((40, 40), z, dtype=float)
        return HeightMap(heights=heights, observed=np.ones_like(heights, dtype=bool))

    def test_shape_is_187(self):
        out = isaac_height_scan(self._flat_map(), (2.0, 2.0), base_z=0.6, yaw=0.0)
        self.assertEqual(out.shape, (187,))

    def test_flat_ground_value(self):
        out = isaac_height_scan(self._flat_map(0.0), (2.0, 2.0), base_z=0.6, yaw=0.0)
        # 0.6 - 0.0 - 0.5 = 0.1, matches the value observed in the real env
        self.assertTrue(np.allclose(out, 0.1))

    def test_step_up_under_rear_ray_lowers_value(self):
        # 1.2 m step under the rearmost forward-ray (x = -0.8, yaw = 0):
        # that ray reads 0.6 - 1.2 - 0.5 = -1.1 -> clipped to -1.0
        heights = np.full((40, 40), 0.0, dtype=float)
        # 1.2 / 0.1 == 11.999... in float, so cover both candidate cells
        heights[11:13, :] = 1.2
        hmap = HeightMap(heights=heights, observed=np.ones_like(heights, dtype=bool))
        out = isaac_height_scan(hmap, (2.0, 2.0), base_z=0.6, yaw=0.0)
        rear_col = out[0 * 17 + 0]
        self.assertAlmostEqual(rear_col, -1.0)
        # front column still flat
        self.assertAlmostEqual(out[0 * 17 + 16], 0.1)

    def test_yaw_rotation_moves_sampling_axes(self):
        # with yaw = pi/2 the forward axis now points along map +y
        heights = np.full((40, 40), 0.0, dtype=float)
        # put a step at map x = 2.0, y = 2.0 - 0.8 (rear after rotation); cover the
        # neighboring cells because cos(pi/2) is not exactly zero and 1.2 / 0.1 is 11.999...
        heights[19:21, 11:13] = 1.0
        hmap = HeightMap(heights=heights, observed=np.ones_like(heights, dtype=bool))
        out = isaac_height_scan(hmap, (2.0, 2.0), base_z=0.6, yaw=np.pi / 2)
        # after +pi/2 rotation the forward axis points along world -y; the rear-center
        # ray (ix=0, iy=5) lands on world (2.0, 1.2)
        rear_col = out[5 * 17 + 0]
        # 0.6 - 1.0 - 0.5 = -0.9
        self.assertAlmostEqual(rear_col, -0.9)

    def test_flattening_row_major_y_outer(self):
        heights = np.full((40, 40), 0.0, dtype=float)
        # step under the right-most lateral ray (y = -0.5, yaw = 0); 1.5 / 0.1 is 14.999...
        heights[19:21, 14:16] = 1.0
        hmap = HeightMap(heights=heights, observed=np.ones_like(heights, dtype=bool))
        out = isaac_height_scan(hmap, (2.0, 2.0), base_z=0.6, yaw=0.0)
        # iy = 0 is y = -0.5 (right), any ix in that row is affected
        self.assertAlmostEqual(out[0 * 17 + 8], -0.9)
        self.assertAlmostEqual(out[10 * 17 + 8], 0.1)

    def test_unobserved_cells_read_zero(self):
        heights = np.full((40, 40), 0.0, dtype=float)
        observed = np.zeros_like(heights, dtype=bool)
        hmap = HeightMap(heights=heights, observed=observed)
        out = isaac_height_scan(hmap, (2.0, 2.0), base_z=0.6, yaw=0.0)
        self.assertTrue(np.all(out == 0.0))


if __name__ == "__main__":
    unittest.main()
