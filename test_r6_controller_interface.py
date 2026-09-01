"""R6 tests: sparse predictions become height values a locomotion policy can query."""
import unittest
import numpy as np
from r6_controller_interface import (
    decode_current_voxels,
    height_map_from_points,
    policy_terrain_window,
    query_height_rectangle,
)

class R6ControllerInterfaceTests(unittest.TestCase):
    def test_decodes_voxel_offsets_and_uses_highest_height_per_xy_cell(self):
        coords=np.array([[0,2,1,3,0],[0,2,1,4,0]],dtype=np.int32)
        points=decode_current_voxels(coords,np.array([[.5,0,.5],[0,0,0.]]),.5)
        np.testing.assert_allclose(points,[[1.25,.5,1.75],[1.,.5,2.]])
        height=height_map_from_points(points,4,4,.5)
        self.assertTrue(height.observed[2,1]); self.assertEqual(height.heights[2,1],2.)

    def test_returns_fixed_rectangular_window_and_marks_unknown_cells(self):
        height=height_map_from_points(np.array([[1.,1.,.4]]),4,4,.5)
        values,visible=query_height_rectangle(height,(2,2),3,3,unknown_height=-1.)
        self.assertEqual(values.shape,(3,3)); self.assertTrue(visible[1,1]); self.assertEqual(values[1,1],.4)
        self.assertFalse(visible[0,0]); self.assertEqual(values[0,0],-1.)

    def test_maps_current_robot_pose_into_latest_map_frame_for_paper_sized_policy_window(self):
        height = height_map_from_points(np.array([[1.05, .55, .4]]), 40, 40, .1)

        window = policy_terrain_window(
            height,
            robot_world_xy=(11.05, 20.55),
            map_origin_world_xy=(10., 20.),
        )

        self.assertEqual(window.center_cell, (10, 5))
        self.assertEqual(window.heights.shape, (16, 10))
        self.assertTrue(window.observed[8, 5])
        self.assertEqual(window.heights[8, 5], .4)

if __name__=='__main__': unittest.main()
