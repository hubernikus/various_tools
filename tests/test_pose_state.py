#!/USSR/bin/python3.9
""" Sample DS spiral-trajectories."""
import unittest

from math import pi

from scipy.spatial.transform import Rotation # scipy rotation

import numpy as np
# import matplotlib.pyplot as plt

from vartools.state import ObjectPose

class TestObjectPose(unittest.TestCase):
    def test_null_pose(self):
        pose = ObjectPose()
        
        position = np.array([1, 0, 0])
        position_trafo = pose.transform_position_from_reference_to_local(position)
        self.assertTrue(np.allclose(position, position_trafo))

        position_trafo = pose.transform_position_from_local_to_reference(position)
        self.assertTrue(np.allclose(position, position_trafo))
        
        position_trafo = pose.transform_direction_from_reference_to_local(position)
        self.assertTrue(np.allclose(position, position_trafo))
        
        position_trafo = pose.transform_direction_from_local_to_reference(position)
        self.assertTrue(np.allclose(position, position_trafo))
        
    def test_position_bijection(self):
        pose = ObjectPose(position=[1, -1])
        position = np.array([2, 0])

        position_trafo = pose.transform_position_from_reference_to_local(position)
        position_repr = pose.transform_position_from_local_to_reference(position_trafo)
        
        self.assertTrue(np.allclose(position, position_repr))
        self.assertTrue(np.allclose(position_trafo, np.array([1, 1])))
        
        pose = ObjectPose(position=[1, -1], orientation=90.0/180*pi)
        position_trafo = pose.transform_position_from_reference_to_local(position)
        position_repr = pose.transform_position_from_local_to_reference(position_trafo)

        self.assertTrue(np.allclose(position, position_repr))
        self.assertTrue(np.allclose(position_trafo, np.array([-1, 1])))
        
        # TODO: for higher dimenions

    def test_orientation_bijection(self):
        pose = ObjectPose(position=[1, -1])
        direction = np.array([1, 1])

        direction_trafo = pose.transform_direction_from_reference_to_local(direction)
        direction_repr = pose.transform_direction_from_local_to_reference(direction_trafo)
        
        self.assertTrue(np.allclose(direction_trafo, direction_repr))
        self.assertTrue(np.allclose(direction_trafo, direction))
        
        pose = ObjectPose(position=[1, -1], orientation=90.0/180*pi)
        direction_trafo = pose.transform_direction_from_reference_to_local(direction)
        direction_repr = pose.transform_direction_from_local_to_reference(direction_trafo)

        self.assertTrue(np.allclose(direction, direction_repr))
        self.assertTrue(np.allclose(direction_trafo, np.array([-1, 1])))
        # TODO: for higher dimenions


if (__name__) == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    print("all done")
