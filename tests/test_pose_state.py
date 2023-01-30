#!/USSR/bin/python3.9
""" Sample DS spiral-trajectories."""
import unittest

from math import pi

from scipy.spatial.transform import Rotation  # scipy rotation

import numpy as np

# import matplotlib.pyplot as plt

from vartools.states import ObjectPose


class TestObjectPose(unittest.TestCase):
    def test_null_pose(self):
        dimension = np.zeros(3)
        pose = ObjectPose(position=dimension)

        position = np.array([1, 0, 0])
        # position_trafo = pose.transform_position_from_reference_to_local(position)
        position_trafo = pose.transform_position_from_relative(position)
        self.assertTrue(np.allclose(position, position_trafo))

        position_trafo = pose.transform_position_to_relative(position)
        self.assertTrue(np.allclose(position, position_trafo))

    def test_position_bijection(self):
        pose = ObjectPose(position=[1, -1])
        position = np.array([2, 0])

        position_trafo = pose.transform_position_to_relative(position)
        position_repr = pose.transform_position_from_relative(position_trafo)

        self.assertTrue(np.allclose(position, position_repr))
        self.assertTrue(np.allclose(position_trafo, np.array([1, 1])))

        pose = ObjectPose(position=[1, -1], orientation=90.0 / 180 * pi)
        position_trafo = pose.transform_position_to_relative(position)
        position_repr = pose.transform_position_from_relative(position_trafo)

        self.assertTrue(np.allclose(position, position_repr))
        self.assertTrue(np.allclose(position_trafo, np.array([1, -1])))

        # TODO: for higher dimenions

    def test_orientation_bijection(self):
        pose = ObjectPose(position=[1, -1])
        direction = np.array([1, 1])

        direction_trafo = pose.transform_direction_to_relative(direction)
        direction_repr = pose.transform_direction_from_relative(direction_trafo)

        self.assertTrue(np.allclose(direction_trafo, direction_repr))
        self.assertTrue(np.allclose(direction_trafo, direction))

        pose = ObjectPose(position=[1, -1], orientation=90.0 / 180 * pi)
        direction_trafo = pose.transform_direction_to_relative(direction)
        direction_repr = pose.transform_direction_from_relative(direction_trafo)

        self.assertTrue(np.allclose(direction, direction_repr))
        self.assertTrue(np.allclose(direction_trafo, np.array([1, -1])))
        # TODO: for higher dimenions


if (__name__) == "__main__":
    # unittest.main(argv=["first-arg-is-ignored"], exit=False)
    Tester = TestObjectPose()
    # Tester.test_null_pose()
    # Tester.test_position_bijection()
    Tester.test_orientation_bijection()

    print("all done")
