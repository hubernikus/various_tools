#!/USSR/bin/python3.9
""" Test the directional space. """
# Author: LukasHuber
# Date: 2021-05-18
# Email: lukas.huber@epfl.ch
# License: BSD (c) 2021

import unittest
import copy

from math import pi

import numpy as np
from numpy import linalg as LA

from vartools.linalg import get_orthogonal_basis

from vartools.directional_space import get_angle_space
from vartools.directional_space import get_angle_space_inverse
from vartools.directional_space import get_angle_from_vector, get_vector_from_angle

from vartools.directional_space import UnitDirection, DirectionBase

# from vartools.directional_space import get_directional_weighted_sum
from vartools.directional_space import get_directional_weighted_sum_from_unit_directions


class TestDirecionalSum(unittest.TestCase):
    def test_directional_weighting(self):
        dim = 3
        n_dir = 2

        null_direction = np.array([1, 0, 0])

        directions = np.zeros((dim, n_dir))
        directions[:, 0] = [0.0, 1, 0]
        directions[:, 1] = [0.0, -1, 0]

        base = DirectionBase(vector=null_direction)

        unit_directions = [
            UnitDirection(base).from_vector(directions[:, dd])
            for dd in range(directions.shape[1])
        ]

        weights = np.array([0.5, 0.5])

        # summed_dir = get_directional_weighted_sum(
        # null_direction=null_direction, directions=directions, weights=weights)

        summed_dir = get_directional_weighted_sum_from_unit_directions(
            base=base, unit_directions=unit_directions, weights=weights
        )

        self.assertTrue(np.allclose(summed_dir, np.array([1, 0, 0])))

    def test_decomposition_directional_weighted_sum(self):
        dim = 3
        n_dir = 2

        null_direction = np.array([-0.4, 0.4, 0.3])
        # null_direction = null_direction / LA.norm(null_direction)

        directions = np.zeros((dim, n_dir))
        directions[:, 0] = [0.1, 2, 0.2]
        directions[:, 1] = [0.3, -1, 0]
        # directions = directions / np.tile(LA.norm(directions, axis=0), (dim, 1))

        weights = np.array([0.3, 0.7])

        base = DirectionBase(vector=null_direction)

        unit_directions = [
            UnitDirection(base).from_vector(directions[:, dd])
            for dd in range(directions.shape[1])
        ]

        # summed_dir = get_directional_weighted_sum(
        # null_direction=null_direction, directions=directions, weights=weights)

        summed_dir = get_directional_weighted_sum_from_unit_directions(
            base=base, unit_directions=unit_directions, weights=weights
        )

        matr = np.hstack((null_direction.reshape(dim, 1), directions))

        components = LA.pinv(matr) @ summed_dir
        self.assertTrue(all(components >= 0))

    def test_execution_multiple_higher_dimension(self):
        """Higher dimension test without true-statement."""
        dim = 4
        n_dir = 5

        null_direction = np.array([-0.4, 0.4, 0.3, 0.9])
        # null_direction = null_direction / LA.norm(null_direction)

        directions = np.zeros((dim, n_dir))
        directions[:, 0] = [0.1, 2, 0.2, 0.1]
        directions[:, 1] = [0.3, -1, 0, -0.9]
        directions[:, 2] = [0.2, 4, -1, 3]
        directions[:, 3] = [1, 0, -2, 3.0]
        directions[:, 4] = [4, -1, 2, 8.0]
        # directions = directions / np.tile(LA.norm(directions, axis=0), (dim, 1))

        weights = np.array([0.3, 0.7, 0.1, -0.1, 0.1])

        base = DirectionBase(vector=null_direction)
        unit_directions = [
            UnitDirection(base).from_vector(directions[:, dd])
            for dd in range(directions.shape[1])
        ]

        # summed_dir = get_directional_weighted_sum(
        # null_direction=null_direction, directions=directions, weights=weights)
        summed_dir = get_directional_weighted_sum_from_unit_directions(
            base=base, unit_directions=unit_directions, weights=weights
        )

        matr = np.hstack((null_direction.reshape(dim, 1), directions))

        # Correct angle through previous implementation
        correct_angle = np.array([-0.15150807, -0.77723109, 0.58805302, 0.16477494])

        self.assertTrue(np.allclose(summed_dir, correct_angle))


if (__name__) == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)

    # user_test = True
    user_test = False
    if user_test:
        Tester = TestDirecionalSum()
        Tester.test_directional_weighting()
