#!/USSR/bin/python3.9
""" Test numerical derivative such as: gradient & hessian."""
import unittest

import numpy as np
from numpy import linalg as LA


class SampleFunctionAndDerivative:
    def __init__(self, matrix, center_position=None, *args, **kwargs):
        self.matrix = matrix
        self.dim = 2
        if center_position is None:
            self.center_position = np.zeros(self.dim)
        else:
            self.center_position = center_position

    def get_function_value(self, position):
        relative_position = position - self.center_position
        hull_value = LA.norm(relative_position) ** 4 - relative_position.T.dot(
            self.matrix
        ).dot(relative_position)
        return hull_value

    def evaluate_gradient(self, position):
        relative_position = position - self.center_position
        gradient = 4 * LA.norm(
            relative_position
        ) ** 2 * relative_position - 2 * self.matrix.dot(relative_position)
        return gradient


class TestNumericalDerivatives(unittest.TestCase):
    def test_numerical_gradient(self):
        from vartools.math import get_numerical_gradient

        obstacle = SampleFunctionAndDerivative(
            matrix=np.array([[10, 0], [0, -1]]), center_position=np.array([0, 3])
        )

        positions = np.array([[2, -3], [0, 0], [1, 1]]).T

        for it_pos in range(positions.shape[1]):
            position = positions[:, it_pos]

            gradient_analytic = obstacle.evaluate_gradient(position=position)
            gradient_numerical = get_numerical_gradient(
                position=position,
                function=obstacle.get_function_value,
                delta_magnitude=1e-6,
            )

            self.assertTrue(
                np.allclose(gradient_analytic, gradient_numerical, rtol=1e-5)
            )

    def test_numerical_hessian(self):
        pass


if (__name__) == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
