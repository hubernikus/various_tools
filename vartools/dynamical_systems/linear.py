"""
Dynamical Systems with a closed-form description.
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021

import numpy as np
from numpy import linalg as LA

from ._base import DynamicalSystem


class LinearSystem(DynamicalSystem):
    """
    Linear Dyanmical system of the form
    dot[x] = A @ (x - center_position) or
    dot[x] = A @ x + b

    Parameters
    ----------
    position: Position at which the dynamical system is evaluated
    center_position: Center of the dynamical system - rotation has to reach <pi/2 at this position
    b:

    Return
    ------
    Velocity (dynamical system) evaluted at the center position
    """

    def __init__(
        self,
        attractor_position=None,
        A_matrix=None,
        b=None,
        dimension=None,
        maximum_velocity: float = None,
        distance_decrease: float = 1,
    ):

        if attractor_position is None:
            if A_matrix is not None:
                dimension = A_matrix.shape[0]
            if dimension is None:
                raise ValueError(
                    "Please indicate dimension explicietly if not using an attractor."
                )
            attractor_position = np.zeros(dimension)

        super().__init__(
            attractor_position=attractor_position,
            dimension=dimension,
            maximum_velocity=maximum_velocity,
        )
        self.distance_decrease = distance_decrease

        if A_matrix is None:
            self.A_matrix = np.eye(self.dimension) * (-1)
        else:
            self.A_matrix = A_matrix

        if b is not None:
            if attractor_position is not None:
                raise ValueError(
                    "center_pos AND baseline default arguments has been used."
                    + "Only one of them possible."
                )
            self.attractor_position = np.linalg.pinv(self.A_matrix) @ b

    def limit_velocity_around_attractor(self, velocity, position):
        dist_attr = LA.norm(position - self.attractor_position)

        if not dist_attr:
            return np.zeros(velocity.shape)

        mag_vel = LA.norm(velocity)
        if not mag_vel:
            return velocity

        if dist_attr > self.distance_decrease:
            desired_velocity = self.maximum_velocity
        else:
            desired_velocity = self.maximum_velocity * (
                dist_attr / self.distance_decrease
            )
        return velocity / mag_vel * desired_velocity

    def evaluate(self, position, max_vel=None):
        velocity = self.A_matrix.dot(position - self.attractor_position)

        if self.maximum_velocity is not None:
            velocity = self.limit_velocity_around_attractor(velocity, position)
        # velocity = self.limit_velocity(velocity, max_vel)
        return velocity

    def is_stable(self):
        """Check stability of given A matrix."""
        A = self.A_matrix + self.A_matrix.T
        eigvals, eigvecs = np.linalg.eig(A)

        if all(eigvals < 0):
            return True
        else:
            return False


class ConstantValue(DynamicalSystem):
    """Returns constant velocity based on the DynamicalSystem parent-class"""

    def __init__(self, velocity):
        self.constant_velocity = velocity

    def evaluate(self, *args, **kwargs):
        """Random input arguments, but always ouptuts same vector-field"""
        return self.constant_velocity
