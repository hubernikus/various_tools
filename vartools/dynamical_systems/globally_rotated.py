"""
Dynamical Systems with a closed-form description.
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021

from math import pi

import numpy as np
from numpy import linalg as LA

from ._base import DynamicalSystem
from vartools.directional_space import get_angle_space_inverse


class GloballyRotated(DynamicalSystem):
    """Returns dynamical system with a mean rotation of 'mean_rotation'
    at position 'rotation_center'

    Parameters
    ----------
    position: Position at which the dynamical system is evaluated
    mean_rotation: angle-space rotation at position.
    """

    def __init__(
        self, mean_rotation, center_position=None, maximum_velocity=None, dimension=2
    ):
        self.mean_rotation = np.array(mean_rotation)
        self.dimension = self.mean_rotation.shape[0] + 1

        super().__init__(
            center_position=center_position,
            maximum_velocity=maximum_velocity,
            dimension=self.dimension,
        )

    def evaluate(self, position):
        dir_attractor = self.attractor_position - position
        if not np.linalg.norm(dir_attractor):  # Zero velocity
            return np.zeros(position.shape)

        velocity = get_angle_space_inverse(
            dir_angle_space=rot_final, null_direction=dir_attractor
        )

        magnitude = np.linalg.norm(position - self.attractor_position)

        if self.maximum_velocity is not None:
            magnitude = min(magnitude, self.maximum_velocity)
        velocity = velocity * magnitude

        return velocity

    def is_stable(self):
        """Is the DS converging to an attractor. Only look at simlest problem."""
        return LA.norm(self.mean_rotation) < pi / 2
