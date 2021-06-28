"""
Dynamical Systems with a closed-form description.
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021

import numpy as np

from vartools.dynamicalsys import DynamicalSystem, allow_max_velocity
from vartools.directional_space import get_angle_space_inverse


class QuadraticAxisConvergence(DynamicalSystem):
    """ Dynamical system wich convergence faster towards x-axis. """
    def __init__(self, main_axis=None, conv_pow=2, stretching_factor=1,
                 center_position=None, maximum_velocity=None):
        super().__init__(center_position=center_position, maximum_velocity=maximum_velocity)

        self.conv_pow = conv_pow
        self.stretching_factor = stretching_factor
        
        if main_axis is not None:
            # TODO
            raise NotImplementedError()

    def evaluate(self, position, maximum_velocity=None):
        velocity = (-1)*stretching_factor*position
        velocity[1:] = np.copysign(velocity[1:]**conv_pow, velocity[1:])

        velocity = self.limit_velocity(velocity, maximum_velocity)
        
        return velocity

