"""
Dynamical Systems with a closed-form description.
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021

import numpy as np

from vartools.dynamicalsys import DynamicalSystem

class QuadraticAxisConvergence(DynamicalSystem):
    """ Dynamical system wich convergence faster towards x-axis. """
    def __init__(self, main_axis=None, conv_pow=2, stretching_factor=1,
                 center_position=None, maximum_velocity=None, dimension=2):
        super().__init__(center_position=center_position, maximum_velocity=maximum_velocity,
                         dimension=dimension)

        self.conv_pow = conv_pow
        self.stretching_factor = stretching_factor
        
        if main_axis is not None:
            # TODO
            raise NotImplementedError()

    def evaluate(self, position):
        position = position - self.center_position
        velocity = (-1)*self.stretching_factor*position
        velocity[1:] = np.copysign(velocity[1:]**self.conv_pow, velocity[1:])

        velocity = self.limit_velocity(velocity)
        return velocity

