"""
Dynamical Systems with a closed-form description.
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021

import warnings
import numpy as np

from vartools.dynamicalsys import DynamicalSystem

class CircularStable(DynamicalSystem):
    """ Dynamical system with Circular Motion x-axis. """
    def __init__(self, radius, factor_controler=1, direction=1, main_axis=None,
                 center_position=None, maximum_velocity=None, dimension=2):
        super().__init__(center_position=center_position, maximum_velocity=maximum_velocity, dimension=dimension)

        self.radius = radius
        self.factor_controler = 1
        
        if abs(direction) != 1:
            warnings.warn("Direction not of magnitude 1. It is automatically reduced.")
            self.direction = np.copysign(1, self.direction)
        
        if main_axis is not None:
            # TODO
            raise NotImplementedError()

    def evaluate(self, position, maximum_velocity=None):
        if len(position.shape)!=1 or position.shape[0]!=2:
            raise ValueError("Position input allowed only of shape (2,)")

        position = position - self.center_position

        pos_norm = np.linalg.norm(position)
        if not pos_norm:
            # Saddle point at center
            return np.zeros(position.shape)

        velocity_linear =  self.radius - pos_norm
        if pos_norm < self.radius:
            velocity_linear = (1-1./velocity_linear)
        velocity_linear = position / pos_norm * velocity_linear

        velocity_circular = self.direction * np.array([-position[1], position[0]])
        velocity_circular = velocity_circular / np.linalg.norm(velocity_circular)

        velocity =  velocity_linear*self.factor_controler + velocity_circular
        
        radius = 1
        velocity = self.limit_velocity(velocity, maximum_velocity)
        return velocity

