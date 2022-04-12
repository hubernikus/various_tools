""" Spiral shaped dynamical system in 3D. """
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# Created: 2021-06-28
# License: BSD (c) 2021

import numpy as np
from ._base import DynamicalSystem

# TODO: test! / visualize


class PendulumDynamics(DynamicalSystem):
    def __init__(
        self,
        length=1,
        weight=1,
        friction=1,
        maximum_velocity=None,
        dimension=2,
        **kwargs
    ):
        super().__init__(
            maximum_velocity=maximum_velocity, dimension=dimension, **kwargs
        )

        self.length = length
        self.weight = weight
        self.friction = friction

    def evaluate(self, position):

        velocity = np.zeros(self.dimension)
        velocity[0] = position[1]
        velocity[1] = (
            -self.length / self.weight * np.sin(position[0])
            - self.friction * self.length * position[1]
        )

        velocity = self.limit_velocity(velocity)
        return velocity


class DuffingOscillator(DynamicalSystem):
    def __init__(
        self,
        delta_factor=0.3,
        alpha_factor=-1.2,
        beta_factor=0.3,
        maximum_velocity=None,
        dimension=2,
        **kwargs
    ):
        super().__init__(
            maximum_velocity=maximum_velocity, dimension=dimension, **kwargs
        )
        # Assign values
        self.delta_factor = delta_factor
        self.alpha_factor = alpha_factor
        self.beta_factor = beta_factor

    def evaluate(self, position):

        velocity = np.zeros(self.dimension)
        velocity[0] = position[1]
        # velocity[1] = -self.delta_factor*((-self.alpha_factor/self.beta_factor)* position[0]
        # - position[0]**3 - position[1])
        velocity[1] = (-1) * (
            self.delta_factor * position[1]
            + self.alpha_factor * position[0]
            + self.beta_factor * position[0] ** 3
        )

        velocity = self.limit_velocity(velocity)
        return velocity


class BifurcationSpiral(DynamicalSystem):
    def __init__(self, maximum_velocity=None, dimension=2, **kwargs):
        super().__init__(
            maximum_velocity=maximum_velocity, dimension=dimension, **kwargs
        )

    def evaluate(self, position):
        velocity = np.zeros(self.dimension)
        velocity[0] = 2 * position[0] - position[0] * position[1]
        velocity[1] = 2 * position[0] ** 2 - position[1]

        velocity = self.limit_velocity(velocity)
        return velocity
