""" Spiral shaped dynamical system in 3D. """
# Author: Sthith
#         Lukas Huber*
# *Email: hubernikus@gmail.com
# Created: 2021-06-23
# License: BSD (c) 2021

import numpy as np
from ._base import DynamicalSystem


class CircularLinear(DynamicalSystem):
    """Return the velocity based on the evaluation of a spiral-shaped dynamical system."""

    def __init__(
        self,
        factor_circular=1.0,
        factor_linear=1.0,
        attractor_position=None,
        orientation=None,
        center_position=None,
        maximum_velocity=None,
        dimension=None,
    ):
        """
        Parameters
        ----------
        complexity_spiral : parameter on 'steepness' of spiral
        p_radius_control : P(ID)-controller to stay on the surface of the ellipse-spiral
        """
        super().__init__(
            center_position=center_position,
            maximum_velocity=maximum_velocity,
            dimension=dimension,
            attractor_position=attractor_position,
        )

        if orientation is not None:
            raise NotImplementedError(
                "TODO: Implement 3D rotation. (quaternion? / scipy?)"
            )

        self.factor_circular = factor_circular
        self.factor_linear = factor_linear

    def evaluate(self, position):
        position = position - self.center_position
        position = position - self.attractor_position

        velocity_circular = np.zeros(self.dimension)
        velocity_circular[0] = position[1]
        velocity_circular[1] = (-1) * position[0]
        velocity_linear = (-1) * position

        velocity = (
            velocity_circular * self.factor_circular
            + velocity_linear * self.factor_linear
        )
        velocity = self.limit_velocity(velocity)

        return velocity
