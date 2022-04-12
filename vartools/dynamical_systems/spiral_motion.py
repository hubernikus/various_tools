""" Spiral shaped dynamical system in 3D. """
# Author: Sthith
#         Lukas Huber*
# *Email: hubernikus@gmail.com
# Created: 2021-06-23
# License: BSD (c) 2021

import numpy as np
from ._base import DynamicalSystem

from vartools.states import ObjectPose


class SpiralStable(DynamicalSystem):
    """Return the velocity based on the evaluation of a spiral-shaped dynamical system."""

    def __init__(
        self,
        complexity_spiral: float = 3,
        p_radius_control: float = 1,
        # i_radius_control=0,  # Not implemented yet.
        axes_stretch=np.array([1, 1, 1]),
        radius: float = 1,
        pose: ObjectPose = None,
        maximum_velocity: np.ndarray = None,
        dimension: int = 3,
    ):
        """
        Parameters
        ----------
        complexity_spiral : parameter on 'steepness' of spiral
        p_radius_control : P(ID)-controller to stay on the surface of the ellipse-spiral
        """
        super().__init__(
            pose=pose, maximum_velocity=maximum_velocity, dimension=dimension
        )

        self.p_radius_control = p_radius_control
        self.complexity_spiral = complexity_spiral

        # TODO: (Properly) implement following tools
        self.axes_stretch = np.array([1, 1, 1])
        self.radius = 1

    def get_positions(self, n_points=None, tt=None):
        """Create initial position."""
        if tt is None:
            if n_points is None:
                raise ValueError("n_points & tt are both None.")
            tt = np.linspace(0, np.pi, n_points)
        else:
            # No lists, only numpy array
            tt = np.array(tt)
            n_points = tt.shape[0]

        dataset = np.zeros((n_points, self.dimension))

        dataset[:, 0] = np.sin(tt) * np.cos(self.complexity_spiral * tt)
        dataset[:, 1] = np.sin(tt) * np.sin(self.complexity_spiral * tt)
        dataset[:, 2] = np.cos(tt)

        # dataset = self.axes_strech*dataset + np.tile(self.center_position, (n_points, 1))
        return dataset

    def evaluate(self, position):
        velocity = np.zeros(position.shape)

        # Bound value in [-1, 1]
        position_norm = np.linalg.norm(position)
        if not position_norm:  # Zero value
            return velocity
        position = position / position_norm

        # z = cos(theta)
        # -dz/dt = sin(theta)
        theta = np.arccos(position[2])

        velocity[0] = (
            position[2] * np.cos(self.complexity_spiral * theta)
            - self.complexity_spiral * position[1]
        )
        velocity[1] = (
            position[2] * np.sin(self.complexity_spiral * theta)
            + self.complexity_spiral * position[0]
        )
        velocity[2] = -np.sqrt(1 - position[2] ** 2)

        if self.p_radius_control:  # Nonzero
            correction_velocity = position * (position_norm - self.radius)
            velocity = velocity - correction_velocity * self.p_radius_control

        velocity = velocity * self.axes_stretch
        velocity = self.limit_velocity(velocity)

        return velocity

    def check_convergence(self, position, convergence_dist=1e-6):
        vec_position_attr = self.get_relative_position_to_attractor(position)
        return np.linalg.norm(vec_position_attr) < convergence_dist
