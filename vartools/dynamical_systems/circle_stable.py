"""
Dynamical Systems with a closed-form description.
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021

import warnings
import numpy as np
from typing import Optional

# Absolute import (currently) allows to define tests
from vartools.dynamical_systems import DynamicalSystem

from vartools.states import ObjectPose


class CircularStable(DynamicalSystem):
    """Dynamical system with Circular Motion x-axis."""

    def __init__(
        self,
        radius,
        factor_controler: float = 1,
        direction: int = 1,
        # main_axis=None,
        pose: Optional[ObjectPose] = None,
        maximum_velocity: float = None,
        dimension: int = 2,
    ):
        if pose is None:
            center_pose = ObjectPose(position=np.zeros(dimension))
        else:
            center_pose = pose
            dimension = pose.position.shape[0]

        super().__init__(
            pose=center_pose,
            maximum_velocity=maximum_velocity,
            dimension=dimension,
        )

        self.radius = radius
        self.factor_controler = 1

        self.direction = np.copysign(1, direction)

        # if main_axis is not None:
        #     # TODO
        #     raise NotImplementedError()

    def evaluate(self, position, maximum_velocity=None):
        if len(position.shape) != 1 or position.shape[0] != 2:
            raise ValueError("Position input allowed only of shape (2,)")

        # position = position - self.center_position
        position = self.pose.transform_position_to_relative(position)

        pos_norm = np.linalg.norm(position)
        if not pos_norm:
            # Saddle point at center
            return np.zeros(position.shape)

        if pos_norm < self.radius:
            velocity_norm = self.radius / (pos_norm) - 1
        else:
            velocity_norm = self.radius - pos_norm

        velocity_linear = position / pos_norm * velocity_norm

        velocity_circular = self.direction * np.array([-position[1], position[0]])
        velocity_circular = velocity_circular / np.linalg.norm(velocity_circular)

        velocity = velocity_linear * self.factor_controler + velocity_circular

        radius = 1
        # Exactly put it to 'maximum velocity' - this makes it not constant at 0..
        # velocity = self.limit_velocity(velocity, maximum_velocity)
        velocity = velocity / np.linalg.norm(velocity) * self.maximum_velocity

        velocity = self.pose.transform_direction_from_relative(velocity)

        return velocity


def test_circular_dynamics(visualize=False):

    circular_ds = CircularStable(radius=1.0, maximum_velocity=2.0)

    if visualize:
        x_lim = [-2, 2]
        y_lim = [-2, 2]

        figsize = (8, 6)

        fig, ax = plt.subplots(figsize=figsize)
        plot_dynamical_system(
            dynamical_system=circular_ds,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
        )

    # No velocity at center
    position = np.zeros(2)
    velocity = circular_ds.evaluate(position)
    assert np.allclose(velocity, np.zeros(2))

    # Pointing away far away (and additionally slight rotation; always)
    position = np.array([1e-3, 0])
    velocity = circular_ds.evaluate(position)
    assert velocity[1] > 0


if (__name__) == "__main__":
    # Import visualization libraries here
    import matplotlib.pyplot as plt  # For debugging only (!)
    from vartools.dynamical_systems import plot_dynamical_system

    # test_circular_dynamics(visualize=False)
    test_circular_dynamics(visualize=False)
