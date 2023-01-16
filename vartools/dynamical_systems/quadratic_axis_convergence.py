"""
Dynamical Systems with a closed-form description.
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021

import math

import numpy as np
import numpy.typing as npt

from typing import Optional

# from . import DynamicalSystem
from vartools.dynamical_systems import DynamicalSystem


class QuadraticAxisConvergence(DynamicalSystem):
    """Dynamical system wich convergence faster towards x-axis."""

    def __init__(
        self,
        attractor_position,
        main_axis: np.ndarray = None,
        conv_pow: float = 2.0,
        stretching_factor: float = 1.0,
        maximum_velocity: float = 1.0,
        dimension: int = 2,
    ):
        super().__init__(
            attractor_position=attractor_position,
            maximum_velocity=maximum_velocity,
            dimension=dimension,
        )

        self.conv_pow = conv_pow
        self.stretching_factor = stretching_factor

        if main_axis is not None:
            # TODO
            raise NotImplementedError()

    def evaluate(self, position):
        position = position - self.attractor_position
        velocity = (-1) * self.stretching_factor * position
        velocity[1:] = np.copysign(velocity[1:] ** self.conv_pow, velocity[1:])

        velocity = self.limit_velocity(velocity)
        return velocity


class AxesFollowingDynamics(DynamicalSystem):
    """
    center_position: The
    Since there is no end point to the dynamics, the maximum_velocity is actually the constant / desired velocity.
    """

    def __init__(
        self,
        main_direction: npt.ArrayLike,
        center_position: Optional[npt.ArrayLike],
        maximum_velocity: float = 1.0,
        perpendicular_scaling: float = 3.0,
    ):
        if not (axis_norm := np.linalg.norm(main_direction)):
            raise ValueError("Give a nonzero axis direction.")

        self.normalized_direction = np.array(main_direction) / axis_norm

        if center_position is None:
            self.center_position = np.zeros_like(self.normalized_direction)
        else:
            self.center_position = np.array(center_position)

        super().__init__(
            maximum_velocity=maximum_velocity,
            dimension=self.normalized_direction.shape[0],
        )
        self.perpendicular_scaling = perpendicular_scaling

    def evaluate(self, position):
        relative_position = self.center_position - position
        distance_along_axis = np.dot(relative_position, self.normalized_direction)

        vector_to_axis = (
            relative_position - self.normalized_direction * distance_along_axis
        )
        distance_to_axis = np.linalg.norm(vector_to_axis)

        if not distance_to_axis:
            return self.normalized_direction * self.maximum_velocity

        vector_to_axis = vector_to_axis / distance_to_axis

        speed_to_axis = (
            distance_to_axis / self.perpendicular_scaling * self.maximum_velocity
        )

        if speed_to_axis >= self.maximum_velocity:
            return vector_to_axis * self.maximum_velocity

        return speed_to_axis * vector_to_axis + self.normalized_direction * math.sqrt(
            self.maximum_velocity**2 - speed_to_axis**2
        )


def _test_quadratic_axes_following():
    global_ds = QuadraticAxisConvergence(
        attractor_position=np.array([2, 1]), maximum_velocity=1
    )

    plt.close("all")
    fig, ax = plot_dynamical_system_quiver(
        dynamical_system=global_ds,
        x_lim=[-15, 15],
        y_lim=[-15, 15],
        n_resolution=30,
    )
    ax.scatter(
        0,
        0,
        marker="*",
        s=200,
        color="black",
        zorder=5,
    )


def test_axes_following_dynamics(visualize=True):
    global_ds = AxesFollowingDynamics(
        center_position=np.array([0, 1]),
        maximum_velocity=1.0,
        main_direction=np.array([2, 0]),
    )

    if visualize:
        plt.close("all")
        fig, ax = plot_dynamical_system_quiver(
            dynamical_system=global_ds,
            x_lim=[-15, 15],
            y_lim=[-15, 15],
            n_resolution=30,
        )
        ax.scatter(
            0,
            0,
            marker="*",
            s=200,
            color="black",
            zorder=5,
        )

    position = np.array([1, 0])
    velocity = global_ds.evaluate(position)
    assert np.dot(global_ds.normalized_direction, velocity) > 0
    assert velocity[1] > 0

    position = np.array([0, 1])
    velocity = global_ds.evaluate(position)
    assert np.dot(global_ds.normalized_direction, velocity) > 0
    assert math.isclose(velocity[1], 0.0)


if (__name__) == "__main__":
    # Imported here to avoid circular conflict
    import matplotlib.pyplot as plt

    from vartools.dynamical_systems import plot_dynamical_system_quiver

    # _test_quadratic_axes_following()
    test_axes_following_dynamics()
