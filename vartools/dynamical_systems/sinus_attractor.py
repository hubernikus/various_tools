"""
Dynamical System with Convergence towards attractor_position
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021

from typing import Optional

import numpy as np
from numpy import linalg as LA

from vartools.states import Pose

from ._base import DynamicalSystem
from .velocity_trimmer import BaseTrimmer


class SinusAttractorSystem(DynamicalSystem):
    """Dynamical System in 2D."""

    # TODO: currently this shows weird behavior - maybe redo?

    def __init__(
        self,
        attractor_position: np.ndarray = None,
        amplitude_y_max: float = 3,
        fade_factor: float = 2,
        stretch_fact_x: float = 1,
        dist_x_decline: float = 4,
        maximum_velocity: Optional[float] = None,
        distance_slowdown: float = 1.0,
        pose: Optional[Pose] = None,
    ):
        if pose is not None:
            # Only pose or attractor position can exist
            attractor_position = None
            dimension = pose.dimension
        else:
            dimension = None
        super().__init__(attractor_position=attractor_position, dimension=dimension)
        self.attractor_position = attractor_position

        self.maximum_velocity = maximum_velocity
        self.distance_slowdown = distance_slowdown

        self.fade_factor = fade_factor
        self.amplitude_y_max = amplitude_y_max
        self.dist_x_decline = dist_x_decline
        self.stretch_fact_x = stretch_fact_x

    def evaluate(self, position):
        """Evaluate with included dynamics."""
        # TODO: move to base-dynamical-system-class
        if self.pose is not None:
            relative_position = self.pose.transform_position_to_relative(position)
        elif self.attractor_position is not None:
            relative_position = position - self.attractor_position

        velocity = self.evaluate_relative(relative_position)

        if self.pose is not None:
            velocity = self.pose.transform_direction_from_relative(velocity)

        return velocity

    def evaluate_relative(self, position):
        """Input is the (relative) position."""
        velocity = self.compute_dynamics(position)
        # breakpoint()
        if self.maximum_velocity is None:
            return velocity

        # velocity = self.trimmer.limit(position=position, velocity=velocity)
        dist_attractor = np.linalg.norm(position)
        if not (velocity_magnitude := np.linalg.norm(velocity)):
            return velocity

        if dist_attractor > self.distance_slowdown:
            return velocity / velocity_magnitude * self.maximum_velocity

        new_magnitude = self.maximum_velocity * (
            dist_attractor / self.distance_slowdown
        )
        # breakpoint()
        return velocity / velocity_magnitude * new_magnitude

    def compute_dynamics(self, relative_position: np.ndarray) -> np.ndarray:
        """Sinus wave mixed with linear system to have converging decrease towards 0."""
        x_abs = abs(relative_position[0])
        if x_abs > self.dist_x_decline:
            amplitude = self.amplitude_y_max
        else:
            amplitude = x_abs / self.dist_x_decline * self.amplitude_y_max

        y_abs = abs(relative_position[1])
        if y_abs > self.fade_factor * amplitude:
            return (-1) * relative_position

        else:
            velocity = np.array([1, amplitude * np.cos(x_abs * self.stretch_fact_x)])
            if relative_position[0] > 0:
                velocity = (-1) * velocity

            if y_abs > amplitude:
                fade_fac = (y_abs - amplitude) / ((self.fade_factor - 1) * amplitude)

                velocity_linear = relative_position

                # Normalize to have equal weight
                velocity = velocity / LA.norm(velocity)
                velocity_linear = velocity_linear / LA.norm(velocity_linear)
                velocity = fade_fac * velocity_linear + (1 - fade_fac) * velocity
            # breakpoint()
        return velocity
