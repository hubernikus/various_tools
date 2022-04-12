"""
Dynamical System with Convergence towards attractor_position
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021

import numpy as np
from numpy import linalg as LA

from ._base import DynamicalSystem
from .velocity_trimmer import BaseTrimmer


class SinusAttractorSystem(DynamicalSystem):
    """Dynamical System in 2D."""

    def __init__(
        self,
        attractor_position: np.ndarray = None,
        amplitude_y_max: float = 3,
        fade_factor: float = 2,
        stretch_fact_x: float = 1,
        dist_x_decline: float = 4,
        trimmer: BaseTrimmer = None,
    ):
        super().__init__(attractor_position=attractor_position)
        self.attractor_position = attractor_position

        self.trimmer = trimmer

        self.fade_factor = fade_factor
        self.amplitude_y_max = amplitude_y_max
        self.dist_x_decline = dist_x_decline
        self.stretch_fact_x = stretch_fact_x

    def evaluate(self, position):
        """Evaluate with included dynamics."""
        # TODO: move to base-dynamical-system-class
        velocity = self.compute_dynamics(position)

        if self.trimmer is not None:
            velocity = self.trimmer.limit(position=position, velocity=velocity)
        return velocity

    def compute_dynamics(self, position):
        """Sinus wave mixed with linear system to have converging decrease towards 0."""
        x_abs = abs(position[0])
        if x_abs > self.dist_x_decline:
            amplitude = self.amplitude_y_max
        else:
            amplitude = x_abs / self.dist_x_decline * self.amplitude_y_max

        y_abs = abs(position[1])
        if y_abs > self.fade_factor * amplitude:
            velocity = self.attractor_position - position
        else:
            velocity = np.array([1, amplitude * np.cos(x_abs * self.stretch_fact_x)])
            if position[0] > 0:
                velocity = (-1) * velocity

            if y_abs > amplitude:
                fade_fac = (y_abs - amplitude) / ((self.fade_factor - 1) * amplitude)

                velocity_linear = self.attractor_position - position

                # Normalize to have equal weight
                velocity = velocity / LA.norm(velocity)
                velocity_linear = velocity_linear / LA.norm(velocity_linear)
                velocity = fade_fac * velocity_linear + (1 - fade_fac) * velocity

        return velocity
