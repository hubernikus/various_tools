"""
Dynamical System with Convergence towards attractor_position
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021

from typing import Optional
import math

import numpy as np
from numpy import linalg as LA

from vartools.states import Pose

from vartools.dynamics import DynamicalSystem


class WavyRotatedDynamics(DynamicalSystem):
    """Dynamical System in 2D."""

    # TODO: currently this shows weird behavior - maybe redo?

    def __init__(
        self,
        maximum_velocity: Optional[float] = None,
        distance_slowdown: float = 1.0,
        rotation_frequency: float = 1.0,
        max_rotation: float = math.pi * 0.25,
        rotation_power: float = 2,
        pose: Optional[Pose] = None,
        dimension: int = 2,
    ):
        super().__init__(dimension=dimension, pose=pose)

        self.maximum_velocity = maximum_velocity
        self.distance_slowdown = distance_slowdown

        if max_rotation >= math.pi * 0.5:
            raise ValueError(
                f"Out of bound input-value >= pi/2. Input: {max_rotation:.3f}"
            )
        self.max_rotation = max_rotation
        self.rotation_frequency = rotation_frequency
        self.rotation_power = rotation_power

    def evaluate(self, position):
        """Evaluate with included dynamics."""
        # TODO: move to base-dynamical-system-class
        if self.pose is not None:
            relative_position = self.pose.transform_position_to_relative(position)

        velocity = self.evaluate_relative(relative_position)

        if self.pose is not None:
            velocity = self.pose.transform_direction_from_relative(velocity)

        return velocity

    def evaluate_relative(self, position):
        """Input is the (relative) position."""

        # Simple stable system
        velocity = (-1) * position
        if not (velocity_magnitude := np.linalg.norm(velocity)):
            return velocity

        dist_attractor = np.linalg.norm(position)

        # Rotate velocity
        rotation_angle = (
            math.sin(dist_attractor ** self.rotation_power * self.rotation_frequency)
            * self.max_rotation
        )
        sin_ = math.sin(rotation_angle)
        cos_ = math.cos(rotation_angle)
        velocity = np.array([[cos_, sin_], [-sin_, cos_]]) @ velocity

        if self.maximum_velocity is None:
            return velocity

        if dist_attractor > self.distance_slowdown:
            return velocity / velocity_magnitude * self.maximum_velocity

        new_magnitude = self.maximum_velocity * (
            dist_attractor / self.distance_slowdown
        )
        return velocity / velocity_magnitude * new_magnitude
