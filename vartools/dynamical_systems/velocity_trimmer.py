"""
Velocity Trimmer For Dynamical Systems:
- Constant Velocity decreasing at attractor
- Maximum Velocity
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021

from abc import ABC, abstractmethod

import numpy as np
from numpy import linalg as LA


class SpuriousAttractorError(Exception):
    def __init__(self, position):
        self.position = position
        super().__init__()

    def __str__(self):
        return f"position={self.position} -> Spurious Attractor (velocity=0) detected."


class BaseTrimmer(ABC):
    """Virtual class which limits DS magnitudes."""

    # def __init__():
    # pass

    @abstractmethod
    def limit(self, velocity: np.ndarray, position: np.ndarray) -> np.ndarray:
        # Limit velocity to a set value
        pass


class ConstVelocityDecreasingAtAttractor(BaseTrimmer):
    """Returns scaled velocity which is only decreasing towards attractor."""

    def __init__(
        self,
        const_velocity: np.ndarray,
        distance_decrease: np.ndarray,
        attractor_position: np.ndarray = None,
    ):
        self.const_velocity = const_velocity
        self.distance_decrease = distance_decrease
        self.attractor_position = attractor_position

    def limit(self, velocity: np.ndarray, position: np.ndarray) -> np.ndarray:
        if self.attractor_position is None:
            dist_attr = LA.norm(position)
        else:
            dist_attr = LA.norm(position - self.attractor_position)

        if not dist_attr:
            return np.zeros(velocity.shape)

        mag_vel = LA.norm(velocity)
        if not mag_vel:
            raise SpuriousAttractorError(position)

        if dist_attr > self.distance_decrease:
            desired_velocity = self.const_velocity
        else:
            desired_velocity = self.const_velocity * (
                dist_attr / self.distance_decrease
            )
        return velocity / mag_vel * desired_velocity


class LimitMaximumVelocity(BaseTrimmer):
    def __init__(self, maximum_velocity: np.ndarray):
        self.maximum_velocity = maximum_velocity

    def limit(self, velocity: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Crops velocity at maximum.
        Note that 'position' position is unused!
        """
        mag_vel = LA.norm(velocity)
        if mag_vel > self.maximum_velocity:
            velocity = velocity / mag_vel * self.maximum_velocity
        return velocity
