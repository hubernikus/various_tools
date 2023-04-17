"""
Dynamical Systems with a closed-form description.
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy import linalg as LA

from vartools.states import ObjectPose


def allow_max_velocity(original_function=None):
    """Decorator to allow to limit the velocity to a maximum."""

    # Reintroduce (?)
    def wrapper(*args, max_vel=None, **kwargs):
        if max_vel is None:
            return original_function(*args, **kwargs)
        else:
            velocity = original_function(*args, **kwargs)

            mag_vel = np.linalg.norm(velocity)
            if mag_vel > max_vel:
                velocity = velocity / mag_vel
            return velocity

    return wrapper


class Dynamics(ABC):
    """Virtual Class for Base dynamical system"""

    def __init__(
        self,
        pose: Optional[ObjectPose] = None,
        maximum_velocity: Optional[float] = None,
        dimension: Optional[int] = None,
        attractor_position: Optional[np.ndarray] = None,
    ):
        if pose is not None:
            self.dimension = pose.position.shape[0]

        self.maximum_velocity = maximum_velocity

        if dimension is not None:
            self.dimension = dimension

        elif attractor_position is not None:
            self.dimension = attractor_position.shape[0]
            self.attractor_position = attractor_position

        elif not hasattr(self, "dimension"):
            raise ValueError(
                "Space dimension cannot be guess from inputs. "
                + "Please define it at initialization."
            )

        if pose is None:
            # Null pose
            self.pose = ObjectPose(position=np.zeros(self.dimension))
        else:
            self.pose: ObjectPose = pose

        self.attractor_position = attractor_position

    @property
    def attractor_position(self):
        return self._attractor_position

    @attractor_position.setter
    def attractor_position(self, value):
        self._attractor_position = value

    def limit_velocity(self, velocity, maximum_velocity=None):
        if maximum_velocity is None:
            if self.maximum_velocity is None:
                return velocity
            else:
                maximum_velocity = self.maximum_velocity

        mag_vel = LA.norm(velocity)

        if mag_vel > maximum_velocity:
            velocity = velocity / mag_vel * maximum_velocity
        return velocity

    @abstractmethod
    def evaluate(self, position: np.ndarray) -> np.ndarray:
        """Returns velocity of the evaluated the dynamical system at 'position'."""
        pass

    def get_relative_position_to_attractor(self, position: np.ndarray) -> np.ndarray:
        if self.attractor_position is None:
            return position
        else:
            return position - self.attractor_position

    def compute_dynamics(self, position: np.ndarray) -> np.ndarray:
        # This or 'evaluate' / to be or not to be?!
        # Could allow for additional cropping
        pass

    def evaluate_array(self, position_array: np.ndarray) -> np.ndarray:
        """Return an array of positions evaluated."""
        velocity_array = np.zeros(position_array.shape)
        for ii in range(position_array.shape[1]):
            velocity_array[:, ii] = self.evaluate_array(position_array[:, ii])
        return velocity_array

    def has_converged(self, position, convergence_margin=1e-1):
        if not hasattr(self, "attractor_position"):
            raise NotImplementedError("Convergence does not exist without attractor.")
        return LA.norm(position - self.attractor_position) < convergence_margin

    def check_convergence(self, *args, **kwargs):
        """Non compulsary function (only for stable systems), but needed to stop integration."""
        raise NotImplementedError("No convergence check implemented.")

    def motion_integration(
        self, start_position: np.array, dt: float, max_iteration: int = 10000
    ):
        """Integrate spiral Motion"""
        dataset = []
        dataset.append(start_position)
        current_position = start_position

        it_count = 0
        while True:
            it_count += 1
            if it_count > max_iteration:
                print("Maximum number of iterations reached.")
                break

            if self.check_convergence(dataset[-1]):
                print("Trajectory converged to goal..")
                break

            delta_vel = self.evaluate(dataset[-1])
            current_position = delta_vel * dt + current_position
            dataset.append(current_position)

        return np.array(dataset).T

class DynamicalSystem(Dynamics):
    # TODO: remove in the future
    pass
