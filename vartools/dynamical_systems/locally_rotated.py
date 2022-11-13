"""
Dynamical Systems with a closed-form description.
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021

# Use python 3.10 [annotations / typematching]
from __future__ import annotations  # Not needed from python 3.10 onwards

import numpy as np
from numpy import linalg as LA

from ._base import DynamicalSystem
from vartools.directional_space import get_angle_space_inverse
from vartools.states import ObjectPose


class LocallyRotated(DynamicalSystem):
    """Returns dynamical system with a mean rotation of 'mean_rotation'
    at position 'rotation_center'

    Parameters
    ----------
    position: Position at which the dynamical system is evaluated
    attractor_position: Center of the dynamical system - rotation has to reach <pi/2
    mean_rotation: angle-space rotation at position.
    influence_radius:
    influence_descent_factor: float > 0

    Return
    ------
    Velocity (dynamical system) evaluted at the center position
    """

    def __init__(
        self,
        max_rotation: (np.ndarray, list),
        influence_pose: ObjectPose = None,
        influence_radius: float = 1,
        influence_axes_length: np.ndarray = None,
        # delta_influence_center: float = 0.1,
        influence_descent_factor: float = 1,
        attractor_influence_radius: float = 1.0,
        *args,
        **kargs,
    ):
        self.max_rotation = np.array(max_rotation)
        self.dimension = self.max_rotation.shape[0] + 1

        super().__init__(*args, **kargs)

        if influence_axes_length is None:
            self.influence_axes_length = np.ones(self.dimension) * influence_radius
        else:
            self.influence_axes_length = influence_axes_length
        self.influence_pose = influence_pose

        # Added weight to the rotation to ensure that @ center the
        # rotation stable (< pi/2)
        self.rotation_weight = np.linalg.norm(self.max_rotation) / (np.pi / 2)

        # Additional influence for center such that stirctly smallre than pi/2
        self.attractor_influence_radius = attractor_influence_radius
        self.influence_descent_factor = influence_descent_factor

    @property
    def influence_pose(self) -> ObjectPose:
        return self._influence_pose

    @influence_pose.setter
    def influence_pose(self, value: ObjectPose):
        """Ensure that the influence pose differs from the local center"""
        # TODO: special pose which can be placed at the center while ensuring stability
        if not LA.norm(value.position):
            raise ValueError("Influence pose cannot be placed at the center.")
        self._influence_pose = value

    def _get_scaled_distance(self, position):
        """Transform to ellipse frame"""
        if self.influence_pose is not None:
            position = self.influence_pose.transform_position_to_relative(position)
        position = position / self.influence_axes_length
        return np.linalg.norm(position)

    def evaluate(self, position: np.ndarray) -> np.ndarray:
        """Get the velocity of the locally rotated ds."""
        rel_position = self.get_relative_position_to_attractor(position)

        mag_pos = np.linalg.norm(rel_position)
        if not mag_pos:  # Zero velocity
            return np.zeros(position.shape)

        weight_rot = self._get_weight(position)

        if weight_rot > 0:
            # Angle space is not defined opposite (where weight is zero)
            rot_final = self.max_rotation * weight_rot

            # TODO: why (-1) ?! change?
            velocity = get_angle_space_inverse(
                dir_angle_space=(-1) * rot_final, null_direction=(-1) * rel_position
            )

        else:
            velocity = (-1) * rel_position

        if self.maximum_velocity is not None:
            mag_pos = min(mag_pos, self.maximum_velocity)
        velocity = velocity * mag_pos

        return velocity

    def _get_weight(self, position: np.ndarray) -> float:
        """Returns weight of local rotation based on RELATIVE to the ObjectPose position.
        Decreasing weight -> less rotated DS."""
        scaled_dist_ellipse = self._get_scaled_distance(position)

        if scaled_dist_ellipse >= 1 + self.influence_descent_factor:
            # No influence far-away
            return 0

        elif scaled_dist_ellipse <= 1:
            # Inner strong-influence core
            weight_rot = 1

        else:
            # scaled_dist_ellipse = 1-scaled_dist_ellipse
            weight_rot = scaled_dist_ellipse - 1
            weight_rot = 1 - (weight_rot / self.influence_descent_factor)

        # Evalaute center weight [inverse of 1]
        # rel_pos = self.get_relative_position_to_attractor(position)
        dist_center = np.linalg.norm(position)

        if dist_center <= 0:
            # dist_center = infinity -> give normalization
            return 0

        elif dist_center < self.attractor_influence_radius:
            # Projection of f: (0, 1] -> (infty -> 0]
            weight_center = self.attractor_influence_radius / dist_center - 1.0
            sum_weights = weight_center + weight_rot
            if sum_weights > 1:
                weight_rot = weight_rot / sum_weights

        # Additional [0-1] weight from dot-product [no influence behind]
        vec_attr_pose = (-1) * self.get_relative_position_to_attractor(
            self.influence_pose.position
        )
        dot_prod = np.dot(vec_attr_pose, position) / (
            LA.norm(vec_attr_pose) * LA.norm(position)
        )
        if dot_prod > 0:
            # in [0, 1]
            weight_rot = weight_rot * (1 - dot_prod)

        return weight_rot


class MultiLocalRotation(DynamicalSystem):
    """A collection of locally rotated DS"""

    def __init__(self, _dynamicsal_systems, *args, **kwargs):
        self._dynamicsal_systems = []

    def evaluate(self):
        self._dynamicsal_systems
        pass

    @property
    def n_systems(self):
        return len(self._dynamics_list)

    def add(self, dynamical_system):
        self._dynamics_list.append(dynamical_system)

    def delete(self, it):
        del self._dynamics_list[it]

    # def
