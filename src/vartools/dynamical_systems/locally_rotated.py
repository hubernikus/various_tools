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

# TODO: move the 'from_ellipse'-subclass to a the dynamical_obstacle_avoidance repo to avoid reverse dependency
from dynamic_obstacle_avoidance.obstacles import Ellipse

class LocallyRotated(DynamicalSystem):
    """ Returns dynamical system with a mean rotation of 'mean_rotation'
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
    def __init__(self, max_rotation: (np.ndarray, list),
                 influence_pose: pose = None,
                 influence_radius: float = 1,
                 influence_axes_length: np.ndarray = None,
                 # delta_influence_center: float = 0.1,
                 influence_descent_factor: float = 1,
                 attractor_influence_radius: float = 1.0,
                 *args, **kargs):
        # TODO: change 'Obstacle' to general 'State' (or similar) & make
        # 'Obstacle' a subclass of the former
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
        self.rotation_weight = np.linalg.norm(self.max_rotation) / (np.pi/2)
        
        # Additional influence for center such that stirctly smallre than pi/2
        self.attractor_influence_radius = attractor_influence_radius
        # self.delta_influence_center = delta_influence_center
        self.influence_descent_factor = influence_descent_factor

    def from_ellipse(self, ellipse: Ellipse) -> LocallyRotated:
        self.influence_pose = ellipse.pose
        self.influence_axes_length = ellipse.axes_length
        return self

    def get_scaled_dist_to_ellipse(self, position):
        """ Transform to ellipse frame"""
        if self.influence_pose is not None:
            position = (
                self.influence_pose.transform_position_from_reference_to_local(
                position)
                )
        position = position / self.influence_axes_length
        return np.linalg.norm(position)
        
    def evaluate(self, position: np.ndarray) -> np.ndarray:
        """ Get the velocity of the locally rotated ds."""
        rel_position = self.get_relative_position_to_attractor(position)
        
        mag_pos = np.linalg.norm(rel_position)
        if not mag_pos: # Zero velocity
            return np.zeros(position.shape)

        weight_rot = self.get_weight(position)
        rot_final = self.max_rotation * weight_rot

        # TODO: why (-1) ?! change?
        velocity = get_angle_space_inverse(
            dir_angle_space=(-1)*rot_final, null_direction=(-1)*rel_position)
            
        if self.maximum_velocity is not None:
            mag_pos = min(mag_pos, self.maximum_velocity)
        velocity = velocity * mag_pos
        return velocity

    def get_weight(self, position: np.ndarray) -> float:
        """ Get weight of local rotation. Dicreasing weight -> less rotated DS."""
        position = np.array(position)
        position = self.pose.transform_position_from_reference_to_local(position)
        
        # Direction towards 'attractor' (at origin)
        scaled_dist_ellipse = self.get_scaled_dist_to_ellipse(position)
        # pose.transform_direction_from_reference_to_local(dir_rot)

        if scaled_dist_ellipse <= 1:
            # Inner strong-influence core
            weight_rot = 1
            
        elif scaled_dist_ellipse >= 1 + self.influence_descent_factor:
            # No influence far-away 
            weight_rot = 0
            
        else:
            # scaled_dist_ellipse = 1-scaled_dist_ellipse
            scaled_dist_ellipse = 1-(scaled_dist_ellipse-1)
            weight_rot = scaled_dist_ellipse / self.influence_descent_factor
            
        # Evalaute center weight [inverse of 1]
        rel_pos = self.get_relative_position_to_attractor(position)
        dist_center = np.linalg.norm(rel_pos)

        if dist_center == 0:
            # dist_center = infinity -> give normalization
            weight_rot = 0
            
        elif dist_center < self.attractor_influence_radius:
            # Projection of f: (0, 1] -> (infty -> 0]
            weight_center = self.attractor_influence_radius/dist_center - 1.0
            sum_weights = (weight_center + weight_rot)
            if sum_weights > 1:
                weight_rot = weight_rot / sum_weights

        # else:
            # weight_center = 0
            # -> no influence of weight_center normalization

        # Additional [0-1] weight from dot-product [no influence behind]
        vec_attr_pose = (-1)*self.get_relative_position_to_attractor(
            self.influence_pose.position)
        
        vec_position_attr = self.get_relative_position_to_attractor(position)

        dot_prod = (np.dot(vec_attr_pose, vec_position_attr)
                    / (LA.norm(vec_attr_pose)*LA.norm(vec_position_attr)))
        
        if dot_prod > 0:
            # in [0, 1]
            weight_rot = weight_rot * (1-dot_prod)
            
        return weight_rot
