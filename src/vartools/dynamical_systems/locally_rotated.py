"""
Dynamical Systems with a closed-form description.
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021

# Use python 3.10 [annotations / typematching]
from __future__ import annotations  # Not needed from python 3.10 onwards

import numpy as np

from ._base import DynamicalSystem
from vartools.directional_space import get_angle_space_inverse
from vartools.state import ObjectPose

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
    
    Return
    ------
    Velocity (dynamical system) evaluted at the center position
    """
    def __init__(self, mean_rotation: np.array = None,
                 influence_pose: pose = None,
                 influence_radius: float = 1,
                 influence_axes_length: np.ndarray = None,
                 delta_influence_center: float = 0.1,
                 influence_descent: float = 0.5,
                 *args, **kargs):
        # TODO: change 'Obstacle' to general 'State' (or similar) & make 'Obstacle' a subclass
        # of the former
        self.mean_rotation = np.array(mean_rotation)

        self.dimension = self.mean_rotation.shape[0] + 1
        
        super().__init__(*args, **kargs)

        if influence_axes_length is None:
            self.influence_axes_length = np.ones(self.dimension) * influence_radius
        else:
            self.influence_axes_length = influence_axes_length
        self.influence_pose = influence_pose
        
        # Added weight to the rotation to ensure that @ center the rotation stable (< pi/2)
        self.rotation_weight = np.linalg.norm(self.mean_rotation) / (np.pi/2)
        
        # Additional influence for center such that stirctly smallre than pi/2
        self.delta_influence_center = delta_influence_center
        self.influence_descent = influence_descent

    def from_ellipse(self, ellipse: Ellipse) -> LocallyRotated:
        self.influence_pose = ellipse.pose
        self.influence_axes_length = ellipse.axes_length
        return self

    def transform_global_to_ellipse_frame(self, direction):
        """ Transform to ellipse frame"""
        if self.influence_pose is not None:
            direction = self.influence_pose.transform_direction_from_reference_to_local(
                direction)
        direction = direction / self.influence_axes_length
        return direction

    def transform_ellipse_to_global_frame(self, direction):
        """ Reverse transform. """
        direction = self.axes_length * direction
        if self.influence_pose is not None:
            direction = self.influence_pose.transform_direction_from_reference_to_local(
                direction)
        return direction
        
    def evaluate(self, position):
        weight_rot = self.get_weight(position)

        rot_final = self.mean_rotation * weight_rot
        
        if self.attractor_position is None:
            attractor_position = np.zeros(self.dimension)
        else:
            attractor_position = self.attractor_position

        dir_attractor = attractor_position - position
        
        if not np.linalg.norm(dir_attractor): # Zero velocity
            return np.zeros(position.shape)

        # TODO: why (-1) ?! change?
        velocity = get_angle_space_inverse(
            dir_angle_space=(-1)*rot_final, null_direction=dir_attractor)
            
        magnitude = np.linalg.norm(position-attractor_position)

        if self.maximum_velocity is not None:
            magnitude = min(magnitude, self.maximum_velocity)
        velocity = velocity * magnitude
        
        return velocity

    def get_weight(self, position: np.ndarray) -> np.ndarray:
        # Evalaute the weights
        dir_rot = self.transform_global_to_ellipse_frame(dir_rot)
        # pose.transform_direction_from_reference_to_local(dir_rot)

        dist_rot = np.linalg.norm(dir_rot)

        unit_radius = 1
        if dist_rot <= unit_radius:
            weight_rot = 1
        elif dist_rot >= 1./self.influence_descent*unit_radius:
            weight_rot = 0
        else:
            dist_rot = dist_rot - unit_radius
            weight_rot = (unit_radius - dist_rot*self.influence_descent)/(unit_radius)

        # Evalaute center weight
        dist_center = np.linalg.norm(position-self.pose.position)
        if dist_center > self.influence_at_center:
            weight_center = 0
        else:
            weight_center = (self.influence_at_center - dist_center) / self.influence_at_center
            weight_center = weight_center * self.rotation_weight

        sum_weights = weight_rot + weight_center
        if sum_weights > 1:
            weight_rot = weight_rot / sum_weights
            
        return weight_rot
