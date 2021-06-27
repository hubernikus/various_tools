"""
Dynamical Systems with a closed-form description.
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021

import numpy as np

from vartools.dynamicalsys import DynamicalSystem, allow_max_velocity
from vartools.directional_space import get_angle_space_inverse

class LocallyRotated(DynamicalSystem):
    """ Returns dynamical system with a mean rotation of 'mean_rotation'
    at position 'rotation_position'

    Parameters
    ----------
    position: Position at which the dynamical system is evaluated
    center_position: Center of the dynamical system - rotation has to reach <pi/2 at this position
    mean_rotation: angle-space rotation at position.
    rotation_position: 
    influence_radius:
    
    Return
    ------
    Velocity (dynamical system) evaluted at the center position
    """
    def __init__(self, mean_rotation, rotation_position, center_position=None,
                 influence_radius=1, delta_influence_center=0.1, influence_descent=0.5):
        self.rotation_position = np.array(rotation_position)
        self.mean_rotation = np.array(mean_rotation)

        if center_position is None:
            self.center_position = np.zeros(self.rotation_position.shape[0])
        else:
            self.center_position = np.array(center_position)
        
        if np.allclose(self.center_position, self.rotation_position, rtol=influence_radius*1e-6):
            raise ValueError("Center and rotation position are too close to each other.")

        self.influence_radius = influence_radius

        # Added weight to the rotation to ensure that @ center the rotation stable (< pi/2)
        self.rotation_weight = np.linalg.norm(self.mean_rotation) / (np.pi/2)
        # Additional influence for center such that stirctly smallre than pi/2
        self.influence_center = self.influence_radius*(1 + delta_influence_center)

        self.influence_descent = influence_descent
        
    def evaluate(self, position, max_vel=None):
        weight_rot = self.get_weight(position)

        rot_final = self.mean_rotation * weight_rot

        dir_attractor = self.center_position - position

        if not np.linalg.norm(dir_attractor): # Zero velocity
            return np.zeros(position.shape)
        
        velocity = get_angle_space_inverse(dir_angle_space=rot_final, null_direction=dir_attractor)
        magnitude = np.linalg.norm(position - self.center_position)

        if max_vel is not None:
            magnitude = min(magnitude, max_vel)

        velocity = velocity * magnitude
        return velocity

    def get_weight(self, position):
        # Evalaute the weights
        dist_rot = np.linalg.norm(position-self.rotation_position)

        if dist_rot <= self.influence_radius:
            weight_rot = 1
        elif dist_rot >= 3*self.influence_radius:
            weight_rot = 0
        else:
            dist_rot = dist_rot - self.influence_radius
            weight_rot = (self.influence_radius-
                          dist_rot*self.influence_descent)/(self.influence_radius)

        # Evalaute center weight
        dist_center = np.linalg.norm(position-self.center_position)
        if dist_center > self.influence_center:
            weight_center = 0
        else:
            weight_center = (self.influence_center - dist_center) / self.influence_center
            weight_center = weight_center * self.rotation_weight

        sum_weights = weight_rot + weight_center
        if sum_weights > 1:
            weight_rot = weight_rot / sum_weights
            
        return weight_rot
        # return weight_center
