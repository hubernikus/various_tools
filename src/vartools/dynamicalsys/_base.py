"""
Dynamical Systems with a closed-form description.
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021

from abc import ABC, abstractmethod

import numpy as np


def allow_max_velocity(original_function=None):
    ''' Decorator to allow to limit the velocity to a maximum.'''
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


class DynamicalSystem(ABC):
    """ Virtual Class for Base dynamical system"""
    def __init__(self, center_position=None, maximum_velocity=None, dimension=None):
        if center_position is None:
            self.center_position = np.zeros(dimension)
        else:
            self.center_position = np.array(center_position)
            self.center_position.shape[0] = dimension

        self.maximum_velocity = maximum_velocity

    def limit_velocity(self, velocity, maximum_velocity):
        if maximum_velocity is None:
            if self.maximum_velocity is None:
                return velocity
            else:
                maximum_velocity = self.maximum_velocity
        
        mag_vel = np.linalg.norm(velocity)
        if mag_vel > maximum_velocity:
            velocity = velocity / mag_vel * maximum_velocity
        return velocity

    @abstractmethod
    def evaluate(self, position):
        """ Return velocity of the evaluated the dynamical system at 'position'."""
        pass

    def compute_dynamics(self, position):
        # This  or 'evaluate' / to be or not to be?!
        pass

    def evaluate_array(self, position_array):
        """ Return an array of positions evluated. """
        velocity_array = np.zeros(position_array.shape)
        for ii in range(position_array.shape[1]):
            velocity_array[:, ii] = self.evaluate_array(position_array[:, ii])
        return velocity_array
