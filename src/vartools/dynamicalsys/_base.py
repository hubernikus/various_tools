"""
Dynamical Systems with a closed-form description.
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021
import numpy as np

from abc import ABC, abstractmethod

def allow_max_velocity(original_function=None):
    ''' Decorator to allow to limit the velocity to a maximum. '''
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
    def __init__(self, center_position=None):
        if center_position is None:
            self.center_position = np.zeros(self.rotation_position.shape[0])
        else:
            self.center_position = np.array(center_position)

    def limit_velocity(self, velocity, max_vel):
        mag_vel = np.linalg.norm(velocity)
        if mag_vel > max_vel:
            velocity = velocity / mag_vel
        return velocity
        
    @abstractmethod
    def evaluate(self, center_position=None):
        """ Return velocity of the evaluated the dynamical system at 'position'."""
        pass

    def evaluate_array(self, position_array):
        """ Return an array of positions evluated. """
        velocity_array = np.zeros(position_array.shape)
        for ii in range(position_array.shape[1]):
            velocity_array[:, ii] = self.evaluate_array(position_array[:, ii])
        return velocity_array
