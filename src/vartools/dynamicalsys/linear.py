"""
Dynamical Systems with a closed-form description.
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021

import numpy as np

from vartools.dynamicalsys import DynamicalSystem, allow_max_velocity
from vartools.directional_space import get_angle_space_inverse


class LinearSystem(DynamicalSystem):
    """
    Linear Dyanmical system of the form
    dot[x] = A @ (x - center_position) or
    dot[x] = A @ x + b

    Parameters
    ----------
    position: Position at which the dynamical system is evaluated
    center_position: Center of the dynamical system - rotation has to reach <pi/2 at this position
    b:
    
    Return
    ------
    Velocity (dynamical system) evaluted at the center position
    """
    def __init__(self, A_matrix=None, center_position=None, b=None):
        if A_matrix is None:
            self.A_matrix = np.eye(position.shape[0]) * (-1)
        else:
            self.A_matrix = A_matrix

        if b is not None:
            if center_position is not None:
                raise ValueError("center_pos AND baseline default arguments has been used." +
                                 "Only one of them possible.")
            center_position = np.linalg.pinv(self.A_matrix) @ b

        super().__init__(center_position)

    def evaluate(self, position, max_vel=None):
        velocity =  self.A_matrix.dot(position - self.center_position)

        if max_vel is not None:
            velocity = self.limit_velocity(velocity, max_vel)

        return velocity


class ConstantValue(DynamicalSystem):
    """ Returns constant velocity based on the DynamicalSystem parent-class"""
    def __init__(self, velocity):
        self.constant_velocity = velocity

    def evaluate(self, *args, **kwargs):
        """ Random input arguments, but always ouptuts same vector-field """
        return self.constant_velocity
    
