"""
Dynamical Systems with a closed-form description.
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021

import numpy as np

from ._base import DynamicalSystem


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
    def __init__(self, attractor_position=None, A_matrix=None, b=None):
        super().__init__(attractor_position=attractor_position)

        if A_matrix is None:
            self.A_matrix = np.eye(self.dimension) * (-1)
        else:
            self.A_matrix = A_matrix

        if b is not None:
            if attractor_position is not None:
                raise ValueError("center_pos AND baseline default arguments has been used." +
                                 "Only one of them possible.")
            self.attractor_position = np.linalg.pinv(self.A_matrix) @ b

    def evaluate(self, position, max_vel=None):
        velocity =  self.A_matrix.dot(position - self.attractor_position)
        velocity = self.limit_velocity(velocity, max_vel)
        return velocity

    def is_stable(self, position):
        """ Check stability of given A matrix """
        A = self.A_matrix + self.A_matrix.T
        eigvals, eigvecs = np.linalg.eig(A)

        if all(eigvals < 0):
            return True
        else:
            return False
        

class ConstantValue(DynamicalSystem):
    """ Returns constant velocity based on the DynamicalSystem parent-class"""
    def __init__(self, velocity):
        self.constant_velocity = velocity

    def evaluate(self, *args, **kwargs):
        """ Random input arguments, but always ouptuts same vector-field """
        return self.constant_velocity
    
