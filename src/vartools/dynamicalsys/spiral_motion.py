""" Spiral shaped dynamical system in 3D. """
# Author: Sthith
#         Lukas Huber
# Created: 2021-06-23
# License: BSD (c) 2021

import numpy as np

def spiral_analytic(c, n_points, dimension=3, tt=None):
    """ Create initial position. """
    dataset = np.zeros((n_points, dimension))

    if tt is None:
        tt = np.linspace(0, np.pi, n_points)
    else:
        # No lists, only numpy array
        tt = np.array(tt)

    dataset[:, 0] = np.sin(tt)*np.cos(c*tt)
    dataset[:, 1] = np.sin(tt)*np.sin(c*tt)
    dataset[:, 2] = np.cos(tt)
    
    return dataset

def spiral_motion_stable(position, dt, c, attractor):
    """ Return the velocity based on the evaluation of a spiral-shaped dynamical system."""

        
def spiral_motion(position, dt, c, attractor, radius_correction=0):
    """ Return the velocity based on the evaluation of a spiral-shaped dynamical system."""
    velocity = np.zeros(position.shape)
    
    # z = cos(theta)
    # -dz/dt = sin(theta)
    theta = np.arccos(position[2])

    velocity[0] = position[2]*np.cos(c*theta) - c*position[1]
    velocity[1] = position[2]*np.sin(c*theta) + c*position[0]
    velocity[2] = -np.sqrt(1-position[2]**2)

    if radius_correction:
        pos_norm = np.linalg.norm(position)
        correction_velocity = position
        velocity += 
    
    return velocity

def spiral_motion_integrator(start_position, dt, c, attractor):
    """ Integrate spiral motion. """
    dataset = []
    dataset.append(start_position)
    current_position = start_position

    while np.linalg.norm(current_position-attractor) > 1e-5 and current_position[2] > -1:
          delta_vel = spiral_motion(dataset[-1], dt, c, attractor)
          current_position = delta_vel*dt + current_position
          dataset.append(current_position)

    return dataset
