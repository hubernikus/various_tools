#!/usr/bin/python3
"""
Visual test to evaluate the dynamicals systems.
"""

__author__ = "LukasHuber"
__date__ = "2021-05-18"
__email__ = "lukas.huber@epfl.ch"

import numpy as np
import matplotlib.pyplot as plt

from vartools.dynamicalsys.closedform import evaluate_linear_dynamical_system
from vartools.dynamicalsys.closedform import evaluate_stable_circle_dynamical_system


def plot_dynamical_system(func, x_lim=None, y_lim=None, num_grid=15, func_kwargs=None):
    """ Evaluate the dynamics of the dynamical system. """
    if func_kwargs is None:
        func_kwargs = {}
        
    dim = 2   # only for 2d implemented
    if x_lim is None:
        x_lim = [-10, 10]
        
    if y_lim is None:
        y_lim = [-10, 10]

    x_vals = np.linspace(x_lim[0], x_lim[1], num_grid)
    y_vals = np.linspace(y_lim[0], y_lim[1], num_grid)

    positions = np.zeros((dim, num_grid, num_grid))
    velocities = np.zeros((dim, num_grid, num_grid))
    for ix in range(num_grid):
        for iy in range(num_grid):
            positions[:, ix, iy] = [x_vals[ix], y_vals[iy]]
            velocities[:, ix, iy] = func(positions[:, ix, iy], max_vel=1, **func_kwargs)

    plt.figure()
    plt.quiver(positions[0, :, :], positions[1, :, :],
               velocities[0, :, :], velocities[1, :, :], color="blue")
               
    plt.show()
    plt.axis('equal')
    plt.xlim(x_lim[0], x_lim[1])
    plt.ylim(y_lim[0], y_lim[1])
    pass


if (__name__) == "__main__":
    plt.close('all')
    plt.ion()  # Interactive plotting
    
    plot_dynamical_system(evaluate_linear_dynamical_system)

    plot_dynamical_system(
        func=evaluate_stable_circle_dynamical_system,
        func_kwargs={'radius': 8})
