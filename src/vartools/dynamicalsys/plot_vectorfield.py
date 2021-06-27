"""
Plot Dynamical System
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_dynamical_system_quiver(func, x_lim=None, y_lim=None, n_resolution=15, func_kwargs=None):
    """ Evaluate the dynamics of the dynamical system. """
    if func_kwargs is None:
        func_kwargs = {}
        
    dim = 2   # only for 2d implemented
    if x_lim is None:
        x_lim = [-10, 10]
        
    if y_lim is None:
        y_lim = [-10, 10]

    x_vals = np.linspace(x_lim[0], x_lim[1], n_resolution)
    y_vals = np.linspace(y_lim[0], y_lim[1], n_resolution)

    positions = np.zeros((dim, n_resolution, n_resolution))
    velocities = np.zeros((dim, n_resolution, n_resolution))
    for ix in range(n_resolution):
        for iy in range(n_resolution):
            positions[:, ix, iy] = [x_vals[ix], y_vals[iy]]
            velocities[:, ix, iy] = func(positions[:, ix, iy], max_vel=1, **func_kwargs)

    plt.figure()
    plt.quiver(positions[0, :, :], positions[1, :, :],
               velocities[0, :, :], velocities[1, :, :], color="blue")

    plt.ion()
    plt.show()
    plt.axis('equal')
    plt.xlim(x_lim[0], x_lim[1])
    plt.ylim(y_lim[0], y_lim[1])
