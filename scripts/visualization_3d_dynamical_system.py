#!/usr/bin/python3
"""
Tools to handle the direction space.
"""
# Author: Lukas Huber
# Created: 2021-05-16

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

from vartools.dynamical_systems import LinearSystem
from vartools.linalg import is_negative_definite, is_positive_definite

plt.ion()


def linear_ds_3d_vectorfield(A_matrix=None, b_offset=None, dimension=3):
    fig = plt.figure()
    ax = fig.gca(projection="3d")

    x, y, z = np.meshgrid(
        np.arange(-0.8, 1, 0.2), np.arange(-0.8, 1, 0.2), np.arange(-0.8, 1, 0.5)
    )

    if A_matrix is None:
        A_matrix = (-1) * np.eye(dimension)

    dynamical_system = LinearSystem(A_matrix=A_matrix, b=b_offset)

    prop = "" if is_positive_definite(A_matrix) else "not "
    print(f"A matrix is {prop}positive definite.")

    u = np.zeros(x.shape)
    v = np.zeros(x.shape)
    w = np.zeros(x.shape)

    for ix in range(x.shape[0]):
        for iy in range(x.shape[1]):
            for iz in range(x.shape[2]):
                pos = np.array([x[ix, iy, iz], y[ix, iy, iz], z[ix, iy, iz]])
                vel = dynamical_system.evaluate(position=pos)
                u[ix, iy, iz], v[ix, iy, iz], z[ix, iy, iz] = tuple(vel)

    # u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    # v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    # w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
    # np.sin(np.pi * z))

    ax.quiver(x, y, z, u, v, w, length=0.1, arrow_length_ratio=0.5)

    plt.show()


def linear_ds_3d_integration(A_matrix=None, b_offset=None):
    """ """
    if A_matrix is None:
        # Default values
        A_matrix = np.array(
            [[-1.2, -0.4, -0.8], [-0.4, -1.2, -0.6], [-0.8, -0.6, -1.2]]
        )

        A_matrix = np.array([[-2, 1, 0], [1, -2, 1], [0, 1, -2]])

        A_matrix = np.array([[-1, 1.0, 0.2], [1.0, -1.0, 1], [0.2, 1, -1.3]])

    if b_offset is None:
        pass

    dynamical_system = LinearSystem(A_matrix=A_matrix, b=b_offset)

    prop = "" if is_negative_definite(A_matrix) else "not "
    print(f"A matrix is {prop}negative definite.")

    dim = 3
    n_traj = 20
    n_iter_max = 1000
    delta_time = 1e-1
    margin_convergence = 1e-3

    x_range = [-1, 1]
    y_range = [-1, 1]
    z_range = [-1, 1]

    min_val = np.array([x_range[0], y_range[0], z_range[0]])
    max_val = np.array([x_range[1], y_range[1], z_range[1]])

    traj_list = []
    for ii in range(n_traj):
        traj_list.append(np.zeros((dim, n_iter_max)))

        traj_list[ii][:, 0] = np.random.rand(3) * (max_val - min_val) + min_val

        for jj in range(1, n_iter_max):
            vel = dynamical_system.evaluate(position=traj_list[ii][:, jj - 1])

            # Check convergence
            if np.linalg.norm(vel) < margin_convergence:
                traj_list[ii] = traj_list[ii][:, :jj]
                print(f"Converged after {jj} iteration.")
                break

            traj_list[ii][:, jj] = traj_list[ii][:, jj - 1] + vel * delta_time

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    for ii in range(n_traj):
        ax.plot(traj_list[ii][0, 0], traj_list[ii][1, 0], traj_list[ii][2, 0], "k.")
        ax.plot(traj_list[ii][0, :], traj_list[ii][1, :], traj_list[ii][2, :])

        if b_offset is None:
            ax.plot(0, 0, 0, "k*")
        else:
            warnings.warn("b_offset not implemented")

    plt.show()


if (__name__) == "__main__":
    linear_ds_3d_vectorfield()
    # linear_ds_3d_integration()
