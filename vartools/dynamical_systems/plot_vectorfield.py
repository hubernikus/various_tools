"""
Plot Dynamical System
"""
# Author: Lukas Huber
# Email: lukas.huber@epfl.ch
# Created: 2021-10-26
import numpy as np
import matplotlib.pyplot as plt

from ._base import DynamicalSystem


def plot_dynamical_system_quiver(n_resolution=None, *args, **kwargs):
    if n_resolution is None:
        n_resolution = 15
    return plot_dynamical_system(
        plottype="quiver", n_resolution=n_resolution, *args, **kwargs
    )


def plot_dynamical_system_streamplot(n_resolution=None, *args, **kwargs):
    if n_resolution is None:
        n_resolution = 100
    return plot_dynamical_system(
        plottype="streamplot", n_resolution=n_resolution, *args, **kwargs
    )


def plot_dynamical_system(
    dynamical_system: DynamicalSystem = None,
    x_lim=None,
    y_lim=None,
    n_resolution=15,
    figsize=(10, 7),
    plottype="quiver",
    axes_equal=True,
    fig_ax_handle=None,
    DynamicalSystem=None,
    color="blue",
    zorder=0,
    ax=None,
):
    """Evaluate the dynamics of the dynamical system."""
    if DynamicalSystem is not None:
        raise Exception(
            "'DynamicalSystem' -> Argument depreciated,"
            + " use 'dynamical_system' instead."
        )

    # dim = 2   # only for 2d implemented

    if x_lim is None:
        x_lim = [-10, 10]

    if y_lim is None:
        y_lim = [-10, 10]

    nx = n_resolution
    ny = n_resolution
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )

    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    velocities = np.zeros(positions.shape)
    for it in range(positions.shape[1]):
        velocities[:, it] = dynamical_system.evaluate(positions[:, it])

    # plt.figure()
    if ax is not None:
        # ax = ax
        fig = None
    elif fig_ax_handle is not None:
        fig, ax = fig_ax_handle
        pass
    elif ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if plottype == "quiver":
        ax.quiver(
            positions[0, :],
            positions[1, :],
            velocities[0, :],
            velocities[1, :],
            color="blue",
        )

    elif plottype == "streamplot":
        ax.streamplot(
            x_vals,
            y_vals,
            velocities[0, :].reshape(nx, ny),
            velocities[1, :].reshape(nx, ny),
            color=color,
            zorder=zorder,
        )
    else:
        raise ValueError(f"Unknown plottype '{plottype}'.")

    if axes_equal:
        ax.set_aspect("equal", adjustable="box")

    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])

    plt.ion()
    plt.show()

    return fig, ax
