#!/USSR/bin/python3.10
""" Test / visualization of line following. """
# Author: Lukas Huber
# Created: 2022-11-25
# Email: lukas.huber@epfl.ch
# License: BSD (c) 2025

import warnings
from functools import partial
import unittest
from math import pi
import math

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from vartools.dynamical_systems import CircularStable

# DirectionBase
from vartools.dynamical_systems import plot_dynamical_system_quiver


def test_circle_following_rotational_avoidance(visualize=False):
    global_ds = CircularStable(radius=10, maximum_velocity=1)

    if visualize:
        plt.close("all")
        fig, ax = plot_dynamical_system_quiver(
            dynamical_system=global_ds,
            x_lim=[-15, 15],
            y_lim=[-15, 15],
            n_resolution=30,
        )
        ax.scatter(
            0,
            0,
            marker="*",
            s=200,
            color="black",
            zorder=5,
        )

    # Specific position (just inside radius)
    radius = global_ds.radius - 1e-5
    position = np.array([10, 0])
    position = position / LA.norm(position) * radius
    velocity = global_ds.evaluate(position)
    assert math.isclose(np.dot(position, velocity), 0, abs_tol=1e4)
    assert np.cross(position, velocity) > 0

    # On the surface
    position = np.array([global_ds.radius, 0])
    velocity = global_ds.evaluate(position)
    assert math.isclose(np.dot(position, velocity), 0)

    # Position at center
    position = np.array([0, 0])
    velocity = global_ds.evaluate(position)
    assert velocity is not None, "Define a velocity at origin."

    # Position is outside
    position = np.array([15, 2])
    velocity = global_ds.evaluate(position)
    assert np.dot(position, velocity) < 0, "Velocity should position inwards."
    assert np.cross(position, velocity) > 0

    # Position is inside
    position = np.array([2, -4])
    velocity = global_ds.evaluate(position)
    assert np.dot(position, velocity) > 0, "Velocity should position outwards."
    assert np.cross(position, velocity) > 0


def test_circular_dynamics(visualize=False):
    circular_ds = CircularStable(radius=1.0, maximum_velocity=2.0)

    if visualize:
        x_lim = [-2, 2]
        y_lim = [-2, 2]

        figsize = (8, 6)

        fig, ax = plt.subplots(figsize=figsize)
        plot_dynamical_system(
            dynamical_system=circular_ds,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
        )

    # No velocity at center
    position = np.zeros(2)
    velocity = circular_ds.evaluate(position)
    assert np.allclose(velocity, np.zeros(2))

    # Pointing away far away (and additionally slight rotation; always)
    position = np.array([1e-3, 0])
    velocity = circular_ds.evaluate(position)
    assert velocity[1] > 0


def test_simple_circular(visualize=False):
    dynamics = SimpleCircularDynamics(pose=Pose.create_trivial(2), radius=2.0)

    if visualize:
        x_lim = [-4, 4]
        y_lim = [-4, 4]

        figsize = (8, 6)

        fig, ax = plt.subplots(figsize=figsize)
        plot_dynamical_system(
            dynamical_system=dynamics,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            plottype="quiver",
            n_resolution=40,
        )

    # Pointing outwards close to obstacle
    position = np.array([0.1, 0.0])
    velocity = dynamics.evaluate(position)

    norm_vel = velocity / np.linalg.norm(velocity)
    norm_pos = position / np.linalg.norm(position)
    assert np.dot(norm_vel, norm_pos) > 0.5


if (__name__) == "__main__":
    # Import visualization libraries here
    import matplotlib.pyplot as plt  # For debugging only (!)
    from vartools.dynamical_systems import plot_dynamical_system

    figtype = ".pdf"

    # test_circular_dynamics(visualize=False)
    # test_circular_dynamics(visualize=True)
    # test_circle_following_rotational_avoidance(visualize=True)
    test_simple_circular(visualize=True)

    print("Tests done")
