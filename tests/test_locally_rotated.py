#!/USSR/bin/python3.9
"""
Dynamical Systems with a closed-form description.
"""
import unittest
from math import pi

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

# from matplotlib.colors import LinearSegmentedColormap

from vartools.states import ObjectPose

from vartools.dynamical_systems import LocallyRotated
from vartools.dynamical_systems import plot_dynamical_system_quiver
from vartools.visualization import VectorfieldPlotter


def test_initialize_zero_max_rotation(visualize=False):
    """Being able to initiate from initial rotation only."""
    dynamical_system = LocallyRotated(
        max_rotation=np.array([0]),
        influence_pose=ObjectPose(position=np.array([4, 4])),
        influence_radius=1,
    )

    if visualize:
        plot_dynamical_system_quiver(dynamical_system=dynamical_system, n_resolution=20)

    position = np.array([1, 0])
    velocity = dynamical_system.evaluate(position=position)

    norm_dot = (np.dot(velocity, position)) / (LA.norm(velocity) * LA.norm(position))
    assert np.isclose(norm_dot, -1)


def test_pi_half_rotation(visualize=False):
    # Test at center
    dynamical_system = LocallyRotated(
        max_rotation=np.array([pi / 2]),
        influence_pose=ObjectPose(position=np.array([4.0, 4.0])),
        influence_radius=2,
        influence_descent_factor=2,
    )

    if visualize:
        visualize_weight(dynamical_system)
        plot_dynamical_system_quiver(dynamical_system=dynamical_system, n_resolution=20)

    # Rotation only in positive (further from the center)
    position = np.array([5.84, -1.54])
    velocity = dynamical_system.evaluate(position=position)
    assert np.cross(position, velocity) > 0, "Rotation in the wrong direction."

    # Rotation only in positive (close to center)
    position = np.array([4.55, 5.77])
    velocity = dynamical_system.evaluate(position=position)
    assert np.cross(position, velocity) > 0, "Rotation in the wrong direction."

    # Test rotation at center
    position = np.array([4.0, 4.0])
    velocity = dynamical_system.evaluate(position=position)
    vel_init = (-1) * position
    norm_dot = (np.dot(velocity, vel_init)) / (LA.norm(velocity) * LA.norm(vel_init))
    assert np.isclose(norm_dot, np.cos(dynamical_system.max_rotation))


def test_weight_far_away(visualize=False):
    dynamical_system = LocallyRotated(
        max_rotation=np.array([0]),
        influence_pose=ObjectPose(position=np.array([4, 4])),
        influence_radius=1,
    )

    if visualize:
        visualize_weight(dynamical_system)

    position = np.array([0, 0])
    weight = dynamical_system._get_weight(position=position)
    assert np.isclose(weight, 0)

    position = np.array([0, 1e-5])
    weight = dynamical_system._get_weight(position=position)
    assert np.isclose(weight, 0)

    position = dynamical_system.influence_pose.position
    weight = dynamical_system._get_weight(position=position)
    assert np.isclose(weight, 1)


def test_ellipse_with_axes(visualize=False):
    dynamical_system = LocallyRotated(
        max_rotation=np.array([0]),
        influence_pose=ObjectPose(
            position=np.array([-2, 3]), orientation=45 * pi / 180.0
        ),
        influence_axes_length=np.array([2, 1]),
    )

    if visualize:
        visualize_weight(dynamical_system)

    position = np.array([0, 0])
    weight = dynamical_system._get_weight(position=position)
    assert np.isclose(weight, 0)

    position = dynamical_system.influence_pose.position
    weight = dynamical_system._get_weight(position=position)
    assert np.isclose(weight, 1)


def test_weight_close_point(visualize=False):
    dynamical_system = LocallyRotated(
        max_rotation=np.array([np.pi * 0.7]),
        influence_pose=ObjectPose(position=np.array([2, 2])),
        influence_radius=5,
    )
    if visualize:
        x_lim = [-20, 20]
        y_lim = [-20, 20]

        fig, ax = visualize_weight(
            dynamical_system, n_resolution=100, x_lim=x_lim, y_lim=y_lim
        )

        my_plotter = VectorfieldPlotter(fig=fig, ax=ax, x_lim=x_lim, y_lim=y_lim)

        my_plotter.vector_color = "black"
        my_plotter.plot(dynamical_system.evaluate, n_resolution=20)

    position = np.array([0, 0])
    weight = dynamical_system._get_weight(position=position)
    assert np.isclose(weight, 0)

    position = np.array([0, 1e-8])
    weight = dynamical_system._get_weight(position=position)
    assert np.isclose(weight, 0)

    position = dynamical_system.influence_pose.position
    weight = dynamical_system._get_weight(position=position)
    assert np.isclose(weight, 1)

    # Zero effect 'behind' attractor
    position = np.array([-0.1, -0.1])
    weight = dynamical_system._get_weight(position=position)
    assert np.isclose(weight, 0)


def plot_ds_around_obstacle():
    from dynamic_obstacle_avoidance.obstacles import Ellipse

    obs = Ellipse(
        center_position=np.array([-8, 0]),
        axes_length=np.array([3, 1]),
        orientation=10.0 / 180 * pi,
    )

    dynamical_system = LocallyRotated(max_rotation=[-np.pi / 2]).from_ellipse(obs)

    plot_dynamical_system_quiver(dynamical_system=dynamical_system, n_resolution=20)
    visualize_weight(dynamical_system)


def plot_dynamical_system():
    dynamical_system = LocallyRotated(
        max_rotation=[-np.pi / 2],
        # mean_rotation=[np.pi],
        influence_pose=ObjectPose(position=np.array([5, 2])),
        influence_radius=3,
    )

    # plot_dynamical_system_quiver(dynamical_system=dynamical_system,
    # n_resolution=20)
    visualize_weight(dynamical_system)


def plot_critical_ds():
    dynamical_system = LocallyRotated(
        mean_rotation=[np.pi], rotation_center=[4, 2], influence_radius=4
    )

    plot_dynamical_system_quiver(dynamical_system=dynamical_system, n_resolution=20)


def visualize_weight(
    dynamical_system=None,
    x_lim=[-10, 10],
    y_lim=[-10, 10],
    dim=2,
    n_resolution=100,
    fig=None,
    ax=None,
):
    if dynamical_system is None:
        dynamical_system = LocallyRotated(
            mean_rotation=[np.pi], rotation_center=[4, 2], influence_radius=4
        )

    import matplotlib.pyplot as plt

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    x_vals = np.linspace(x_lim[0], x_lim[1], n_resolution)
    y_vals = np.linspace(y_lim[0], y_lim[1], n_resolution)

    gamma_values = np.zeros((n_resolution, n_resolution))
    positions = np.zeros((dim, n_resolution, n_resolution))

    for ix in range(n_resolution):
        for iy in range(n_resolution):
            positions[:, ix, iy] = [x_vals[ix], y_vals[iy]]

            gamma_values[ix, iy] = dynamical_system._get_weight(positions[:, ix, iy])

    n_split = 128
    # newcolors = cm.get_cmap('Reds_r', n_split)
    # newcolors = cm.get_cmap('YlOrBr', n_split)
    newcolors = cm.get_cmap("Oranges", n_split)
    # bottom = cm.get_cmap('rBlues', 128)
    newcolors = newcolors(np.linspace(0, 0.6, n_split))

    # newcolors = np.vstack((top(np.linspace(0, 1, 128)),
    # bottom(np.linspace(0, 1, 128))))
    newcmp = ListedColormap(newcolors, name="OrangeBlue")

    cs = plt.contourf(
        positions[0, :, :],
        positions[1, :, :],
        gamma_values,
        np.arange(0.0, 1.01, 0.01),
        # extend="max",
        # alpha=1.0,
        # alpha=0.9,
        zorder=-3,
        # colors="OrRd",
        # cmap="Reds",
        cmap=newcmp,
        # antialiased=True,
    )

    if dynamical_system.attractor_position is None:
        attractor = np.zeros(dynamical_system.dimension)
    else:
        attractor = dynamical_system.attractor_position

    ax.plot(attractor[0], attractor[1], "k*", markersize=14)
    ax.plot(
        dynamical_system.influence_pose.position[0],
        dynamical_system.influence_pose.position[1],
        "ko",
    )

    ax.axis("equal")
    fig.colorbar(cs, ticks=np.linspace(0, 1, 11))

    return fig, ax


if (__name__) == "__main__":
    plt.ion()
    plt.close("all")
    # test_initialize_zero_max_rotation(visualize=False)
    # test_pi_half_rotation(visualize=False)
    test_weight_close_point(visualize=True)
    # test_ellipse_with_axes(visualize=True)
    pass

print("Done")
