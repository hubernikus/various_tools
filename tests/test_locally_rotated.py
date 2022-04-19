#!/USSR/bin/python3.9
"""
Dynamical Systems with a closed-form description.
"""
import unittest
from math import pi

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from vartools.states import ObjectPose

from vartools.dynamical_systems import LocallyRotated
from vartools.dynamical_systems import plot_dynamical_system_quiver


class TestSpiralmotion(unittest.TestCase):
    def test_initialize_zero_max_rotation(self, visualize=False):
        """Being able to initiate from initial rotation only."""
        dynamical_system = LocallyRotated(
            max_rotation=np.array([0]),
            influence_pose=ObjectPose(position=np.array([4, 4])),
            influence_radius=1,
        )

        if visualize:
            plot_dynamical_system_quiver(
                dynamical_system=dynamical_system, n_resolution=20
            )

        position = np.array([1, 0])
        velocity = dynamical_system.evaluate(position=position)

        norm_dot = (np.dot(velocity, position)) / (
            LA.norm(velocity) * LA.norm(position)
        )
        self.assertTrue(np.isclose(norm_dot, -1))

    def test_initialize_more_than_pi_max_rotation(self, visualize=False):
        # Test at center
        dynamical_system = LocallyRotated(
            max_rotation=np.array([pi]),
            influence_pose=ObjectPose(position=np.array([4.0, 4.0])),
            influence_radius=1,
        )

        if visualize:
            self.visualize_weight(dynamical_system)
            plot_dynamical_system_quiver(
                dynamical_system=dynamical_system, n_resolution=20
            )

        position = np.array([4.0, 4.0])
        velocity = dynamical_system.evaluate(position=position)
        vel_init = (-1) * position
        norm_dot = (np.dot(velocity, vel_init)) / (
            LA.norm(velocity) * LA.norm(vel_init)
        )
        self.assertTrue(np.isclose(norm_dot, np.cos(dynamical_system.max_rotation)))

    def test_weight_far_away(self, visualize=False):
        dynamical_system = LocallyRotated(
            max_rotation=np.array([0]),
            influence_pose=ObjectPose(position=np.array([4, 4])),
            influence_radius=1,
        )

        if visualize:
            self.visualize_weight(dynamical_system)

        position = np.array([0, 0])
        weight = dynamical_system.get_weight(position=position)
        self.assertAlmostEqual(weight, 0)

        position = np.array([0, 1e-5])
        weight = dynamical_system.get_weight(position=position)
        self.assertAlmostEqual(weight, 0)

        position = dynamical_system.influence_pose.position
        weight = dynamical_system.get_weight(position=position)
        self.assertAlmostEqual(weight, 1)

    def test_ellipse_with_axes(self, visualize=False):
        dynamical_system = LocallyRotated(
            max_rotation=np.array([0]),
            influence_pose=ObjectPose(
                position=np.array([-2, 3]), orientation=45 * pi / 180.0
            ),
            influence_axes_length=np.array([2, 1]),
        )

        if visualize:
            self.visualize_weight(dynamical_system)

        position = np.array([0, 0])
        weight = dynamical_system.get_weight(position=position)
        self.assertAlmostEqual(weight, 0)

        position = dynamical_system.influence_pose.position
        weight = dynamical_system.get_weight(position=position)
        self.assertAlmostEqual(weight, 1)

    def test_weight_close_point(self, visualize=False):
        dynamical_system = LocallyRotated(
            max_rotation=np.array([1]),
            influence_pose=ObjectPose(position=np.array([2, 2])),
            influence_radius=3,
        )
        if visualize:
            self.visualize_weight(dynamical_system)

        position = np.array([0, 0])
        weight = dynamical_system.get_weight(position=position)
        self.assertAlmostEqual(weight, 0)

        position = np.array([0, 1e-8])
        weight = dynamical_system.get_weight(position=position)
        self.assertAlmostEqual(weight, 0)

        position = dynamical_system.influence_pose.position
        weight = dynamical_system.get_weight(position=position)
        self.assertAlmostEqual(weight, 1)

        # Zero effect 'behind' attractor
        position = np.array([-0.1, -0.1])
        weight = dynamical_system.get_weight(position=position)
        self.assertAlmostEqual(weight, 0)

    def plot_ds_around_obstacle(self):
        from dynamic_obstacle_avoidance.obstacles import Ellipse

        obs = Ellipse(
            center_position=np.array([-8, 0]),
            axes_length=np.array([3, 1]),
            orientation=10.0 / 180 * pi,
        )

        dynamical_system = LocallyRotated(max_rotation=[-np.pi / 2]).from_ellipse(obs)

        plot_dynamical_system_quiver(dynamical_system=dynamical_system, n_resolution=20)
        self.visualize_weight(dynamical_system)

    def plot_dynamical_system(self):
        dynamical_system = LocallyRotated(
            max_rotation=[-np.pi / 2],
            # mean_rotation=[np.pi],
            influence_pose=ObjectPose(position=np.array([5, 2])),
            influence_radius=3,
        )

        # plot_dynamical_system_quiver(dynamical_system=dynamical_system,
        # n_resolution=20)
        self.visualize_weight(dynamical_system)

    def plot_critical_ds(self):
        dynamical_system = LocallyRotated(
            mean_rotation=[np.pi], rotation_center=[4, 2], influence_radius=4
        )

        plot_dynamical_system_quiver(dynamical_system=dynamical_system, n_resolution=20)

    def visualize_weight(
        self, dynamical_system=None, x_lim=[-10, 10], y_lim=[-10, 10], dim=2
    ):
        if dynamical_system is None:
            dynamical_system = LocallyRotated(
                mean_rotation=[np.pi], rotation_center=[4, 2], influence_radius=4
            )

        n_resolution = 100

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1)

        x_vals = np.linspace(x_lim[0], x_lim[1], n_resolution)
        y_vals = np.linspace(y_lim[0], y_lim[1], n_resolution)

        gamma_values = np.zeros((n_resolution, n_resolution))
        positions = np.zeros((dim, n_resolution, n_resolution))

        for ix in range(n_resolution):
            for iy in range(n_resolution):
                positions[:, ix, iy] = [x_vals[ix], y_vals[iy]]

                gamma_values[ix, iy] = dynamical_system.get_weight(positions[:, ix, iy])

        cs = plt.contourf(
            positions[0, :, :],
            positions[1, :, :],
            gamma_values,
            np.arange(0.0, 1.0, 0.05),
            extend="max",
            alpha=0.6,
            zorder=-3,
        )

        if dynamical_system.attractor_position is None:
            attractor = np.zeros(dynamical_system.dimension)
        else:
            attractor = dynamical_system.attractor_position

        ax.plot(attractor[0], attractor[1], "k*")
        ax.plot(
            dynamical_system.influence_pose.position[0],
            dynamical_system.influence_pose.position[1],
            "ko",
        )

        ax.axis("equal")
        cbar = fig.colorbar(cs)


if (__name__) == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)

    manual_tets = True
    if manual_tets:
        Tester = TestSpiralmotion()
        Tester.test_ellipse_with_axes(visualize=True)

        # Tester.test_weight_close_point(visualize=True)
        # Tester.test_initialize_zero_max_rotation(visualize=True)
        # Tester.test_initialize_more_than_pi_max_rotation(visualize=True)

        # Tester.plot_dynamical_system()
        # Tester.plot_critical_ds()
        # Tester.visualize_weight()
        # Tester.plot_ds_around_obstacle()

print("Done")
