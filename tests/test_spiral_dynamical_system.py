#!/USSR/bin/python3.9
""" Sample DS spiral-trajectories."""
# Author: Sthith
#         Lukas Huber*
# *Email: hubernikus@gmail.com
# Created: 2021-06-23
# License: BSD (c) 2021

import unittest

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from vartools.dynamical_systems import SpiralStable


class TestSpiralmotion(unittest.TestCase):
    def test_creation(self, visualize=False):
        # Complexity of the spiral)
        complexity_spiral = 15

        # Base spiral
        SpiralDS = SpiralStable(complexity_spiral=complexity_spiral, p_radius_control=1)

        dataset_analytic = SpiralDS.get_positions(n_points=500)

        dt = 0.0005
        # start_position = SpiralDS.get_positions(n_points=1, tt=[0.001])[0, :]
        # start_position = np.array([1, 2, 1])
        start_position = np.array([0.2, 0.2, 0.4])
        dataset_ds = SpiralDS.motion_integration(start_position, dt).T

        # dataset_ds = np.array(spiral_motion_integrator(start_position, dt,
        # complexity_spiral, end_point))

        if visualize:
            # fig = plt.figure("Figure: c = "+str(complexity_spiral))
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1, projection="3d")
            ax.plot(
                dataset_analytic[:, 0],
                dataset_analytic[:, 1],
                dataset_analytic[:, 2],
                "b",
            )
            ax.scatter(
                dataset_analytic[0, 0],
                dataset_analytic[0, 1],
                dataset_analytic[0, 2],
                color="black",
                marker="o",
                label="Start",
            )
            ax.scatter(
                dataset_analytic[-1, 0],
                dataset_analytic[-1, 1],
                dataset_analytic[-1, 2],
                color="black",
                marker="*",
                label="End",
            )
            ax.plot(
                [SpiralDS.attractor_position[0]],
                [SpiralDS.attractor_position[1]],
                [SpiralDS.attractor_position[2]],
                "k*",
            )

            ax.set_xlabel("X(m)", fontsize=18, labelpad=15)
            ax.set_ylabel("Y(m)", fontsize=18, labelpad=15)
            ax.set_zlabel("Z(m)", fontsize=18, labelpad=15)
            ax.legend(fontsize=18)
            ax.set_title("Demonstration", fontsize=20, pad=20)

            ax_ds = fig.add_subplot(1, 2, 2, projection="3d")
            ax_ds.plot(dataset_ds[:, 0], dataset_ds[:, 1], dataset_ds[:, 2], "r--")
            ax_ds.set_xlabel("X(m)", fontsize=18, labelpad=15)
            ax_ds.set_ylabel("Y(m)", fontsize=18, labelpad=15)
            ax_ds.set_zlabel("Z(m)", fontsize=18, labelpad=15)
            ax_ds.set_title("Spiral DS", fontsize=20, pad=20)
            ax_ds.plot(
                [SpiralDS.attractor_position[0]],
                [SpiralDS.attractor_position[1]],
                [SpiralDS.attractor_position[2]],
                "k*",
            )

            plt.ion()
            plt.show()


if __name__ == "__main__":
    unittest.main()

    test_visualization = False
    if test_visualization:
        Tester = TestSpiralmotion()
        Tester.test_creation(visualize=test_visualization)

print("Done")
