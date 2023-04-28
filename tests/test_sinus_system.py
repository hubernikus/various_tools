import math
import numpy as np

import matplotlib.pyplot as plt

from vartools.dynamics import plot_dynamical_system_quiver
from vartools.dynamics import WavyRotatedDynamics
from vartools.states import Pose


def test_sinus_system(visualize=False):
    attractor = np.array([4.0, -3])
    initial_dynamics = WavyRotatedDynamics(
        pose=Pose(position=np.array([0, 0]), orientation=0),
        maximum_velocity=1.0,
        rotation_frequency=1,
        rotation_power=1.2,
        max_rotation=0.4 * math.pi,
    )

    if visualize:
        x_lim = [-6.5, 6.5]
        y_lim = [-5.5, 5.5]
        n_grid = 30
        figsize = (4, 3.5)

        fig, ax = plt.subplots(figsize=figsize)
        plot_dynamical_system_quiver(
            dynamical_system=initial_dynamics, x_lim=[-5, 5], y_lim=[-4, 4], ax=ax
        )
        ax.plot(
            initial_dynamics.pose.position[0],
            initial_dynamics.pose.position[1],
            "*k",
        )

    # Evaluate
    position = np.array([0.01, 0])
    velocity = initial_dynamics.evaluate(position)
    assert abs(velocity[0]) > abs(
        velocity[1]
    ), "System is expect to point towards attractor fast."


if (__name__) == "__main__":
    test_sinus_system(visualize=True)
