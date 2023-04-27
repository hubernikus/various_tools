import numpy as np

import matplotlib.pyplot as plt

from vartools.dynamical_systems import plot_dynamical_system_quiver
from vartools.dynamical_systems import SinusAttractorSystem


def test_sinus_system(visualize=False):
    x_lim = [-6.5, 6.5]
    y_lim = [-5.5, 5.5]
    n_grid = 30
    figsize = (4, 3.5)

    attractor = np.array([4.0, -3])
    initial_dynamics = SinusAttractorSystem(
        attractor_position=attractor,
        maximum_velocity=1.0,
    )

    fig, ax = plt.subplots(figsize=figsize)
    plot_dynamical_system_quiver(
        dynamical_system=initial_dynamics, x_lim=[-5, 5], y_lim=[-4, 4], ax=ax
    )
    ax.plot(
        initial_dynamics.attractor_position[0],
        initial_dynamics.attractor_position[1],
        "*k",
    )


if (__name__) == "__main__":
    test_sinus_system()
