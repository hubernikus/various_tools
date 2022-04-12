#!/USSR/bin/python3.9
""" Sample various DS."""
from math import pi

import numpy as np
import matplotlib.pyplot as plt

from vartools.dynamical_systems import (
    PendulumDynamics,
    DuffingOscillator,
    BifurcationSpiral,
)
from vartools.dynamical_systems import plot_dynamical_system_streamplot


def visualize_pendulum_system():
    my_dynamics = PendulumDynamics(maximum_velocity=1)
    plot_dynamical_system_streamplot(
        dynamical_system=my_dynamics,
        # x_lim=[-5, 10], y_lim=[-np.pi/2, np.pi/2], axes_equal=False)
        x_lim=[-5, 10],
        y_lim=[-2 * pi, 2 * pi],
        axes_equal=True,
    )
    plt.title("Pendulum Dynamics")


def visualize_duffing_system():
    my_dynamics = DuffingOscillator(maximum_velocity=1)
    plot_dynamical_system_streamplot(
        dynamical_system=my_dynamics,
        # x_lim=[-3.5, 3.5], y_lim=[-6, 6], axes_equal=False)
        x_lim=[-8, 8],
        y_lim=[-6, 6],
        axes_equal=True,
    )
    plt.title("Duffing Oscillator")


def visualize_spiral_system():
    my_dynamics = BifurcationSpiral(maximum_velocity=1)
    plot_dynamical_system_streamplot(
        dynamical_system=my_dynamics,
        # x_lim=[-6.0, 6.0], y_lim=[-2.5, 6.5], axes_equal=False)
        x_lim=[-6.0, 6.0],
        y_lim=[-5.0, 5.0],
        axes_equal=True,
    )
    plt.title("Bifurcation Spiral")


def visualize_sinuswave_system(save_figure=False):
    from vartools.dynamical_systems import ConstVelocityDecreasingAtAttractor
    from vartools.dynamical_systems import SinusAttractorSystem

    trimmer = ConstVelocityDecreasingAtAttractor(
        const_velocity=1.0, distance_decrease=1.0
    )
    my_dynamics = SinusAttractorSystem(trimmer=trimmer, attractor_position=np.zeros(2))

    plot_dynamical_system_streamplot(
        dynamical_system=my_dynamics,
        # x_lim=[-6.0, 6.0], y_lim=[-2.5, 6.5], axes_equal=False)
        x_lim=[-8.0, 1.0],
        y_lim=[-4.0, 4.0],
        axes_equal=True,
    )
    plt.title("Sinus Attractor System")

    if save_figure:
        figure_name = "sinus_attractor_system"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


if (__name__) == "__main__":
    plt.ion()
    # visualize_pendulum_system()
    # visualize_duffing_system()
    # visualize_spiral_system()
    # visualize_sinuswave_system(save_figure=True)

print("Done")
