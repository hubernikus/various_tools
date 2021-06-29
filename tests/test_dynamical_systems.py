#!/USSR/bin/python3.9
""" Sample DS spiral-trajectories."""
import unittest

from math import pi

import numpy as np
import matplotlib.pyplot as plt

from vartools.dynamical_systems import PendulumDynamics, DuffingOscillator, BifurcationSpiral
from vartools.dynamical_systems import plot_dynamical_system_streamplot


class TestDynamicalSystems(unittest.TestCase):
    def visualize_pendulum(self):
        MyDynamics = PendulumDynamics(maximum_velocity=1)
        plot_dynamical_system_streamplot(
            DynamicalSystem=MyDynamics,
            # x_lim=[-5, 10], y_lim=[-np.pi/2, np.pi/2], axes_equal=False)
            x_lim=[-5, 10], y_lim=[-2*pi, 2*pi], axes_equal=True)
                                         
        plt.title("Pendulum Dynamics")

    def visualize_duffing(self):
        MyDynamics = DuffingOscillator(maximum_velocity=1)
        plot_dynamical_system_streamplot(DynamicalSystem=MyDynamics,
                                         # x_lim=[-3.5, 3.5], y_lim=[-6, 6], axes_equal=False)
                                         x_lim=[-8, 8], y_lim=[-6, 6], axes_equal=True)
        plt.title("Duffing Oscillator")

    def visualize_spiral(self):
        MyDynamics = BifurcationSpiral(maximum_velocity=1)
        plot_dynamical_system_streamplot(DynamicalSystem=MyDynamics,
                                         # x_lim=[-6.0, 6.0], y_lim=[-2.5, 6.5], axes_equal=False)
                                         x_lim=[-6.0, 6.0], y_lim=[-5.0, 5.0], axes_equal=True)
        plt.title("Bifurcation Spiral")


if __name__ == '__main__':
    # unittest.main()
    visual_tets = True
    if visual_tets:
        MyTester = TestDynamicalSystems()
        MyTester.visualize_pendulum()
        MyTester.visualize_duffing()
        MyTester.visualize_spiral()
    
print('Done')

