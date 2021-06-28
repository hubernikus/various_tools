#!/USSR/bin/python3.9
""" Sample DS spiral-trajectories."""
import unittest

import numpy as np
import matplotlib.pyplot as plt

from vartools.dynamicalsys import PendulumDynamics, DuffingOscillator, BifurcationSpiral
from vartools.dynamicalsys import plot_dynamical_system_streamplot


class TestSpiralmotion(unittest.TestCase):
    def visualize_pendulum(self):
        MyDynamics = PendulumDynamics(maximum_velocity=1)
        plot_dynamical_system_streamplot(DynamicalSystem=MyDynamics,
                                         x_lim=[-5, 10], y_lim=[-np.pi/2, np.pi/2], axes_equal=False)
        plt.title("Pendulum Dynamics")

    def visualize_duffing(self):
        MyDynamics = DuffingOscillator(maximum_velocity=1)
        plot_dynamical_system_streamplot(DynamicalSystem=MyDynamics,
                                         x_lim=[-3.5, 3.5], y_lim=[-6, 6], axes_equal=False)
        plt.title("Duffing Oscillator")

    def visualize_spiral(self):
        MyDynamics = BifurcationSpiral(maximum_velocity=1)
        plot_dynamical_system_streamplot(DynamicalSystem=MyDynamics,
                                         x_lim=[-6.0, 6.0], y_lim=[-2.5, 6.5], axes_equal=True)
        plt.title("Bifurcation Spiral")


if __name__ == '__main__':
    # unittest.main()
    visual_tets = True
    if visual_tets:
        Tester = TestSpiralmotion()
        Tester.visualize_pendulum()
        Tester.visualize_duffing()
        Tester.visualize_spiral()
    
print('Done')

