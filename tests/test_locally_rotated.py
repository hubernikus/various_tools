#!/USSR/bin/python3.9
"""
Dynamical Systems with a closed-form description.
"""
import unittest

import numpy as np
import matplotlib.pyplot as plt

from vartools.dynamical_systems import LocallyRotated
from vartools.dynamical_systems import plot_dynamical_system_quiver


class TestSpiralmotion(unittest.TestCase):
    def plot_dynamical_system(self):
        DynamicalSystem = LocallyRotated(
            mean_rotation=[-np.pi/2],
            # mean_rotation=[np.pi],
            rotation_center=[5, 2],
            influence_radius=3)
        
        plot_dynamical_system_quiver(DynamicalSystem=DynamicalSystem,
                                     n_resolution=20)

        self.test_weight(DynamicalSystem)

    def plot_critical_ds(self):
        DynamicalSystem = LocallyRotated(
            mean_rotation=[np.pi],
            rotation_center=[4, 2],
            influence_radius=4)
        
        plot_dynamical_system_quiver(DynamicalSystem=DynamicalSystem,
                                     n_resolution=20)
        
    def test_weight(self, DynamicalSystem=None, x_lim=[-10, 10], y_lim=[-10, 10], dim=2):
        if DynamicalSystem is None:
            DynamicalSystem = LocallyRotated(
                mean_rotation=[np.pi],
                rotation_center=[4, 2],
                influence_radius=4,
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

                gamma_values[ix, iy] = DynamicalSystem.get_weight(positions[:, ix, iy])

        cs = plt.contourf(positions[0, :, :], positions[1, :, :],  gamma_values, 
                         np.arange(0.0, 1.0, 0.05),
                         extend='max', alpha=0.6, zorder=-3)
        cbar = fig.colorbar(cs)

        pass

if __name__ == '__main__':
    # unittest.main()
    
    manual_tets = True
    if manual_tets:
        Tester = TestSpiralmotion()
        # Tester.plot_dynamical_system()
        # Tester.plot_critical_ds()
        Tester.test_weight()
    
print('Done')



