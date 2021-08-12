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
    # def test_weight(self):
    
    def test_initialize_from_mean_rotation(self):
        """ Being able to initiate from initial rotation only. """
        dynamical_system = LocallyRotated(
            max_rotation=np.array([0]),
            influence_pose=ObjectPose(position=np.array([4, 4])),
            influence_radius=1
            )

        position = np.array([1, 0])
        velocity = dynamical_system.evaluate(position=position)

        norm_dot = (np.dot(velocity, position))/(LA.norm(velocity)*LA.norm(position))
        self.assertTrue(np.isclose(norm_dot, -1))

        # Test at center
        dynamical_system = LocallyRotated(
            max_rotation=np.array([pi*3/4]),
            influence_pose=ObjectPose(position=np.array([0.1, 0.1])),
            influence_radius=1,
            )
        
        position = np.array([0.1, 0.1])
        velocity = dynamical_system.evaluate(position=position)
        norm_dot = (np.dot(velocity, position))/(LA.norm(velocity)*LA.norm(position))
        breakpoint()
        self.assertTrue(np.isclose(norm_dot, np.cos(dynamical_system.max_rotation)))

        position = np.array([0.0, 0.0])
        velocity = dynamical_system.evaluate(position=position)
        norm_dot = (np.dot(velocity, position))/(LA.norm(velocity)*LA.norm(position))
        self.assertTrue(norm_dot > 0)
        
        
    def plot_ds_around_obstacle(self):
        from dynamic_obstacle_avoidance.obstacles import Ellipse
        obs = Ellipse(
            center_position=np.array([-8, 0]), 
            axes_length=np.array([3, 1]),
            orientation=10./180*pi,
        )

        dynamical_system = LocallyRotated(max_rotation=[-np.pi/2]).from_ellipse(obs)

        plot_dynamical_system_quiver(DynamicalSystem=dynamical_system,
                                     n_resolution=20)
        self.visualize_weight(dynamical_system)
        
        
    def plot_dynamical_system(self):
        DynamicalSystem = LocallyRotated(
            max_rotation=[-np.pi/2],
            # mean_rotation=[np.pi],
            influence_pose=ObjectPose(position=np.array([5, 2])),
            influence_radius=3)
        
        plot_dynamical_system_quiver(DynamicalSystem=DynamicalSystem,
                                     n_resolution=20)

        self.visualize_weight(DynamicalSystem)
        

    def plot_critical_ds(self):
        DynamicalSystem = LocallyRotated(
            mean_rotation=[np.pi],
            rotation_center=[4, 2],
            influence_radius=4)
        
        plot_dynamical_system_quiver(DynamicalSystem=DynamicalSystem,
                                     n_resolution=20)
        
    def visualize_weight(self, DynamicalSystem=None, x_lim=[-10, 10], y_lim=[-10, 10], dim=2):
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

if (__name__) == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    manual_tets = False
    if manual_tets:
        Tester = TestSpiralmotion()
        Tester.plot_dynamical_system()
        # Tester.plot_critical_ds()
        # Tester.visualize_weight()
        # Tester.plot_ds_around_obstacle()
    
print('Done')



