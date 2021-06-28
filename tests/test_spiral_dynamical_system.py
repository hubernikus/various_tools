#!/USSR/bin/python3.9
""" Sample DS spiral-trajectories."""
import unittest

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from vartools.dynamicalsys import SpiralStable

class TestSpiralmotion(unittest.TestCase):
    def test_creation(self):
        # Complexity of the spiral
        complexity_spiral = 15
        
        # Base spiral
        SpiralDS = SpiralStable(c_factor=15, p_radius_control=1)
                                       
        dataset_analytic = SpiralDS.get_positions(n_points=500)

        dt = 0.0005
        # start_position = SpiralDS.get_positions(n_points=1, tt=[0.001])[0, :]
        start_position = np.array([1, 1, 1.5])
        dataset_ds = SpiralDS.motion_integration(start_position, dt).T
        
        # dataset_ds = np.array(spiral_motion_integrator(start_position, dt,
                                                       # complexity_spiral, end_point))
                                                       
        fig = plt.figure("Figure: c = "+str(complexity_spiral))
        ax = fig.add_subplot(1,2,1, projection='3d')
        ax.plot(dataset_analytic[:,0], dataset_analytic[:,1], dataset_analytic[:,2],'b')
        ax.scatter(dataset_analytic[0,0], dataset_analytic[0,1],
                   dataset_analytic[0,2],color='black',marker='o',label='Start')
        ax.scatter(dataset_analytic[-1,0], dataset_analytic[-1,1],
                   dataset_analytic[-1,2],color='black',marker='s',label='End')
        
        ax.set_xlabel('X(m)', fontsize=18, labelpad=15)
        ax.set_ylabel('Y(m)', fontsize=18, labelpad=15)
        ax.set_zlabel('Z(m)', fontsize=18, labelpad=15)
        ax.legend(fontsize=18)
        ax.set_title('Demonstration', fontsize=20, pad=20)

        ax_ds = fig.add_subplot(1,2,2, projection='3d')
        ax_ds.plot(dataset_ds[:,0], dataset_ds[:,1], dataset_ds[:,2],'r--')
        ax_ds.set_xlabel('X(m)', fontsize=18,labelpad=15)
        ax_ds.set_ylabel('Y(m)', fontsize=18,labelpad=15)
        ax_ds.set_zlabel('Z(m)', fontsize=18,labelpad=15)
        ax_ds.set_title('Spiral DS', fontsize=20, pad=20)
        
        plt.ion()
        plt.show()
        
if __name__ == '__main__':
    # unittest.main()
    manual_tets = True
    if manual_tets:
        Tester = TestSpiralmotion()
        Tester.test_creation()
    
print('Done')

