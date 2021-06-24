#!/USSR/bin/python3.9
""" Sample DS spiral-trajectories."""
import unittest

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from vartools.dynamicalsys.spiral_motion import spiral_motion_integrator, get_symbolic_expression
from vartools.dynamicalsys.spiral_motion import spiral_pos, spiral

import sympy

class TestSpiralmotion(unittest.TestCase):
    def test_creation(self):
        # Terminal for the spiral DS
        terminalPoint = [0,0,-1]

        N = 500     # Points in the base spiral
        c = 15      # Complexity of the spiral
        dT = 0.0005 # Used during forward orbit construction

        # Starting for a forward orbit of the spiral DS (immediate vicinity of [0,0,1])
        startPoint = spiral(c, 0.001)

        # Base spiral
        dataSetOrig = spiral_pos(c, N, 3)

        # Forward orbits of the spiral DS
        theta = sympy.symbols('theta', real=True)
        velExpr = get_symbolic_expression(c, theta)
        print("here")
        dataSetDS = np.array(spiral_motion_integrator(
            startPoint, dT, c, theta, velExpr, terminalPoint))

        fig = plt.figure("c = "+str(c))
        ax = fig.add_subplot(1,2,1, projection='3d')
        ax.plot(dataSetOrig[:,0], dataSetOrig[:,1], dataSetOrig[:,2],'b')
        ax.scatter(dataSetOrig[0,0], dataSetOrig[0,1],
                   dataSetOrig[0,2],color='black',marker='o',label='Start')
        ax.scatter(dataSetOrig[-1,0], dataSetOrig[-1,1],
                   dataSetOrig[-1,2],color='black',marker='s',label='End')
        
        ax.set_xlabel('X(m)', fontsize=18, labelpad=15)
        ax.set_ylabel('Y(m)', fontsize=18, labelpad=15)
        ax.set_zlabel('Z(m)', fontsize=18, labelpad=15)
        ax.legend(fontsize=18)
        ax.set_title('Demonstration', fontsize=20, pad=20)

        axDS = fig.add_subplot(1,2,2, projection='3d')
        axDS.plot(dataSetDS[:,0], dataSetDS[:,1], dataSetDS[:,2],'r--')
        axDS.set_xlabel('X(m)', fontsize=18,labelpad=15)
        axDS.set_ylabel('Y(m)', fontsize=18,labelpad=15)
        axDS.set_zlabel('Z(m)', fontsize=18,labelpad=15)
        axDS.set_title('Spiral DS', fontsize=20, pad=20)
        
        plt.show()

if __name__ == '__main__':
    # unittest.main()

    manual_tets = True
    if manual_tets:
        Tester = TestSpiralmotion()
        Tester.test_creation()
    
print('Done')
