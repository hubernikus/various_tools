""" Spiral shaped dynamical system in 3D. """
# Author: Sthith
#         Lukas Huber
# Created: 2021-06-23
# License: BSD (c) 2021

import numpy as np
import sympy

def spiral(c, T):
    x = np.sin(T)*np.cos(c*T)
    y = np.sin(T)*np.sin(c*T)
    z = np.cos(T)
    return [x,y,z]

def spiral_pos(c, N, dim):
    dataSet = np.zeros((N, dim))
    T = np.linspace(0, np.pi, N)
    dataSet[:,0] = np.sin(T)*np.cos(c*T)
    dataSet[:,1] = np.sin(T)*np.sin(c*T)
    dataSet[:,2] = np.cos(T)
    
    return dataSet

def get_symbolic_expression(c, symbolX):
    xDotExpr = sympy.expand_trig(sympy.cos(c*symbolX))
    yDotExpr = sympy.expand_trig(sympy.sin(c*symbolX))
    return [xDotExpr, yDotExpr]


def spiral_motion(position, dt, c, attractor):
    """ Return the velocity based on the evaluation of a spiral-shaped dynamical system."""
    velocity = np.zeros(position.shape)

    theta = np.arccos(velocity[2])
    velocity[2] = -np.sqrt(1-position[2]**2)
    velocity[0] = position[2]*np.cos(c*theta) - c*position[1]
    velocity[1] = position[2]*np.sin(c*theta) + c*position[0]

    return velocity

def spiral_motion_integrator(startPoint, dt, c, symbolX, velExpr, attractor):
    [x,y,z] = startPoint
    print("C:",c)
    dataSet = [[x,y,z]]
    goal = np.array(attractor).astype(float)
    current = np.array(startPoint)
    
    while np.linalg.norm(current-attractor) > 1e-5 and z > -1:
        zDot = -np.sqrt(1-z**2)
        xDot = z*velExpr[0].subs([(sympy.cos(symbolX),z), (sympy.sin(symbolX),-zDot)]) - c*y
        yDot = z*velExpr[1].subs([(sympy.cos(symbolX),z), (sympy.sin(symbolX),-zDot)]) + c*x

        vel = spiral_motion(np.array([x, y, z]), dt, c, attractor)
        
        breakpoint()
        x = x + xDot*dt
        y = y + yDot*dt
        z = z + zDot*dt

        print(x, y, z)

        dataSet = dataSet + [[x,y,z]]
        current = np.array([x,y,z]).astype(float)

    return dataSet
