"""
The :mod:`DynamicalSystem` module implements mixture modeling algorithms.
"""
# Various Dynamical Systems
from ._base import allow_max_velocity, DynamicalSystem
from .linear import LinearSystem, ConstantValue
from .circle_stable import CircularStable
from .locally_rotated import LocallyRotated
from .quadratic_axis_convergence import QuadraticAxisConvergence

# Helper functions for visualization
from .plot_vectorfield import plot_dynamical_system_quiver


__all__ = ['allow_max_velocity',
           'DynamicalSystem',
           'LinearSystem',
           'ConstantValue',
           'CircularStable'
           'LocallyRotated',
           'QuadraticAxisConvergence',
           'plot_dynamical_system_quiver',
           ]
