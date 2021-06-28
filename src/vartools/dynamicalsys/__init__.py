"""
The :mod:`DynamicalSystem` module implements mixture modeling algorithms.
"""
# Various Dynamical Systems
from ._base import allow_max_velocity, DynamicalSystem
from .linear import LinearSystem
from .locally_rotated import LocallyRotated
from .plot_vectorfield import plot_dynamical_system_quiver


__all__ = ['allow_max_velocity',
           'DynamicalSystem',
           'LinearSystem',
           'ConstantValue',
           'LocallyRotated',
           'plot_dynamical_system_quiver',
           ]
