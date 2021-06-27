"""
The :mod:`DynamicalSystem` module implements mixture modeling algorithms.
"""
# Various modulation functions

from ._base import allow_max_velocity, DynamicalSystem
from .locally_rotated import LocallyRotated
from .plot_vectorfield import plot_dynamical_system_quiver

__all__ = ['allow_max_velocity',
           'DynamicalSystem',
           'LocallyRotated',
           'plot_dynamical_system_quiver',
           ]
