"""
The :mod:`DynamicalSystem` module implements mixture modeling algorithms.
"""
# Various Dynamical Systems
from ._base import allow_max_velocity, DynamicalSystem
from .linear import LinearSystem, ConstantValue
from .circle_stable import CircularStable
from .circular_and_linear import CircularLinear
from .spiral_motion import SpiralStable
from .locally_rotated import LocallyRotated
from .quadratic_axis_convergence import QuadraticAxisConvergence, AxesFollowingDynamics
from .multiattractor_dynamics import (
    PendulumDynamics,
    DuffingOscillator,
    BifurcationSpiral,
)
from .sinus_attractor import SinusAttractorSystem

# Various Dynamical System Adaptation Functions
from .velocity_trimmer import BaseTrimmer, ConstVelocityDecreasingAtAttractor

# Helper functions for visualization
from .plot_vectorfield import plot_dynamical_system
from .plot_vectorfield import plot_dynamical_system_quiver
from .plot_vectorfield import plot_dynamical_system_streamplot


__all__ = [
    "allow_max_velocity",
    "DynamicalSystem",
    "LinearSystem",
    "ConstantValue",
    "CircularStable" "SpiralStable",
    "LocallyRotated",
    "QuadraticAxisConvergence",
    "AxesFollowingDynamics",
    "PendulumDynamics",
    "DuffingOscillator",
    "BifurcationSpiral",
    "SinusAttractorSystem",
    "BaseTrimmer",
    "ConstVelocityDecreasingAtAttractor",
    "plot_dynamical_system",
    "plot_dynamical_system_quiver",
    "plot_dynamical_system_streamplot",
]
