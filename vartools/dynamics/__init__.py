"""
The :mod:`DynamicalSystem` module implements mixture modeling algorithms.
"""
# TODO: this directory is expected to replace the dynamical_systems directory
# Various Dynamical Systems

from ..dynamical_systems._base import allow_max_velocity, Dynamics 
from ..dynamical_systems._base import Dynamics as DynamicalSystem
from ..dynamical_systems.linear import LinearSystem, ConstantValue
from ..dynamical_systems.circle_stable import CircularStable
from ..dynamical_systems.circular_and_linear import CircularLinear
from ..dynamical_systems.spiral_motion import SpiralStable
from ..dynamical_systems.locally_rotated import LocallyRotated
from ..dynamical_systems.quadratic_axis_convergence import QuadraticAxisConvergence, AxesFollowingDynamics
from ..dynamical_systems.multiattractor_dynamics import (
    PendulumDynamics,
    DuffingOscillator,
    BifurcationSpiral,
)
from ..dynamical_systems.sinus_attractor import SinusAttractorSystem

# Various Dynamical System Adaptation Functions
from ..dynamical_systems.velocity_trimmer import BaseTrimmer, ConstVelocityDecreasingAtAttractor

# Helper functions for visualization
from ..dynamical_systems.plot_vectorfield import plot_dynamical_system
from ..dynamical_systems.plot_vectorfield import plot_dynamical_system_quiver
from ..dynamical_systems.plot_vectorfield import plot_dynamical_system_streamplot


__all__ = [
    "allow_max_velocity",
    "DynamicalSystem",
    "Dynamics",
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
