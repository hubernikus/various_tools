"""
The `direction_space` module allows representing code as part of the direction space.
"""
# Various Dynamical Systems
from .directional_space import get_angle_from_vector, get_vector_from_angle
from .directional_space import UnitDirection

from .directional_space import get_angle_space_inverse
from .directional_space import get_angle_space
from .directional_space import get_directional_weighted_sum

from .directional_space import get_angle_space_of_array
from .directional_space import get_angle_space_inverse_of_array


__all__ = ['get_angle_from_vector',
           'get_vector_from_angle',
           'UnitDirection',
           'get_angle_space',
           'get_angle_space_inverse',
           'get_directional_weighted_sum',
           'get_angle_space_of_array',
           'get_angle_space_inverse_of_array',
           ]
