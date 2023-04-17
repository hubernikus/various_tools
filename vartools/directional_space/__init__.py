"""
The 'direction_space' module allows representing code as part of the direction space.
"""
# Various Dynamical Systems
from .unit_direction import get_angle_from_vector, get_vector_from_angle
from .unit_direction import UnitDirection

# from .unit_direction import DirectionBase

from .directional_space import get_angle_space
from .directional_space import get_angle_space_inverse

from .directional_space import get_directional_weighted_sum
from .directional_space import get_directional_weighted_sum_from_unit_directions

from .directional_space import get_angle_space_of_array
from .directional_space import get_angle_space_inverse_of_array

from .addition import fast_directional_vector_addition
from .addition import fast_directional_unit_vector_addition

__all__ = [
    "get_angle_from_vector",
    "get_vector_from_angle",
    "UnitDirection",
    # "DirectionBase",
    "get_angle_space",
    "get_angle_space_inverse",
    "get_directional_weighted_sum",
    "get_directional_weighted_sum_from_unit_directions",
    "get_angle_space_of_array",
    "get_angle_space_inverse_of_array",
    "fast_directional_vector_addition",
    "fast_directional_unit_vector_addition",
]
