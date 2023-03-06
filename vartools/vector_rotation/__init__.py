"""
The :mod:`rotation_vector` with various types of base-dynamics.
"""

# Various Obstacle Descriptions
from .single_rotation import VectorRotationXd
from .single_rotation import rotate_direction

__all__ = [
    "VectorRotationXd",
    "rotate_direction",
]
