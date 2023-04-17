"""
Allows for fast and efficient addition similar to directional summing
"""
import numpy as np


def fast_directional_unit_vector_addition(
    vector1: np.ndarray,
    vector2: np.ndarray,
    weight: float = 1.0,
    normalize_input: bool = True,
):
    """The weight is added on vector2"""
    if normalize_input:
        vector1 = vector1 / linalg.norm(vector2)
        vector2 = vector2 / linalg.norm(vector2)

    vector_out = (1 - weight) * vector1 + weight * vector2
    if not (vec_norm := linalg.norm(vector_out)):
        raise ValueError(
            "Vector are opposing each other - directional summing not possible!"
        )
    return vector_out / vec_norm


def fast_directional_vector_addition(
    vector1: np.ndarray,
    vector2: np.ndarray,
    weight: float = 1.0,
    normalize_input: bool = True,
):
    """The weight is added on vector2"""
    if not (norm1 := linalg.norm(vector1)):
        return vector2 * weight

    if not (norm2 := linalg.norm(vector2)):
        return vector1 * (1 - weight)

    vector1 = vector1 / norm1
    vector2 = vector2 / norm2

    vector_out = (1 - weight) * vector1 + weight * vector2
    if not (vec_norm := linalg.norm(vector_out)):
        raise ValueError(
            "Vector are opposing each other - directional summing not possible!"
        )
    return vector_out / vec_norm * ((1 - weight) * norm1 + weight * norm2)
