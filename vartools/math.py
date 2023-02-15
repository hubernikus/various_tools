""" Various tools to help / and speed up."""
from .linalg import get_orthogonal_basis

from enum import Enum, auto

from typing import Callable, Optional
import warnings

import numpy as np
from numpy import linalg as LA
import numpy.typing as npt

Vector = npt.ArrayLike


def get_numerical_gradient_of_vectorfield(
    position: np.ndarray,
    function: Callable[[np.ndarray], np.ndarray],
    delta_magnitude: float = 1e-6,
) -> np.ndarray:
    dimension = position.shape[0]
    dim_out = function(position).shape[0]

    tensor_low = np.zeros((dimension, dim_out))
    tensor_high = np.zeros((dimension, dim_out))

    for ii in range(dimension):
        delta_vec = np.zeros(dimension)
        delta_vec[ii] = delta_magnitude / 2.0

        tensor_low[ii, :] = function(position - delta_vec)
        tensor_high[ii, :] = function(position + delta_vec)

    return (tensor_high - tensor_low) / delta_magnitude


def get_numerical_gradient(
    position: np.ndarray,
    function: Callable[[np.ndarray], float],
    delta_magnitude: float = 1e-6,
) -> np.ndarray:
    """Returns the numerical derivative of an input function at the specified position."""
    dimension = position.shape[0]

    vec_low = np.zeros(dimension)
    vec_high = np.zeros(dimension)
    for ii in range(dimension):
        delta_vec = np.zeros(dimension)
        delta_vec[ii] = delta_magnitude / 2.0

        vec_low[ii] = function(position - delta_vec)
        vec_high[ii] = function(position + delta_vec)
    return (vec_high - vec_low) / delta_magnitude


def get_numerical_hessian_fast(
    position: np.ndarray,
    function,
    # function: Callable[[np.ndarray], float],
    delta_magnitude: float = 1e-6,
) -> np.ndarray:
    """Returns (numerical) Hessian-Matrix of 'function' at 'position'.
    Calculation is speed up, by going in positive-delta direction only."""

    dimension = position.shape[0]

    hessian = np.zeros((dimension, dimension))
    f_eval = function(position)
    f_dx_eval = np.zeros(dimension)

    pos_deltas = np.eye(dimension) * delta_magnitude

    for ix in range(dimension):
        f_dx_eval[ix] = function(position + pos_deltas[:, ix])
        for iy in range(0, ix + 1):
            f_dx_dx_eval = function(position + pos_deltas[:, ix] + pos_deltas[:, iy])

            hessian[ix, iy] = (
                f_dx_dx_eval - f_dx_eval[ix] - f_dx_eval[iy] + f_eval
            ) / (delta_magnitude * delta_magnitude)
            if ix != iy:
                hessian[iy, ix] = hessian[ix, iy]
    return hessian


def get_numerical_hessian(
    position: np.ndarray,
    function,
    # function: Callable[[np.ndarray], float],
    delta_magnitude: float = 1e-6,
) -> np.ndarray:
    """Returns (numerical) Hessian-Matrix of 'function' at 'position'.
    Calculation is centered around position."""
    dimension = position.shape[0]

    hessian = np.zeros((dimension, dimension))
    pos_deltas = np.eye(dimension) * (delta_magnitude * 0.5)
    for ix in range(dimension):
        for iy in range(0, ix + 1):
            f_dx_dy_eval = function(position + pos_deltas[:, ix] + pos_deltas[:, iy])
            f_ndx_dy_eval = function(position - pos_deltas[:, ix] + pos_deltas[:, iy])
            f_dx_ndy_eval = function(position + pos_deltas[:, ix] - pos_deltas[:, iy])
            f_ndx_ndy_eval = function(position - pos_deltas[:, ix] - pos_deltas[:, iy])

            hessian[ix, iy] = (
                f_dx_dy_eval - f_ndx_dy_eval - f_dx_ndy_eval + f_ndx_ndy_eval
            ) / (delta_magnitude * delta_magnitude)
            if ix != iy:
                hessian[iy, ix] = hessian[ix, iy]
    return hessian


def get_scaled_orthogonal_projection(vector):
    """Returns scaled orthogonal projection of the form  P_v = ||v||^2 I_n − v v^T.

    It has following properties:
    (i) P_v = P_v^T (symmetry)
    (ii) P_v 2 = ||v||2 P_v
    (iii) the spectrum of P_v is composed of 0 and ||v||^2
          with algebraic multiplicity 1 and n − 1, respectively
    (iv) P_v z = ||v||^2 z for all z ∈ R n on the projective subspace defined by v∈R_n
    (v) P_v w = 0 for all w ∈ R n such that vw;
    (vi) 12 w^T Ṗ_v w = v^T P_w v̇."""
    return LA.norm(vector) * np.eye(vector.shape[0]) - vector.reshape(
        -1, 1
    ) @ vector.reshape(1, -1)


class IntersectionType(Enum):
    CLOSE = auto()
    FAR = auto()
    BOTH = auto()


CircleIntersectionType = IntersectionType


def get_intersection_with_circle(
    start_position: np.ndarray,
    direction: np.ndarray,
    radius: float,
    only_positive: Optional[bool] = None,
    intersection_type: CircleIntersectionType = CircleIntersectionType.FAR,
) -> Optional[np.ndarray]:
    """Returns intersection with circle with center at 0
    of of radius 'radius' and the line defined as 'start_position + x * direction'

    If 'only_positive=True', then only intersection at furthest distance
    to start_point is returned.
    """
    if not radius:  # Zero radius
        return None

    if only_positive is not None:
        warnings.warn("Remove only_positive")
        # Make depreciated argument work
        if only_positive:
            intersection_type = CircleIntersectionType.FAR
        else:
            intersection_type = CircleIntersectionType.BOTH

    # Binomial Formula to solve for x in:
    # || dir_reference + x * (delta_dir_conv) || = radius
    AA = np.sum(direction**2)
    BB = 2 * np.dot(direction, start_position)
    CC = np.sum(start_position**2) - radius**2
    DD = BB**2 - 4 * AA * CC

    if DD < 0:
        # No intersection with circle
        return None

    if intersection_type == CircleIntersectionType.FAR:
        # Only negative direction due to expected negative A (?!) [returns max-direction]..
        fac_direction = (-BB + np.sqrt(DD)) / (2 * AA)
        point = start_position + fac_direction * direction
        return point

    elif intersection_type == CircleIntersectionType.CLOSE:
        # Only negative direction due to expected negative A (?!) [returns max-direction]..
        fac_direction = (-BB - np.sqrt(DD)) / (2 * AA)
        point = start_position + fac_direction * direction
        return point

    elif intersection_type == CircleIntersectionType.BOTH:
        factors = (-BB + np.array([-1, 1]) * np.sqrt(DD)) / (2 * AA)
        points = (
            np.tile(start_position, (2, 1)).T
            + np.tile(factors, (start_position.shape[0], 1))
            * np.tile(direction, (2, 1)).T
        )
        return points

    else:
        raise ValueError()


def get_intersection_between_line_and_plane(
    line_position: Vector,
    line_direction: Vector,
    plane_position: Vector,
    plane_normal: np.ndarray,
    positive_only: bool = False,
) -> Vector:
    """Returns the intersection position of a plane and a point."""
    basis = get_orthogonal_basis(plane_normal)

    if not np.dot(line_direction, basis[:, 0]):
        warnings.warn("Plan is parallel to line.")
        if np.dot((line_position - plane_position), plane_normal):
            raise ValueError("No intersection possible.")

        return line_position

    basis[:, 0] = (-1) * line_direction
    factors = LA.pinv(basis) @ (line_position - plane_position)

    if positive_only and factors[0] < 0:
        return None

    return line_position + line_direction * factors[0]
