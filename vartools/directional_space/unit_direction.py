"""
Directional Space function to use
Helper function for directional & angle evaluations
"""
# Author: LukasHuber
# Created: 2021-05-18
# Email: lukas.huber@epfl.ch

# Use python 3.10 [annotations / typematching]
from __future__ import annotations  # Not needed from python 3.10 onwards

import warnings
import copy

# from typing import Callable
from math import pi

import numpy as np
from numpy import linalg as LA

from vartools.linalg import get_orthogonal_basis


class UnitDirectionError(Exception):
    def __init__(self, message="Error with Unit Direction Handling"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"


class InversionError(UnitDirectionError):
    def __init__(self, value):
        self.value = value
        super().__init__()

    def __str__(self):
        return (
            f"angle_norm={self.value} "
            + "-> Inversion not possible, norm should be in [0, pi["
        )


class ZeroVectorError(Exception):
    def __str__(self):
        return f"Zero vector is passed. No angle-space transformation possible."


class DirectionBaseError(Exception):
    """Base class for exceptions."""

    pass


class NonEqualBaseError(DirectionBaseError):
    """Raised when inconsistency in base.
    Attributes
    ----------
    """

    def __init__(self, message="Direction Base is not equal."):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return f"{self.message}"


def get_angle_from_vector(
    direction: np.ndarray,
    # base: DirectionBase,
    base: np.ndarray,
    cos_margin: float = 1e-8,
) -> np.ndarray:
    """
    Returns a angle evalauted from the direciton & null_matrix

    Parameters
    ---------
    vector: unit vector of dimension (dim,)
    base : null-matrix-base of floats of dimension (dim, dim)

    Returns
    ------
    angle : angle-space value of dimension (dim-1,)
    """
    # if not np.isclose(LA.norm(direction), 1):  # Normalize
    #     raise ValueError("Not unit vector.")

    if not (norm_dir := LA.norm(direction)):  # Normalize
        raise ValueError("Undefined vector.")

    direction = direction / norm_dir

    # direction_referenceSpace = base.null_matrix.T.dot(direction)
    direction_referenceSpace = base.T.dot(direction)
    # direction_referenceSpace = base.dot(direction)

    # Make sure to catch numerical error of cosinus calculation
    cos_direction = direction_referenceSpace[0]

    if cos_direction >= (1.0 - cos_margin):
        # Trivial solution
        angle = np.zeros(direction_referenceSpace.shape[0] - 1)
        return angle

    elif cos_direction <= -(1.0 - cos_margin):
        # This value has to be used with care, since it's close to signularity.
        # but because transformation can be used to evaluate the total angle
        # no 'warning' is raised.
        angle = np.zeros(direction_referenceSpace.shape[0] - 1)
        angle[0] = pi
        return angle

    angle = direction_referenceSpace[1:]
    # No zero-check since non-trivial through previous one.
    # angle = (angle / LA.norm(self._angle))
    angle = angle / LA.norm(angle)
    angle = angle * np.arccos(cos_direction)
    return angle


def get_vector_from_angle(
    angle: np.ndarray,
    base: np.ndarray
    # base: DirectionBase
) -> np.ndarray:
    """
    Returns a unit vector transformed back from the angle/direction-space.

    Parameters
    ---------
    angle : angle-space value of dimension (dim-1,)
    base : null-matrix-base of floats of dimension (dim, dim)

    Returns
    ------
    vector: unit vector of dimension (dim,)
    """
    norm_angle = LA.norm(angle)
    if norm_angle:
        vector = base.dot(
            np.hstack((np.cos(norm_angle), np.sin(norm_angle) * angle / norm_angle))
        )
    else:
        vector = base[:, 0]
    return vector


class UnitDirection(object):
    """Direction of the length 1 which can be respresented in angle space.
    Not that this space is not Eucledian but it

    Properties
    ----------
    _vector : Each element in this space can be equivalently described as a unit-vector
    _angle : The transformation of the angle space
    """

    def __init__(
        self,
        base: np.ndarray = None,
        # base: DirectionBase = None,
        unit_direction: UnitDirection = None,
    ):
        """
        To create the angle space on of several 'reference angles / directions' have to be
        pass to the function.

        Parameters
        ----------
        base: DirectionBase [base is not copied]
        unit_direction = UnitDirection [base is copied]
        """
        if base is not None:
            if len(base.shape) == 1:
                # Vector needs to be transformed into base
                self.base = get_orthogonal_basis(base)
            else:
                self.base = base

        elif unit_direction is not None:
            # self.base = copy.deepcopy(self.unit_direction.base)
            self.base = self.unit_direction.base
            # TODO: depreciated
        else:
            raise ValueError("No direction argument is given.")

        self._angle = None
        self._vector = None

    @staticmethod
    def from_unit_direction(cls, unit_direction: UnitDirection) -> None:
        self = cls(base=unit_direction.base).from_angle(unit_direction.angle)
        return self

    def from_angle(self, value: (np.ndarray, list)) -> np.ndarray:
        """Update angle and reset 'equivalent' vector."""
        self._angle = np.array(value)
        self._vector = None
        return self

    def from_vector(self, value: (np.ndarray, list)) -> None:
        """Update (and normalize) vector and reset angle."""
        value = np.array(value)
        value_norm = LA.norm(value)
        if not value_norm:  # zero value
            raise ZeroVectorError()

        self._vector = value / value_norm
        self._angle = None
        return self

    @property
    def dimension(self) -> int:
        # return self.base.null_matrix.shape[0]
        return self._base.shape[0]

    @property
    def magnitude(self) -> float:
        return LA.norm(self.angle)

    @property
    def null_matrix(self) -> np.ndarray:
        return self._base

    @property
    def base_matrix(self) -> np.ndarray:
        return self._base.null_matrix

    @property
    def base(self) -> np.ndarray:
        return self._base

    @base.setter
    def base(self, value: np.ndarray) -> None:
        if hasattr(self, "_base"):
            # Reset angles / vector
            self._angle = None
            self._vector = None
        self._base = value

    def __repr__(self):
        return f"UnitDirection({str(self.as_angle())}) \n" f"{str(self.base)}"

    def __eq__(self, other: UnitDirection) -> float:
        return np.allclose(self.base, other.base) and np.allclose(
            self.as_angle(), other.as_angle()
        )

    def __neq__(self, other: UnitDirection) -> float:
        return not (self == other)

    def __add__(self, other: UnitDirection) -> UnitDirection:
        if not np.allclose(self.base, other.base):
            raise ValueError("Bases are not equal.")
        return UnitDirection(self.base).from_angle(self.as_angle() + other.as_angle())

    def __radd__(self, other: UnitDirection) -> UnitDirection:
        return self + other

    def __sub__(self, other: UnitDirection) -> UnitDirection:
        return self + (-1) * other

    def __mul__(self, other: float) -> UnitDirection:
        return UnitDirection(self.base).from_angle(self.as_angle() * other)

    def __rmul__(self, other: UnitDirection) -> UnitDirection:
        return self * other

    def __truediv__(self, other: float) -> UnitDirection:
        return self * (1.0 / other)

    # def __rdiv__(self, other: float) -> UnitDirection:
    #     return self * other

    def norm(self) -> float:
        """Return norm of angle."""
        return LA.norm(self.as_angle())

    def get_distance_to(self, other: UnitDirection) -> float:
        if not np.allclose(self.base, other.base):
            raise NonEqualBaseError()
        return LA.norm(self.as_angle() - other.as_angle())

    def invert_normal(self) -> UnitDirection:
        """Invert the normal of the unit vector"""
        angle_norm = self.norm()
        if angle_norm > pi:  # Strictly bigger, since (.)=pi is projected to the center
            raise InversionError(value=self.norm())
        elif not angle_norm:  # at center
            new_angle = np.zeros(self._angle.shape)
            new_angle[0] = pi
        else:
            new_angle = self.as_angle() / angle_norm * (pi - angle_norm)

        new_base = (-1) * self.base

        return UnitDirection(base=new_base).from_angle(new_angle)

    def _get_unitdirection_relative_to_angle(self, new_base_angle):
        """Evaluate  a new 'basis' that allows rotational transferring."""
        null_matrix = self.null_matrix

        if not LA.norm(new_base_angle):
            # Same base vector -> no transformation needed
            return copy.deepcopy(self)

        if not (LA.norm(new_base_angle) - np.pi) % (2 * np.pi):
            raise ValueError("Normal transform is not defined for an angle of 'pi'.")

        # Create basis in the directional space (without base-vector)
        directional_basis = get_orthogonal_basis(new_base_angle)

        new_null_matrix = np.zeros((self.dimension, self.dimension))
        new_null_matrix[:, 0] = get_vector_from_angle(
            angle=new_base_angle,
            base=null_matrix,
        )
        # Get the directions of the tangents
        for ii in range(1, self.dimension - 1):
            new_null_matrix[:, ii] = get_vector_from_angle(
                np.pi * 0.5 * directional_basis[ii, :], null_matrix
            )

        # Get last tangent
        dot_prod = np.dot(new_null_matrix[:, 0], null_matrix[:, 0])
        proj_base_vector = new_null_matrix[:, 0] * np.abs(dot_prod)
        tangent = null_matrix[:, 0] * np.copysign(1, dot_prod) - proj_base_vector
        new_null_matrix[:, -1] = tangent / LA.norm(tangent)

        # Rebase the angle to new angle-space
        new_angle = new_base_angle - self.as_angle()
        new_angle = directional_basis.T @ new_angle

        return UnitDirection(new_null_matrix).from_angle(new_angle)

    def project_onto_sphere(
        self, reference_vector: UnitDirection, radius=pi / 2
    ) -> UnitDirection:
        """Project onto sphere"""
        raise NotImplementedError()

    def get_shortest_angle(self, other):
        """Get shortesst angle distance between points."""
        pass

    def as_angle(self, cos_margin: float = 1e-5) -> np.ndarray:
        if self._angle is not None:
            return self._angle
        if self._vector is None:
            raise ValueError("Set vector or angle value before evaluating.")

        self._angle = get_angle_from_vector(
            direction=self._vector, base=self.base, cos_margin=cos_margin
        )
        return self._angle

    def as_vector(self) -> np.ndarray:
        if self._vector is not None:
            return self._vector
        if self._angle is None:
            raise ValueError("Set vector or angle value before evaluating.")

        self._vector = get_vector_from_angle(angle=self._angle, base=self.base)
        return self._vector

    def transform_to_base(self, new_base: np.ndarray) -> None:
        """Rebase to new base, and evaluate the 'self.angle' with respect to the center of
        the (new) base."""
        if np.allclose(self.base, new_base):
            return copy.deepcopy(self)

        # Make sure the angle is calculated
        angle = self.as_angle()
        vector = self.as_vector()

        angle_in_newbase = get_angle_from_vector(vector, base=new_base)

        normal_in_newbase = get_angle_from_vector(self.base[0], base=new_base)

        direction_of_windup_check = np.dot(angle_in_newbase, normal_in_newbase)
        dist_angle_to_normal = LA.norm(angle_in_newbase - normal_in_newbase)

        angle_norm = LA.norm(angle)
        if (
            not direction_of_windup_check
            and (dist_angle_to_normal - angle_norm) > pi / 2
        ):
            angle_norm = LA.norm(angle_in_newbase)
            if angle_norm:
                unit_angle = angle_in_newbase / angle_norm
            else:
                unit_angle = normal_in_newbase / LA.norm(normal_in_newbase)

            windup_max = 3  # Define maximum windup

            # Try positive
            it_wind = 1
            while True:
                new_angle_in_newbase = unit_angle * (
                    angle_norm + 2 * pi * it_wind * direction_of_windup_check
                )
                new_dist_angle_to_normal = LA.norm(
                    new_angle_in_newbase - normal_in_newbase
                )

                if new_dist_angle_to_normal < dist_angle_to_normal:
                    angle_in_newbase = new_angle_in_newbase
                else:
                    break

                if it_wind > windup_max:
                    warnings.warn("Maximum windup reached.")
                    break
                it_wind += 1

        return UnitDirection(new_base).from_angle(angle_in_newbase)

    def transform_to_base_old_try2(self, new_base: np.ndarray) -> None:
        """Rebase to new base, and evaluate the 'self.angle'
        with respect to the center of the (new) base."""
        if self.base == new_base:
            return copy.deepcopy(self)

        # Make sure the angle is calculated
        angle = self.as_angle()

        dim = self.dimension
        # Transform the base into the new_space
        new_base_as_angle = np.zeros((dim - 1, dim))
        new_base_vecs = np.zeros((dim - 1, dim - 1))
        for ii in range(dim):
            new_base_as_angle[:, ii] = get_angle_from_vector(
                new_base[ii], base=self.base
            )

            # Do the 'proximity check' (pi-discontuinity) for the vectors
            # by comparing each vector the the normal of the original base.
            if ii > 0:
                base_norm = LA.norm(new_base_as_angle[:, ii] - new_base_as_angle[:, 0])
                if base_norm >= pi:
                    warnings.warn("TODO: Throughfully test this case...")
                    dist_new_base = LA.norm(
                        new_base_as_angle[:, ii] - new_base_as_angle[:, 0]
                    )

                    # Project accross origin
                    new_base_opposite_angle = (
                        new_base_as_angle[:, ii] / base_norm * (base_norm - 2 * pi)
                    )

                    dist_new_base_opposite = LA.norm(
                        new_base_opposite_angle - new_base_as_angle[:, 0]
                    )

                    if dist_new_base_opposite < dist_new_base:
                        warnings.warn(
                            "Did a transform. Is this the only case where it happens?"
                        )
                        new_base_as_angle[:, ii] = new_base_opposite_angle

            # Don't normalize, in order to account for 'direction'
            new_base_vecs[:, ii - 1] = (
                new_base_as_angle[:, ii] - new_base_as_angle[:, 0]
            )

        # nullvector_in_newbase = get_angle_from_vector(
        # new_base[0], base=self.base)

        # Normalize vectors
        # new_base_vecs = new_base_vecs / (pi*0.5)
        new_base_vecs = new_base_vecs / np.tile(
            LA.norm(new_base_vecs, axis=0), (new_base_vecs.shape[1], 1)
        )

        # new_angle = angle - nullvector_in_newbase
        # new_angle = angle + nullvector_in_newbase
        # new_angle = angle + new_base_as_angle[:, 0]
        new_angle = angle - new_base_as_angle[:, 0]

        temp_copy_new_angle = np.copy(new_angle)  # TODO: remove....
        # new_angle = LA.pinv(new_base_vecs).dot(angle - new_base_as_angle[:, 0])
        new_angle_new_base = LA.pinv(new_base_vecs).dot(new_angle)

        # Make sure length is conserved
        # TODO: is this too hacky?! / what are the alternatives..
        # OR devide base by pi/2 (?)
        # new_angle_norm = LA.norm(new_angle_new_base)
        # if new_angle_norm:
        # new_angle_new_base = new_angle_new_base / new_angle_norm * LA.norm(new_angle)

        # nullvector_in_newbase = new_base_as_angle[:, ii] = get_angle_from_vector(
        # self.base[0], base=new_base)

        # new_angle = new_angle + nullvector_in_newbase
        if True:
            normed = new_base_vecs / np.tile(
                LA.norm(new_base_vecs, axis=0), (dim - 1, 1)
            )
            print("angle dot prod", np.dot(normed[:, 0], normed[:, 1]))
            print("new_base_vecs", new_base_vecs)

        return UnitDirection(new_base).from_angle(new_angle)


# class DirectionBase:
#     # This seems redundant to existing 'base'-array
#     """Directional base class to store the null_matrix / base_matrix
#     which allows to represent vectors."""

#     def __init__(
#         self,
#         matrix: np.ndarray = None,
#         vector: np.ndarray = None,
#         direction_base: DirectionBase = None,
#     ):
#         # Should it be a mutable OR immutable object?
#         # TODO MAYBE: tests(?)

#         if matrix is not None:
#             self._matrix = np.array(matrix)

#         elif vector is not None:
#             self._matrix = get_orthogonal_basis(vector)

#         elif direction_base is not None:
#             self._matrix = np.array(direction_base.null_matrix)

#         else:
#             raise ValueError("No input argument as a base of the space.")

#     def __getitem__(self, arg: int) -> np.ndarrary:
#         return self._matrix[:, arg]

#     def __repr__(self):
#         # return f"DirectionBase({str(self._matrix)})"
#         return f"DirectionBase({np.array2string(self._matrix, separator=', ')})"

#     def __eq__(self, other: DirectionBase) -> bool:
#         return np.allclose(self.null_matrix, other.null_matrix)

#     def __ne__(self, other: DirectionBase) -> bool:
#         return not (self == other)

#     def __matmul__(self, other: np.ndarray):
#         return self.dot(other)

#     @property
#     def T(self) -> DirectionBase:
#         # Transpose
#         selfcopy = copy.deepcopy(self)
#         selfcopy._matrix = selfcopy._matrix.T
#         return selfcopy

#     @property
#     def null_matrix(self) -> DirectionBase:
#         return self._matrix

#     def dot(self, other: np.ndarray):
#         # Dot product
#         return self._matrix.dot(other)

#     def invert_normal(self):
#         """Invert the normal / first vector."""
#         selfcopy = copy.deepcopy(self)
#         selfcopy._matrix[:, 0] = (-1) * selfcopy._matrix[:, 0]
#         return selfcopy

#     # @null_matrix.setter
#     # def null_matrix(self, value):
#     # self._matrix = value
