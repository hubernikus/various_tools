#!/USSR/bin/python3
""" Create the rotation space which is so much needed. ... """
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-07-07

# Use python 3.10 [annotations / typematching]
from __future__ import annotations  # Not needed from python 3.10 onwards

import copy
import warnings
import math
from typing import Optional

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# import numpy.typing as npt
from numpy import linalg as LA

import networkx as nx

from vartools.linalg import get_orthogonal_basis

Vector = np.ndarray
VectorArray = np.ndarray

NodeType = int


def rotate_direction(
    direction: Vector, base: VectorArray, rotation_angle: float
) -> Vector:
    """Returns the rotated of the input vector with respect to the base and rotation angle."""
    if not (dir_norm := LA.norm(direction)):
        # Zero vector can not be rotated
        return direction

    direction = direction / dir_norm

    dot_prods = np.dot(base.T, direction)
    angle = math.atan2(dot_prods[1], dot_prods[0]) + rotation_angle

    # Convert angle to the two basis-axis
    out_direction = math.cos(angle) * base[:, 0] + math.sin(angle) * base[:, 1]
    out_direction *= math.sqrt(sum(dot_prods**2))

    # Finally, add the orthogonal part (no effect in 2D, but important for higher dimensions)
    out_direction += direction - np.sum(dot_prods * base, axis=1)
    return out_direction * dir_norm


def rotate_array(
    directions: VectorArray,
    base: VectorArray,
    rotation_angle: float,
) -> VectorArray:
    """Rotate upper level base with respect to total."""
    dimension, n_dirs = directions.shape

    directions = directions / LA.norm(directions, axis=0)

    # Matrix dimensions: [2 x n_dirs ] <- [dimension x 2 ].T @ [dimension x n_dirs]
    dot_prods = np.dot(base.T, directions)
    angles = np.arctan2(dot_prods[1, :], dot_prods[0, :]) + rotation_angle

    # Compute output from rotation
    out_vectors = np.tile(base[:, 0], (n_dirs, 1)).T * np.tile(
        np.cos(angles), (dimension, 1)
    ) + np.tile(base[:, 1], (n_dirs, 1)).T * np.tile(np.sin(angles), (dimension, 1))
    out_vectors *= np.tile(np.sqrt(np.sum(dot_prods**2, axis=0)), (dimension, 1))

    # Finally, add the orthogonal part (no effect in 2D, but important for higher dimensions)
    out_vectors += directions - (base @ dot_prods)

    return out_vectors


@dataclass
class VectorRotationXd:
    """This approach allows successive modulation which can be added up.

    Attributes
    ----------
    base array of size [dimension x 2]: The (orthonormal) base constructed from the to
        input directions
    rotation_angle (float): The rotation angle resulting from the two input directions
    """

    base: VectorArray
    rotation_angle: float

    @classmethod
    def from_directions(cls, vec_init: Vector, vec_rot: Vector) -> VectorRotationXd:
        """Alternative constructor base on two input vectors which define the
        initialization."""

        # # Normalize both vectors
        vec_init = vec_init / LA.norm(vec_init)
        vec_rot = vec_rot / LA.norm(vec_rot)

        dot_prod = np.dot(vec_init, vec_rot)
        if dot_prod == (-1):
            warnings.warn("Antiparallel vectors")

        if abs(dot_prod) < 1:
            vec_perp = vec_rot - vec_init * dot_prod
            vec_perp = vec_perp / LA.norm(vec_perp)

        else:
            # (Anti-)parallel vectors => calculate random perpendicular vector
            vec_perp = np.zeros(vec_init.shape)
            if not LA.norm(vec_init[:2]):
                vec_perp[0] = 1
            else:
                vec_perp[0] = vec_init[1]
                vec_perp[1] = vec_init[0] * (-1)
                vec_perp[:2] = vec_perp[:2] / LA.norm(vec_perp[:2])

        angle = np.arccos(min(max(dot_prod, -1), 1))
        return cls(base=np.array([vec_init, vec_perp]).T, rotation_angle=angle)

    # def __mult__(self, factor) -> VectorRotationXd:
    #     instance_copy = copy.deepcopy(self)
    #     instance_copy.rotation_angle = instance_copy.rotation_angle * factor
    #     return instance_copy

    @property
    def base0(self):
        return self.base[:, 0]

    @property
    def dimension(self):
        try:
            return self.base.shape[0]
        except AttributeError:
            warnings.warn("base has not been defined")
            return None

    def get_second_vector(self) -> Vector:
        """Returns the second vector responsible for the rotation"""
        return rotate_direction(
            direction=self.base[:, 0],
            rotation_angle=self.rotation_angle,
            base=self.base,
        )

    def rotate(self, direction, rot_factor: float = 1):
        """Returns the rotated of the input vector with respect to the base and rotation angle
        rot_factor: factor gives information about extension of rotation"""
        return rotate_direction(
            direction=direction,
            rotation_angle=rot_factor * self.rotation_angle,
            base=self.base,
        )

    def rotate_vector_rotation(
        self, rotation: VectorRotationXd, rot_factor: float = 1
    ) -> VectorRotationXd:
        rotation = copy.deepcopy(rotation)
        rotation.base = rotate_array(
            directions=rotation.base,
            base=rotation.base,
            rotation_angle=rot_factor * self.rotation_angle,
        )
        return rotation

    def inverse_rotate(self, direction):
        return rotate_direction(
            direction=direction,
            rotation_angle=(-1) * self.rotation_angle,
            base=self.base,
        )


class VectorRotationSequence:
    """
    Vector-Rotation environment based on multiple vectors

    Attributes
    ----------
    vectors_array (np.array of shape [dimension x n_rotations + 1]):
        (storing) the inital array of vectors
    basis_array (numpy array of  shape [dimension x n_rotations x 2]):
        contains the basis of all rotations
    rotation_angles: The rotation between going from one to the next basis
    """

    def __init__(self, vectors_array: np.ndarray) -> None:
        # Normalize
        self.vectors_array = vectors_array / LA.norm(vectors_array, axis=0)

        dot_prod = np.sum(
            self.vectors_array[:, 1:] * self.vectors_array[:, :-1], axis=0
        )

        if np.sum(dot_prod == (-1)):  # Any of the values
            raise ValueError("Antiparallel vectors.")

        # Evaluate basis and angles
        vec_perp = self.vectors_array[:, 1:] - self.vectors_array[:, :-1] * dot_prod
        vec_perp = vec_perp / LA.norm(vec_perp, axis=0)

        self.basis_array = np.stack((self.vectors_array[:, :-1], vec_perp), axis=2)
        self.rotation_angles = np.arccos(dot_prod)

    @property
    def n_rotations(self):
        return self.basis_array.shape[1]

    @property
    def dimension(self):
        return self.basis_array.shape[0]

    def base(self) -> Vector:
        return self.basis_array[:, [0, -1]]

    def append(self, direction: Vector) -> None:
        self.basis_array = np.hstack((self.basis_array, direction.reshape(-1, 1)))

        raise NotImplementedError("Finish updating basis and rotation angles.")

    def rotate(self, direction: Vector, rot_factor: float = 1) -> Vector:
        """Rotate over the whole length of the vector."""
        weights = np.zeros(self.n_rotations)
        weights[-1] = rot_factor
        return self.rotate_weighted(direction, weights=weights)

    def rotate_weighted(self, direction: Vector, weights: list[float] = None) -> Vector:
        """
        Returns the rotated direction vector with repsect to the (rotation-)weights

        weights (list of floats (>=0) with length [self.n_rotations]): indicates fraction
        of each rotation which is applied.
        """
        # Starting at the root
        cumulated_weights = np.cumsum(weights[::-1])[::-1]

        if not math.isclose(cumulated_weights[0], 1):
            warnings.warn("Weights are summing up to more than 1.")

        temp_base = np.copy(self.basis_array)
        if weights is None:
            temp_angle = self.rotation_angles

        else:
            temp_angle = self.rotation_angles * cumulated_weights

            # Update the basis of rotation weights from top-to-bottom
            # by rotating upper level base with respect to total
            for ii in reversed(range(self.n_rotations - 1)):
                temp_base[:, (ii + 1) :, :] = rotate_array(
                    directions=temp_base[:, (ii + 1) :, :].reshape(self.dimension, -1),
                    base=temp_base[:, ii, :],
                    rotation_angle=self.rotation_angles[ii]
                    * (1 - cumulated_weights[ii]),
                ).reshape(self.dimension, -1, 2)

        # Finally: rotate from bottom-to-top
        for ii in range(self.n_rotations):
            direction = rotate_direction(
                direction=direction,
                rotation_angle=temp_angle[ii],
                base=temp_base[:, ii, :],
            )
        return direction
