"""
Different linear algebraig helper function (mainly) based on numpy
"""
# Author: Lukas Huber
# Created: 2019-11-15
# Email: lukas.huber@epfl.ch

import warnings
from functools import lru_cache

import numpy as np


def logical_xor(value1: float, value2: float) -> bool:
    return bool(value1) ^ bool(value2)


def is_positive_definite(x: np.ndarray) -> bool:
    """Check if input matrix x is positive definite and return True/False."""
    return np.all(np.linalg.eigvals(x) > 0)


def is_negative_definite(x: np.ndarray) -> bool:
    """Check if input matrix x is positive definite and return True/False."""
    return np.all(np.linalg.eigvals(x) < 0)


class OrthogonalBasisError(Exception):
    def __init__(self, message):
        self._message = message
        super().__init__(message)

    def __str__(self):
        return f"{self._message} -> Orthogonal basis matrix not defined."

# @lru_cache(maxsize=10)
# TODO: expand cache for this [numpy-arrays]
# TODO: OR make cython
def get_orthogonal_basis(vector: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Get Orthonormal basis matrxi for an dimensional input vector."""
    # warnings.warn("Basis implementation is not continuous.") (?! problem?)
    v_norm = np.linalg.norm(vector)
    vector = vector / v_norm

    dim = vector.shape[0]
    if dim <= 1:
        return vector.reshape((dim, dim))

    basis_matrix = np.zeros((dim, dim))

    if dim == 2:
        basis_matrix[:, 0] = vector
        basis_matrix[:, 1] = np.array([-basis_matrix[1, 0], basis_matrix[0, 0]])

    elif dim == 3:
        basis_matrix[:, 0] = vector
        basis_matrix[:, 1] = np.array([-vector[1], vector[0], 0])

        norm_vec2 = np.linalg.norm(basis_matrix[:, 1])
        if norm_vec2:
            basis_matrix[:, 1] = basis_matrix[:, 1] / norm_vec2
        else:
            basis_matrix[:, 1] = [1, 0, 0]

        basis_matrix[:, 2] = np.cross(basis_matrix[:, 0], basis_matrix[:, 1])

        norm_vec = np.linalg.norm(basis_matrix[:, 2])
        if norm_vec:
            basis_matrix[:, 2] = basis_matrix[:, 2] / norm_vec

    elif dim > 3:
        # TODO: ensure smoothness for general basis for d > 3 (?!?)
        # if True:
        basis_matrix[:, 0] = vector
        if vector[0]:  # nonzero value
            it_start = 1
        else:
            basis_matrix[0, 1] = 1
            it_start = 2

        for ii in range(it_start, dim):
            if vector[ii]:  # nonzero
                basis_matrix[:ii, ii] = vector[:ii]
                basis_matrix[ii, ii] = -np.sum(vector[:ii] ** 2) / vector[ii]
                basis_matrix[: ii + 1, ii] = basis_matrix[
                    : ii + 1, ii
                ] / np.linalg.norm(basis_matrix[: ii + 1, ii])
            else:
                basis_matrix[ii, ii] = 1

    return basis_matrix
