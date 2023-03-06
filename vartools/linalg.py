"""
Different linear algebraig helper function (mainly) based on numpy
"""
# Author: Lukas Huber
# Created: 2019-11-15
# Email: lukas.huber@epfl.ch

# from functools import lru_cache
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
    vector = vector / np.linalg.norm(vector)

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

        ind_zeros = np.isclose(vector, 0.0)
        n_zeros = sum(ind_zeros)
        ind_nonzero = np.logical_not(ind_zeros)

        n_nonzeros = sum(ind_nonzero)
        sub_vector = vector[ind_nonzero]
        sub_matrix = np.zeros((n_nonzeros, n_nonzeros))
        sub_matrix[:, 0] = sub_vector

        for ii, jj in enumerate(np.arange(ind_zeros.shape[0])[ind_zeros]):
            basis_matrix[jj, ii + 1] = 1.0

        for ii in range(1, n_nonzeros):
            sub_matrix[:ii, ii] = sub_vector[:ii]
            sub_matrix[ii, ii] = -np.sum(sub_vector[:ii] ** 2) / sub_vector[ii]
            sub_matrix[: ii + 1, ii] = sub_matrix[: ii + 1, ii] / np.linalg.norm(
                sub_matrix[: ii + 1, ii]
            )

            basis_matrix[ind_nonzero, n_zeros + ii] = sub_matrix[:, ii]
    return basis_matrix
