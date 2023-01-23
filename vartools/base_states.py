"""
Basic state to base anything on.
"""
# Author: Lukas Huber
# Mail: lukas.huber@epfl.ch
# License: BSD (c) 2021
# import time
# Use python 3.10 [annotations / typematching]
from __future__ import annotations  # Not needed from python 3.10 onwards

import warnings

import numpy as np
from scipy.spatial.transform import Rotation  # scipy rotation


def get_rotation_matrix(orientation: np.ndarray) -> np.ndarray:
    """Return rotation matrix based on 2D-orientation input."""
    matrix = np.array(
        [
            [np.cos(orientation), -np.sin(orientation)],
            [np.sin(orientation), np.cos(orientation)],
        ]
    )
    return matrix


class State(object):
    """Basic state class which allows encapsulates further."""

    def __init__(
        self, typename=None, State=None, name="default", reference_frame="base"
    ):
        if State is not None:
            self = copy.deepcopy(State)
        else:
            self.typename = typename
            self.reference_frame = reference_frame
            self.name = name

    @property
    def typename(self):
        return self._typename

    @typename.setter
    def typename(self, value):
        self._typename = value

    @property
    def reference_frame(self):
        return self._reference_frame

    @reference_frame.setter
    def reference_frame(self, value):
        self._reference_frame = value

    @property
    def center_position(self):
        return self._center_position
o
    @center_position.setter
    def center_position(self, value):
        if isinstance(value, list):
            self._center_position = np.array(value)
        else:
            self._center_position = value

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        """Orientation is of type `float` for 2D and of type `np.array`/scipy.rotation for 3D and higher."""
        if self.dim == 2:
            # self.compute_rotation_matrix()
            self._orientation = value

        elif self.dim == 3:
            if not isinstance(value, Rotation):
                raise TypeError("Use 'scipy - Rotation' type for 3D orientation.")
            self._orientation = value

        else:
            if value is not None and np.sum(np.abs(value)):  # nonzero value
                warnings.warn("Rotation for dimensions > 3 not defined.")
            self._orientation = value

    def transform_global2relative(self, position):
        """Transform a position from the global frame of reference
        to the obstacle frame of reference"""
        # TODO: transform this into wrapper / decorator
        if not position.shape[0] == self.dim:
            raise ValueError("Wrong position dimensions")

        if self.dim == 2:
            if len(position.shape) == 1:
                return self.rotation_matrix.T.dot(
                    position - np.array(self.center_position)
                )
            elif len(position.shape) == 2:
                n_points = position.shape[1]
                return self.rotation_matrix.T.dot(
                    position - np.tile(self.center_position, (n_points, 1)).T
                )
            else:
                raise ValueError("Unexpected position-shape")

        elif self.dim == 3:
            if len(position.shape) == 1:
                return self._orientation.inv().apply(position - self.center_position)

            elif len(position.shape) == 2:
                n_points = position.shape[1]
                return (
                    self._orientation.inv()
                    .apply(position.T - np.tile(self.center_position, (n_points, 1)))
                    .T
                )

        else:
            warnings.warn(
                "Rotation for dimensions {} need to be implemented".format(self.dim)
            )
            return position

    def transform_relative2global(self, position):
        """Transform a position from the obstacle frame of reference
        to the global frame of reference"""
        if not isinstance(position, (list, np.ndarray)):
            raise TypeError(
                "Position={} is of type {}".format(position, type(position))
            )

        if self.dim == 2:
            if len(position.shape) == 1:
                return self.rotation_matrix.dot(position) + self.center_position
            elif len(position.shape) == 2:
                n_points = position.shape[1]
                return (
                    self.rotation_matrix.dot(position)
                    + np.tile(self.center_position, (n_points, 1)).T
                )
            else:
                raise ValueError("Unexpected position-shape")

        elif self.dim == 3:
            if len(position.shape) == 1:
                return self._orientation.apply(position) + self.center_position

            elif len(position.shape) == 2:
                n_points = position.shape[1]
                return (
                    self._orientation.apply(position.T)
                    + +np.tile(self.center_position, (n_points, 1))
                ).T

            else:
                raise ValueError("Unexpected position-shape")

        else:
            warnings.warn(
                "Rotation for dimensions {} need to be implemented".format(self.dim)
            )
            return position

    def transform_relative2global_dir(self, direction):
        """Transform a direction, velocity or relative position to the local-frame"""
        if self.dim == 2:
            return self.rotation_matrix.dot(direction)

        elif self.dim == 3:
            return self._orientation.apply(direction.T).T

        else:
            warnings.warn("Not implemented for higer dimensions")
            return direction

    def transform_global2relative_dir(self, direction):
        """Transform a direction, velocity or relative position to the local-frame"""
        if self.dim == 2:
            return self.rotation_matrix.T.dot(direction)

        elif self.dim == 3:
            return self._orientation.inv.apply(direction.T).T

        else:
            warnings.warn("Not implemented for higer dimensions")
            return direction
