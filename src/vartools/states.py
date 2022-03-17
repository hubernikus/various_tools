"""
Basic state to base anything on.
"""
# Author: Lukas Huber
# Mail: lukas.huber@epfl.ch
# License: BSD (c) 2021
# import time
# Use python 3.10 [annotations / typematching]
from __future__ import annotations  # Not needed from python 3.10 onwards

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.spatial.transform import Rotation  # scipy rotation

# TODO: use this as an attribute for further calculations
# !WARNING: This is still very experimental


def get_rotation_matrix(orientation: np.ndarray) -> np.ndarray:
    """Return rotation matrix based on 2D-orientation input."""
    matrix = np.array(
        [
            [np.cos(orientation), -np.sin(orientation)],
            [np.sin(orientation), np.cos(orientation)],
        ]
    )
    return matrix


class BaseState:
    def __init__(self, position, orientation, velocity, angular_velocity):
        pass


class Time:
    pass


class Stamp:
    def __init__(self, seq: int = None, stamp: Time = None, frame_id: str = None):
        self.seq = seq
        self.time = time
        self.frame_id = frame_id


@dataclass
class ObjectTwist:
    linear: np.ndarray
    angular: np.ndarray


class ObjectPose:
    """(ROS)-inspired pose of an object of dimension
    Attributes
    ----------
    Position

    """

    def __init__(
        self,
        position: np.ndarray = None,
        orientation: np.ndarray = None,
        stamp: Stamp = None,
        dimension: int = None,
    ):
        # 2D case has rotation matrix
        self._rotation_matrix = None

        # Assign values
        self.position = position
        self.orientation = orientation
        self.stamp = stamp

    @property
    def dimension(self):
        if self.position is None:
            return None
        return self.position.shape[0]

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        if value is None:
            self._position = value
            return
        self._position = np.array(value)

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        """ Value is of type 'float' for 2D or `np.array`/`scipy.rotation` for 3D and higher."""
        if value is None:
            self._orientation = value
            return

        if self.dimension == 2:
            self._orientation = value
            self._rotation_matrix = get_rotation_matrix(self.orientation)

        elif self.dimension == 3:
            if not isinstance(value, Rotation):
                raise TypeError("Use 'scipy - Rotation' type for 3D orientation.")
            self._orientation = value

        else:
            if value is not None and np.sum(np.abs(value)):  # nonzero value
                warnings.warn("Rotation for dimensions > 3 not defined.")
            self._orientation = value

    def update(self, delta_time: float, twist: ObjectTwist):
        if twist.linear is not None:
            self.position = position + twist.linear * delta_time

        if twist.angular is not None:
            breakpoint()
            # Not implemented
            self.angular = position + twist.agnular * delta_time

    def transform_position_from_reference_to_local(
        self, position: np.ndarray
    ) -> np.ndarray:
        """Transform a position from the global frame of reference
        to the obstacle frame of reference"""
        if not self.position is None:
            position = position - self.position

        return self.apply_rotation_reference_to_local(direction=position)

    def transform_position_from_local_to_reference(
        self, position: np.ndarray
    ) -> np.ndarray:
        """Transform a position from the obstacle frame of reference
        to the global frame of reference"""
        position = self.apply_rotation_local_to_reference(direction=position)

        if not self.position is None:
            position = position + self.position

        return position

    def transform_direction_from_reference_to_local(
        self, direction: np.ndarray
    ) -> np.ndarray:
        """Transform a direction, velocity or relative position to the global-frame"""
        return self.apply_rotation_reference_to_local(direction)

    def apply_rotation_reference_to_local(self, direction: np.ndarray) -> np.ndarray:
        if self._orientation is None:
            return direction

        if self.dimension == 2:
            return self._rotation_matrix.T.dot(direction)

        elif self.dimension == 3:
            return self._orientation.inv.apply(direction.T).T
        else:
            warnings.warn("Not implemented for higer dimensions")
            return direction

    def transform_direction_from_local_to_reference(
        self, direction: np.ndarray
    ) -> np.ndarray:
        """Transform a direction, velocity or relative position to the obstacle-frame"""
        return self.apply_rotation_local_to_reference(direction)

    def apply_rotation_local_to_reference(self, direction: np.ndarray) -> np.ndarray:
        if self._orientation is None:
            return direction

        if self.dimension == 2:
            return self._rotation_matrix.dot(direction)

        elif self.dimension == 3:
            return self._orientation.apply(direction.T).T

        else:
            warnings.warn("Not implemented for higer dimensions")
            return direction


class Wrench:
    def __init__(self, linear, angular):
        pass


class ConstantMovingState:
    pass


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
        if self.dim == 2:
            self.compute__rotation_matrix()
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
                return self._rotation_matrix.T.dot(
                    position - np.array(self.center_position)
                )
            elif len(position.shape) == 2:
                n_points = position.shape[1]
                return self._rotation_matrix.T.dot(
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
                return self._rotation_matrix.dot(position) + self.center_position
            elif len(position.shape) == 2:
                n_points = position.shape[1]
                return (
                    self._rotation_matrix.dot(position)
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
            return self._rotation_matrix.dot(direction)

        elif self.dim == 3:
            return self._orientation.apply(direction.T).T

        else:
            warnings.warn("Not implemented for higer dimensions")
            return direction

    def transform_global2relative_dir(self, direction):
        """Transform a direction, velocity or relative position to the local-frame"""
        if self.dim == 2:
            return self._rotation_matrix.T.dot(direction)

        elif self.dim == 3:
            return self._orientation.inv.apply(direction.T).T

        else:
            warnings.warn("Not implemented for higer dimensions")
            return direction
