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


class ObjectTwist:
    def __init__(
        self,
        linear: np.ndarray = None,
        angular: np.ndarray = None,
        dimension: float = None,
    ):

        if dimension is None:
            if linear is None:
                self.dimension = 2
            else:
                self.dimension = dimension

        self.linear = linear
        self.angular = angular

    @property
    def linear(self):
        return self._linear

    @linear.setter
    def linear(self, value):
        if value is None:
            self._linear = np.zeros(self.dimension)
        else:
            self._linear = np.array(value)


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
        """Value is of type 'float' for 2D
        or `numpy.array`/`scipy.spatial.transform.Rotation` for 3D and higher."""
        if value is None:
            self._orientation = value
            return

        if self.dimension == 2:
            self._orientation = value
            # self.rotation_matrix = get_rotation_matrix(self.orientation)

        elif self.dimension == 3:
            if not isinstance(value, Rotation):
                raise TypeError("Use 'scipy - Rotation' type for 3D orientation.")
            self._orientation = value

        else:
            if value is not None and np.sum(np.abs(value)):  # nonzero value
                warnings.warn("Rotation for dimensions > 3 not defined.")
            self._orientation = value

    @property
    def rotation_matrix(self):
        if self.dimension != 2:
            warnings.warn("Orientation matrix only used for useful for 2-D rotations.")
            return
        if self.orientation is None:
            return np.eye(self.dimension)

        _cos = np.cos(self.orientation)
        _sin = np.sin(self.orientation)
        return np.array([[_cos, (-1) * _sin], [_sin, _cos]])

    def update(self, delta_time: float, twist: ObjectTwist):
        if twist.linear is not None:
            self.position = position + twist.linear * delta_time

        if twist.angular is not None:
            breakpoint()
            # Not implemented
            self.angular = position + twist.agnular * delta_time

    def transform_position_from_reference_to_local(self, *args, **kwargs):
        # TODO: is being renamed -> remove original]
        return self.transform_position_from_relative(*args, **kwargs)

    def transform_pose_to_relative(self, pose: ObjectPose) -> ObjectPose:
        pose.position = self.transform_position_to_relative(pose.position)

        if self.orientation is None:
            return pose

        if pose.orientation is None:
            pose.orientation = self.orientation
            return pose

        if self.dimension != 2:
            raise NotImplementedError()

        pose.orientation += self.orientation

    def transform_pose_from_relative(self, pose: ObjectPose) -> ObjectPose:
        pose.position = self.transform_position_from_relative(pose.position)

        if self.orientation is None:
            return pose

        if pose.orientation is None:
            pose.orientation = (-1) * self.orientation
            return pose

        if self.dimension != 2:
            raise NotImplementedError()

        pose.orientation -= self.orientation

        return pose

    def transform_position_from_relative(self, position: np.ndarray) -> np.ndarray:
        """Transform a position from the global frame of reference
        to the obstacle frame of reference"""
        if not self.position is None:
            position = position - self.position

        return self.apply_rotation_reference_to_local(direction=position)

    def transform_position_from_local_to_reference(
        self, position: np.ndarray
    ) -> np.ndarray:
        return self.transform_position_to_relative(position)

    def transform_position_to_relative(self, position: np.ndarray) -> np.ndarray:
        """Transform a position from the obstacle frame of reference
        to the global frame of reference"""
        position = self.apply_rotation_local_to_reference(direction=position)

        if self.position is not None:
            position = position + self.position

        return position

    def transform_direction_from_reference_to_local(
        self, direction: np.ndarray
    ) -> np.ndarray:
        """Transform a direction, velocity or relative position to the global-frame."""
        return self.apply_rotation_reference_to_local(direction)

    def apply_rotation_reference_to_local(self, direction: np.ndarray) -> np.ndarray:
        if self._orientation is None:
            return direction

        if self.dimension == 2:
            return self.rotation_matrix.T.dot(direction)

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
            return self.rotation_matrix.dot(direction)

        elif self.dimension == 3:
            return self._orientation.apply(direction.T).T

        else:
            warnings.warn("Not implemented for higer dimensions")
            return direction


class Wrench:
    def __init__(self, linear, angular):
        pass
