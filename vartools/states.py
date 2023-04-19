"""
Basic state to base anything on.
"""
# Author: Lukas Huber
# Mail: lukas.huber@epfl.ch
# License: BSD (c) 2021

# Not needed from python 3.11 onwards
from __future__ import annotations

import copy
import warnings
from typing import Optional
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from scipy.spatial.transform import Rotation  # scipy rotation


class BaseState:
    def __init__(self, position, orientation, velocity, angular_velocity):
        pass


@dataclass(slots=True)
class Stamp:
    seq: int = 0
    timestamp: int = 0
    frame_id: str = ""


@dataclass(slots=True)
class TwistStamped:
    stamp: Stamp
    pose: Pose


@dataclass(slots=True)
# @dataclass
class Twist:
    linear: npt.ArrayLike
    angular: Optional[npt.ArrayLike | float]

    def __post_init__(self):
        self.linear = np.array(self.linear)

        if self.dimension == 3 and self.angular is not None:
            self.angular = np.array(self.angular)

    @property
    def dimension(self) -> int:
        return self.linear.shape[0]

    @classmethod
    def create_trivial(cls, dimension: int) -> Self:
        if dimension == 2:
            angular = 0.0
        elif dimension == 3:
            angular = np.zeros(dimension)
        else:
            angular = None
        # return cls(np.zeros(dimension), angular)
        # This has been breaking once when python was running for too long
        # restarting fixed it... But where did the bug come from?
        return Twist(np.zeros(dimension), angular)


# @dataclass
@dataclass(slots=True)
class ObjectTwist(Twist):
    # TODO remove in the future
    pass


@dataclass(slots=True)
class Pose:
    position: npt.ArrayLike
    # 2D or 3D
    orientation: Optional[float | Rotation] = None

    def __post_init__(self):
        self.position = np.array(self.position)

    @classmethod
    def create_trivial(cls, dimension: int) -> Self:
        if dimension == 2:
            orientation = 0.0
        elif dimension == 3:
            orientation = Rotation.from_euler(0, "x")
        else:
            orientation = None

        return cls(np.zeros(dimension), orientation)

    @property
    def dimension(self) -> int:
        return self.position.shape[0]

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
            self.position = self.position + twist.linear * delta_time

        if twist.angular is not None:
            self.orientation = self.orientation + twist.angular * delta_time

    def transform_position_from_reference_to_local(self, *args, **kwargs):
        # TODO: is being renamed -> remove original]
        return self.transform_position_from_relative(*args, **kwargs)

    def transform_pose_to_relative(self, pose: ObjectPose) -> ObjectPose:
        pose = copy.deepcopy()
        pose.position = self.transform_position_to_relative(pose.position)

        if self.orientation is None:
            return pose

        if pose.orientation is not None:
            pose.orientation = pose.orientation - self.orientation
            return pose

        if self.dimension != 2:
            raise NotImplementedError()

        pose.orientation += self.orientation
        return pose

    def transform_pose_from_relative(self, pose: ObjectPose) -> ObjectPose:
        pose = copy.deepcopy(pose)
        pose.position = self.transform_position_from_relative(pose.position)

        if self.orientation is None:
            return pose

        if pose.orientation is not None:
            pose.orientation = pose.orientation + self.orientation
            return pose

        if self.dimension != 2:
            raise NotImplementedError()

        pose.orientation -= self.orientation

        return pose

    def transform_position_from_relative(self, position: np.ndarray) -> np.ndarray:
        """Transform a position from the global frame of reference
        to the obstacle frame of reference"""
        position = self.transform_direction_from_relative(direction=position)

        if self.position is not None:
            position = position + self.position

        return position

    def transform_positions_from_relative(self, positions: np.ndarray) -> np.ndarray:
        positions = self.transform_directions_from_relative(direction=positions)
        if not self.position is None:
            positions = positions + np.tile(self.position, (positions.shape[1], 1)).T

        return positions

    def transform_position_from_local_to_reference(
        self, position: np.ndarray
    ) -> np.ndarray:
        return self.transform_position_to_relative(position)

    def transform_position_to_relative(self, position: np.ndarray) -> np.ndarray:
        """Transform a position from the obstacle frame of reference
        to the global frame of reference"""
        if self.position is not None:
            position = position - self.position

        position = self.transform_direction_to_relative(direction=position)
        return position

    def transform_positions_to_relative(self, positions: np.ndarray) -> np.ndarray:
        if not self.position is None:
            positions = positions - np.tile(self.position, (positions.shape[1], 1)).T

        positions = self.transform_directions_to_relative(directions=positions)
        return positions

    def transform_direction_from_reference_to_local(
        self, direction: np.ndarray
    ) -> np.ndarray:
        """Transform a direction, velocity or relative position to the global-frame."""
        raise
        # return self.apply_rotation_reference_to_local(direction)

    def transform_direction_from_local_to_reference(
        self, direction: np.ndarray
    ) -> np.ndarray:
        """Transform a direction, velocity or relative position to the obstacle-frame"""
        raise
        # return self.apply_rotation_local_to_reference(direction)

    def transform_direction_from_relative(self, direction: np.ndarray) -> np.ndarray:
        if self.orientation is None:
            return direction

        if self.dimension == 2:
            return self.rotation_matrix.dot(direction)

        elif self.dimension == 3:
            return self.orientation.apply(direction.T).flatten()
        else:
            warnings.warn("Not implemented for higer dimensions")
            return direction

    def transform_directions_from_relative(self, directions: np.ndarray) -> np.ndarray:
        return self.transform_direction_from_relative(directions)

    def transform_direction_to_relative(self, direction: np.ndarray) -> np.ndarray:
        if self.orientation is None:
            return direction

        if self.dimension == 2:
            return self.rotation_matrix.T.dot(direction)

        elif self.dimension == 3:
            return self.orientation.inv().apply(direction.T).flatten()
        else:
            warnings.warn("Not implemented for higer dimensions")
            return direction

    def transform_directions_to_relative(self, directions: np.ndarray) -> np.ndarray:
        return self.transform_direction_to_relative(directions)

    def apply_rotation_reference_to_local(self, direction: np.ndarray) -> np.ndarray:
        if self.orientation is None:
            return direction

        if self.dimension == 2:
            return self.rotation_matrix.T.dot(direction)

        elif self.dimension == 3:
            return self.orientation.inv().apply(direction.T).T
        else:
            warnings.warn("Not implemented for higer dimensions")
            return direction

    def apply_rotation_local_to_reference(self, direction: np.ndarray) -> np.ndarray:
        if self.orientation is None:
            return direction

        if self.dimension == 2:
            return self.rotation_matrix.dot(direction)

        elif self.dimension == 3:
            return self.orientation.apply(direction.T).flatten()

        else:
            warnings.warn("Not implemented for higer dimensions")
            return direction


@dataclass(slots=True)
class ObjectPose(Pose):
    # TODO: remove in the future
    pass


@dataclass(slots=True)
class PoseStamped:
    pose: Pose
    stamp: Stamp


@dataclass(slots=True)
class Wrench:
    linear: np.ndarray
    angular: np.ndarray

    @classmethod
    def create_trivial(cls, dimension: int) -> Self:
        return cls(np.zeros(dimension), np.zeros(dimension))


@dataclass(slots=True)
class WrenchStamped:
    Wrench: Wrench
    stamp: Stamp
