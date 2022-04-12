"""
Angle math for python in 2D
Helper function for directional & angle evaluations
"""
# Author: Lukas Huber
# Created: 2019-11-15
# Email: lukas.huber@epfl.ch

import warnings
from math import pi

import numpy as np

# TODO: optimize for speed. Cython?


def angle_is_between(angle_test, angle_low, angle_high):
    """Verify if angle_test is in between angle_low & angle_high"""
    delta_low = angle_difference_directional(angle_test, angle_low)
    delta_high = angle_difference_directional(angle_high, angle_test)

    return delta_low > 0 and delta_high > 0


def angle_is_in_between(angle_test, angle_low, angle_high, margin=1e-9):
    """Verify if angle_test is in between angle_low & angle_high
    Values are between [0, 2pi].
    Margin to account for numerical errors."""
    delta_low = angle_difference_directional_2pi(angle_test, angle_low)
    delta_high = angle_difference_directional_2pi(angle_high, angle_test)

    delta_tot = angle_difference_directional_2pi(angle_high, angle_low)

    return np.abs((delta_high + delta_low) - delta_tot) < margin


def angle_modulo(angle):
    """Get angle in [-pi, pi["""
    return ((angle + pi) % (2 * pi)) - pi


def angle_difference_directional_2pi(angle1, angle2):
    angle_diff = angle1 - angle2
    while angle_diff > 2 * pi:
        angle_diff -= 2 * pi
    while angle_diff < 0:
        angle_diff += 2 * pi
    return angle_diff


def angle_difference_directional(angle1, angle2):
    """
    Difference between two angles ]-pi, pi]
    Note: angle1-angle2 (non-commutative)
    """
    angle_diff = angle1 - angle2
    while angle_diff > pi:
        angle_diff = angle_diff - 2 * pi
    while angle_diff <= -pi:
        angle_diff = angle_diff + 2 * pi
    return angle_diff


def angle_difference(angle1, angle2):
    return angle_difference_directional(angle1, angle2)


def angle_difference_abs(angle1, angle2):
    """
    Difference between two angles [0,pi[
    angle1-angle2 = angle2-angle1(commutative)
    """
    angle_diff = np.abs(angle2 - angle1)
    while angle_diff >= pi:
        angle_diff = 2 * pi - angle_diff
    return angle_diff


def transform_polar2cartesian(
    magnitude, angle, center_position=None, center_point=None
):
    """Transform 2d from polar- to cartesian coordinates."""
    # Only 2D input
    if not center_point is None:
        # TODO remove center_position or center_position
        center_position = center_point

    magnitude = np.reshape(magnitude, (-1))
    angle = np.reshape(angle, (-1))

    if center_position is None:
        points = (
            magnitude * np.vstack((np.cos(angle), np.sin(angle)))
            + np.tile(center_position, (magnitude.shape[0], 1)).T
        )
    else:
        # points = [r, phi]
        points = (
            magnitude * np.vstack((np.cos(angle), np.sin(angle)))
            + np.tile(center_position, (magnitude.shape[0], 1)).T
        )

    return np.squeeze(points)


def transform_cartesian2polar(points, center_position=None, second_axis_is_dim=True):
    """
    Two dimensional transformation of cartesian to polar coordinates
    Based on center_position (default value center_position=np.zeros(dim))
    """
    # TODO -- check dim and etc
    # Don't just squeeze, maybe...
    points = np.squeeze(points)
    if second_axis_is_dim:
        points = points.T
    dim = points.shape[0]

    if isinstance(center_position, type(None)):
        center_position = np.zeros(dim)
    else:
        center_position = np.squeeze(center_position)

    if len(points.shape) == 1:
        points = points - center_position

        angle = np.arctan2(points[1], points[0])
    else:
        points = points - np.tile(center_position, (points.shape[1], 1)).T
        angle = np.arctan2(points[1, :], points[0, :])

    magnitude = np.linalg.norm(points, axis=0)

    # output: [r, phi]
    return magnitude, angle


def periodic_weighted_sum(angles, weights, reference_angle=None):
    """Weighted Average of angles (1D)"""
    # TODO: unify with directional_weighted_sum() // see above
    # Extend to dimenions d>2
    if isinstance(angles, list):
        angles = np.array(angles)
    if isinstance(weights, list):
        weights = np.array(weights)

    if reference_angle is None:
        if len(angles) > 2:
            raise NotImplementedError(
                "No mean defined for periodic function with more than two angles."
            )
        reference_angle = (
            angle_difference_directional(angles[0], angles[1]) / 2.0 + angles[1]
        )
        reference_angle = angle_modulo(reference_angle)

    angles = angle_modulo(angles - reference_angle)

    mean_angle = angles.T.dot(weights)
    mean_angle = angle_modulo(mean_angle + reference_angle)

    return mean_angle
