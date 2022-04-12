"""
Dynamical Systems with a closed-form description.
"""
# TODO remove this file.. -> new implementation is in classes
# Author: Lukas Huber
# License: BSD (c) 2021
import numpy as np
from vartools.dynamicalsys import DynamicalSystem, allow_max_velocity

# def decorator(original_function=None, *, optional_argument1=None, optional_argument2=None, ...):
#     def _decorate(function):
#         @functools.wraps(function)
#         def wrapped_function(*args, **kwargs):
#             # ...
#             pass
#         return wrapped_function

#     if original_function:
#         return _decorate(original_function)

#     return _decorate


# def limit_velocity(original_function=None, *, max_vel=1.0):
#     """ Decorator which applies a maximum velocity to the dynamical systems."""

#     def _decorate(function):
#         @functools.wraps(function)
#         def wrapped_function(*args, **kwargs):
#             velocity = func(*args, **kwargs)
#             mag_vel = np.linalg.norm(velocity)
#             if mag_vel > max_vel:
#                 velocity = velocity / mag_vel

#             return velocity

#         return wrapped_function

#     if original_function:
#         return _decorate(original_function)
#     return _decorate


def parallel_ds(position, direction):
    """Constant dynamical system in one direction."""
    return direction


@allow_max_velocity
def evaluate_linear_dynamical_system(
    position, A_matrix=None, center_position=None, b=None
):
    """Linear Dyanmical system of the form
    dot[x] = A @ (x - center_position) or
    dot[x] = A @ x + b
    """

    if A_matrix is None:
        A_matrix = np.eye(position.shape[0]) * (-1)

    if center_position is not None:
        if b is not None:
            raise ValueError(
                "center_pos AND baseline default arguments has been used."
                + "Only one of them possible."
            )

        return A_matrix.dot(position - center_position)

    elif b is not None:
        breakpoint()
        return A_matrix.dot(position) + b

    else:
        return A_matrix.dot(position)


@allow_max_velocity
def evaluate_stable_circle_dynamical_system(
    position, radius, center_position=None, factor_linearsys=1, direction=1
):
    """Return the stable linear-circle dynamical-system evaluated at position position.
    direction: the mathematical direction of the DS.
    """
    if len(position.shape) != 1 or position.shape[0] != 2:
        raise ValueError("Position input allowed only of shape (2,)")

    if center_position is not None:
        # Compute based on relative position
        position = position - center_position

    pos_norm = np.linalg.norm(position)
    if not pos_norm:
        return np.zeros(position.shape)

    velocity_linear = radius - pos_norm
    if pos_norm < radius:
        velocity_linear = 1 - 1.0 / velocity_linear
    velocity_linear = position / pos_norm * velocity_linear

    if abs(direction) != 1:
        warnings.warn("Direction not of magnitude 1. It is automatically reduced.")
        direction = np.copysign(1, direction)

    velocity_circular = direction * np.array([-position[1], position[0]])
    velocity_circular = velocity_circular / np.linalg.norm(velocity_circular)

    return velocity_linear * factor_linearsys + velocity_circular


@allow_max_velocity
def ds_quadratic_axis_convergence(
    position, center_position=None, main_axis=None, conv_pow=2, stretching_factor=1
):
    """Dynamical system wich convergence faster towards x-axis."""
    # TODO: add additional paramters

    if center_position is not None:
        position = position - center_position

    if main_axis is not None:
        # TODO
        raise NotImplementedError()

    velocity = (-1) * stretching_factor * position
    velocity[1:] = np.copysign(velocity[1:] ** conv_pow, velocity[1:])
    return velocity
