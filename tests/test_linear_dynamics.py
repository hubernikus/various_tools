"""
Test linear dynamical system.
"""

import numpy as np

from vartools.dynamical_systems import LinearSystem
from vartools.dynamical_systems import plot_dynamical_system_quiver


def test_converging_curvy_ds(visualize=False):
    dynamics = LinearSystem(
        attractor_position=np.zeros(2), A_matrix=np.array([[-1, -2], [2, -1]])
    )

    if visualize:
        plot_dynamical_system_quiver(
            dynamical_system=dynamics, x_lim=[-5, 5], y_lim=[-4, 4], axes_equal=True
        )

    position = np.zeros(2)
    assert np.allclose(
        dynamics.evaluate(position), np.zeros(2)
    ), "Attractor is instable."

    position = np.array([3, 4.0])
    velocity = dynamics.evaluate(position)
    assert np.linalg.norm(velocity) > 0, "Zero velocity away from attractor."
    assert np.dot(velocity, position) < 0, "System is not quadratic Lyapunov stable."


if (__name__) == "__main__":
    test_converging_curvy_ds(visualize=True)

    # import sys
    # print(f"Done testing of {sys.argv[0]}")
