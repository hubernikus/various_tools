"""
Test General State used for Obstacle Description
"""
import numpy as np
from scipy.spatial.transform import Rotation

from vartools.states import Pose


def test_3d_multi_rotations():
    initial_positions = np.array([[1, 0, 0.0], [0.0, 1.0, 0.0]]).T

    pose = Pose(np.array([0, 0, 1]), Rotation.from_euler("z", np.pi / 2))
    final_pos = pose.transform_positions_from_relative(initial_positions)

    assert np.allclose(final_pos[:, 0], [0, 1, 1])
    assert np.allclose(final_pos[:, 1], [-1, 0, 1])


if (__name__) == "__main__":
    test_3d_multi_rotations()
