import math
import numpy as np

from vartools.angle_math import get_orientation_from_direction


def test_orientation_from_direction():
    orientation = get_orientation_from_direction(
        direction=[0, 1, 2.0], null_vector=[0, 0, 1]
    )
    rot_xyz = orientation.as_euler("xyz", degrees=True)

    assert 0 < (-1) * rot_xyz[0] < 45
    assert math.isclose(rot_xyz[1], 0)
    assert math.isclose(rot_xyz[2], 0)

    orientation = get_orientation_from_direction(
        direction=[0, 0, 1], null_vector=[0, 0, 1]
    )
    rot_xyz = orientation.as_euler("xyz")
    assert np.allclose(rot_xyz, [0, 0, 0])


if (__name__) == "__main__":
    test_orientation_from_direction()
