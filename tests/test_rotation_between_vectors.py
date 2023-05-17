import numpy as np

from vartools.linalg import get_rotation_between_vectors


def test_vector_rotation():
    vec_final = np.array([0.14494458, 0.33341965, 0.02992903])
    vec_final = vec_final / np.linalg.norm(vec_final)

    vec_init1 = np.array([0, 1.0, 0])
    rotation1 = get_rotation_between_vectors(vec_init1, vec_final)
    # euler1 = rotation1.as_euler("zyx")
    vec_recons1 = rotation1.apply(vec_init1)
    assert np.allclose(vec_final, vec_recons1)

    vec_init2 = np.array([0, -1.0, 0])
    rotation2 = get_rotation_between_vectors(vec_init2, vec_final)
    # euler2 = rotation2.as_euler("zyx")
    vec_recons2 = rotation2.apply(vec_init2)
    assert np.allclose(vec_final, vec_recons2)


if (__name__) == "__main__":
    test_vector_rotation()
    print("Tests successful.")
