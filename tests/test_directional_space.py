#!/USSR/bin/python3.9
""" Test the directional space. """
# Author: LukasHuber
# Date: 2021-05-18
# Email: lukas.huber@epfl.ch
# License: BSD (c) 2021

import unittest
import copy

from math import pi

import numpy as np
from numpy import linalg as LA

from vartools.linalg import get_orthogonal_basis

from vartools.directional_space import get_angle_space
from vartools.directional_space import get_angle_space_inverse
from vartools.directional_space import get_angle_from_vector, get_vector_from_angle

from vartools.directional_space import UnitDirection


class TestDirectionalSpace(unittest.TestCase):
    directions_3d = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
        [0.3, -0.5, 0.7],
        [-0.9, -0.4, 0.3],
        [1, 1, 1],
        [-1, -1, -1],
        [-2, 3, -1],
    ]

    # Example rotations in degrees (rotations < 90 deg OR pi/2)
    rotations_euler = [
        [0, 0, 0],
        [90, 0, 0],
        [0, 90, 0],
        [0, 0, 90],
        [-90, 0, 0],
        [0, -90, 0],
        [0, 0, -90],
        [34, 12, 25],
        [20, -40, 10],
        [-33, -20, 8],
    ]

    def test_creation(self):
        """Test for plus/minus & multiplication operator."""
        for ii in range(2, 10):
            # From vector
            vector = np.ones(ii) / (ii * 1.0)
            base = get_orthogonal_basis(vector)

    def test_inversion_and_bijectiveness_3d(self):
        """Test of unit-direction inversion in 2d-3d."""
        # dimensions = [2, 3, 4, 10, 20]
        dim = 3
        null_matrix = np.eye(dim)

        from scipy.spatial.transform import Rotation

        for direction in self.directions_3d:
            for rot in self.rotations_euler:
                direction = direction / LA.norm(direction)
                rot = Rotation.from_euler("zyx", [0, 0, 30], degrees=True)
                null_matr_new = rot.apply(null_matrix)
                new_base = null_matr_new

                dir0 = UnitDirection(base=new_base).from_vector(direction)
                dir_inv = dir0.invert_normal()

                dir_reprod = dir_inv.invert_normal()
                self.assertTrue(np.allclose(dir0.base, dir_reprod.base))

                self.assertTrue(np.allclose(dir0.as_angle(), dir_reprod.as_angle()))
                self.assertTrue(np.allclose(dir0.base[0], (-1) * dir_inv.base[0]))
                self.assertTrue(np.isclose(pi - dir0.norm(), dir_inv.norm()))

    # def test_inversion_examples(self):
    #     """ Test specific examples which have caused problems"""
    #     dim = 3
    #     base0 = DirectionBase(matrix=np.eye(dim))
    #     inv_conv_rotated = UnitDirection(base0).from_angle(np.array([pi*0.6, pi*0.4]))
    #     angle = Tester.inv_nonlinear.as_angle()

    #     inv_inv_nonlin = inv_conv_rotated.invert_normal()

    #     # Done

    def test_repetitive_nonnorm_influence(self):
        base = np.eye(3)

        dir1 = UnitDirection(base).from_angle(np.array([1.88495559, 1.25663706]))
        dir2 = UnitDirection(base).from_angle(np.array([-5.02654825, -4.39822972]))

        angle1 = dir1.as_angle()
        angle2 = dir2.as_angle()

        aa = (dir1 - dir2).norm()

        self.assertTrue(np.allclose(angle2, dir2.as_angle()))
        self.assertTrue(np.allclose(angle1, dir1.as_angle()))

    def test_mult_operators(self):
        dim = 3
        base = np.eye(dim)
        dir1 = UnitDirection(base).from_angle(np.array([1, 0]))

        # Multply with float factor
        fac2 = 3
        dir2 = fac2 * dir1
        dir3 = dir1 * fac2

        # self.assertTrue(dir1.base == dir2.base)
        self.assertTrue(np.allclose(dir1.as_angle() * fac2, dir2.as_angle()))

        # self.assertTrue(dir2.base == dir3.base)
        self.assertTrue(np.allclose(dir2.as_angle(), dir3.as_angle()))

    def test_add_operators(self):
        dim = 3
        base = np.eye(dim)
        dir1 = UnitDirection(base).from_angle(np.array([1, 0]))

        # Multply with float factor
        fac2 = 3
        dir2 = fac2 * dir1
        dir3 = dir1 * fac2

        self.assertTrue(np.allclose(dir1.base, dir2.base))
        self.assertTrue(np.allclose(dir1.as_angle() * fac2, dir2.as_angle()))

        self.assertTrue(np.allclose(dir2.base, dir3.base))
        self.assertTrue(np.allclose(dir2.as_angle(), dir3.as_angle()))

    def test_comparison_operator_direction_base(self):
        dim = 3
        base = np.eye(dim)
        dir1 = UnitDirection(base).from_angle(np.array([1, 0]))
        dir2 = UnitDirection(base).from_angle(np.array([0, 1]))

        dir3 = dir1 + dir2
        dir_correct = UnitDirection(base).from_angle(np.array([1, 1]))

        self.assertTrue(np.allclose(dir1.base, dir3.base))
        self.assertTrue(np.allclose(dir2.base, dir3.base))

        self.assertTrue(dir3 == dir_correct)

    def test_orthonormality_matrix(self):
        """Test if matix in higher dimension is orthogonal."""
        # TODO: remove randomness
        margin = 1e-6
        max_dim = 50
        val_range = [-10, 10]

        with self.assertRaises(Exception):
            OrthogonalBasisError(np.array([]))

        with self.assertRaises(Exception):
            OrthogonalBasisError(np.array([1]))

        for dd in range(2, 10):
            with self.assertRaises(Exception):
                OrthogonalBasisError(np.zeros(dd))

        n_repetitions = 100
        for ii in range(n_repetitions):
            dim = np.random.randint(2, max_dim)
            null_direction = np.random.normal(loc=0.0, scale=10.0, size=dim)

            vec = get_orthogonal_basis(null_direction)
            for jj in range(dim):
                for kk in range(jj + 1, dim):
                    self.assertTrue(vec[:, jj].dot(vec[:, kk]) < margin)

    def test_angle_space_distance(self):
        dim = 3
        base = np.eye(dim)
        dir1 = UnitDirection(base).from_angle(np.array([1.88495559, 1.25663706]))
        dir2 = UnitDirection(base).from_angle(np.array([-3.14159265, -3.14159265]))

        dd1 = dir1.as_angle()
        dd2 = dir2.as_angle()

        self.assertAlmostEqual(LA.norm(dd2 - dd1), dir1.get_distance_to(dir2))

    def old_test_bijectional_space(self):
        """Test that forward&inverse directional space is ennaluating."""
        n_repetitions = 100

        # dim = 2
        dimensions = [2, 3, 4, 10]
        for dim in dimensions:
            for ii in range(n_repetitions):
                null_direction = np.random.normal(loc=0.0, scale=10.0, size=dim)

                vec_init = np.random.normal(loc=0.0, scale=10.0, size=dim)

                # Normalize for later comparison
                norm_vec_init = np.linalg.norm(vec_init)
                if not norm_vec_init:
                    continue

                vec_angle = get_angle_space(vec_init, null_direction)
                null_direction = null_direction / np.linalg.norm(null_direction)

                vec_init_rec = get_angle_space_inverse(vec_angle, null_direction)

                self.assertTrue(
                    all(np.isclose(vec_init / np.linalg.norm(vec_init), vec_init_rec))
                )

    def test_special_angle_displacement(self):
        """Test how the displacement behaves when going from space 1 to space 2."""
        null_direction = np.array([1, 0, 0])

        base0 = get_orthogonal_basis(null_direction)
        direction0 = UnitDirection(base=base0)

        direction0.from_angle([0, 0])
        # print('vector 0', direction0.as_vector())
        # print('base 0 \n', direction0.null_matrix)

        null_direction = np.array([0, 1, 0])
        null_matrix = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        base1 = null_matrix
        # print('base 1 \n', direction1.null_matrix)

        direction0.transform_to_base(base1)

    def test_base_transform_same_normal(self):
        """Test that unit-direction-angle length is the same if same normal is existent."""
        from scipy.spatial.transform import Rotation

        # Default [Random?] Values
        null_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        base0 = null_matrix

        directions = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.3, 0.4, 0.78]]

        for ind, direction in enumerate(directions):
            direction0 = UnitDirection(base=base0).from_vector(direction)
            angle_init = direction0.as_angle()

            rot = Rotation.from_euler("zyx", [0, 0, 30], degrees=True)
            null_matr_new = rot.apply(null_matrix)
            new_base = null_matr_new

            direction0.transform_to_base(new_base=new_base)
            angle_after = direction0.as_angle()

            self.assertTrue(
                np.isclose(np.linalg.norm(angle_after), np.linalg.norm(angle_init)),
                "Angle after transformation should have the same norm for simple x-rotation.",
            )

        # print("Successful norm-test.")

    def test_check_bijection(self):
        from scipy.spatial.transform import Rotation

        base = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        initial_vector = np.array([0.0, 1.0, 0.0])

        angle = get_angle_from_vector(initial_vector, base)
        reconst_vector = get_vector_from_angle(angle, base)

        self.assertTrue(np.allclose(initial_vector, reconst_vector))

    def test_check_bijection_rebasing(self):
        from scipy.spatial.transform import Rotation

        base = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        directions = self.directions_3d
        rotations = self.rotations_euler

        for initial_vector in directions:
            initial_vector = initial_vector / LA.norm(initial_vector)

            for rot_vec in rotations:
                rot = Rotation.from_euler("zyx", rot_vec, degrees=True)
                new_null_matr = rot.apply(base)
                new_base = new_null_matr

                if np.allclose((-1) * new_base[0], initial_vector):
                    continue

                # Apply transformation
                direction = UnitDirection(base=base).from_vector(initial_vector)
                direction_rebased = direction.transform_to_base(new_base)

                # Transform back
                direction_back = direction_rebased.transform_to_base(base)

                self.assertTrue(
                    np.allclose(direction.as_vector(), direction_back.as_vector()),
                    "Vector value after backtransformation not consistent.",
                )
                self.assertTrue(
                    np.allclose(direction.as_angle(), direction_back.as_angle()),
                    "Angle value after backtransformation not consistent.",
                )
        # print("Done bijection-rebasing test.")

    def test_180_degree_rotation(visualize=False):
        initial_vector = np.array([0, 1, 0])
        base = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        check_angle = np.array([0, 0])

    def test_90_degree_rotation(self, visualize=False):
        initial_vector = np.array([0, 1, 0])
        base = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        new_base = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        check_angle = np.array([0, 0])

        if visualize:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.subplot(1, 2, 1)
        # Apply test
        direction = UnitDirection(base=base).from_vector(initial_vector)
        direction_rebased = direction.transform_to_base(new_base)

        if visualize:
            angles = np.linspace(0, 2 * np.pi, 50)
            angle_init = direction.as_angle()
            plt.plot(0.5 * pi * np.cos(angles), 0.5 * pi * np.sin(angles), "k")
            plt.plot(0, 0, "ko", label=f"n0={np.round(base[0])}")
            plt.plot(pi / 2, 0, "ko", label=f"e1={np.round(base[1])}")
            plt.plot(0, pi / 2, "ko", label=f"e2={np.round(base[2])}")
            plt.legend()
            plt.axis("equal")
            plt.plot(angle_init[0], angle_init[1], "or")

            vec_labels = [
                f"n0={np.round(new_base[0])}",
                f"e1={np.round(new_base[1])}",
                f"e2={np.round(new_base[2])}",
            ]
            # UnitDirection = UnitDirection(base0)
            norm_angle = UnitDirection(base).from_vector(new_base[0]).as_angle()
            # colors = ['

            # for ii in range(3):
            for ii in range(len(vec_labels)):
                # angle = get_angle_from_vector(direction=null_matrix, base=base0)
                angle = UnitDirection(base).from_vector(new_base[ii]).as_angle()
                plt.plot(angle[0], angle[1], "o", label=vec_labels[ii])

            plt.subplot(1, 2, 2)
            plt.plot(0.5 * pi * np.cos(angles), 0.5 * pi * np.sin(angles), "k")
            plt.plot(0, 0, "ko", label=f"n0={np.round(new_base[0], 1)}")
            plt.plot(pi / 2, 0, "ko", label=f"e1={np.round(new_base[1])}")
            plt.plot(0, pi / 2, "ko", label=f"e2={np.round(new_base[2])}")
            plt.legend()
            plt.axis("equal")

            plt.ion()
            plt.show()

        # TODO: create check
        # self.assertTrue(np.allclose(direction_rebased.as_angle(), check_angle))

    def visual_test_base_transform(self):
        null_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        base0 = null_matrix
        direction0 = UnitDirection(base=base0).from_vector([0, 0, 1])
        angle_init = direction0.as_angle()

        from scipy.spatial.transform import Rotation

        rot = Rotation.from_euler("zyx", [0, 0, 90], degrees=True)
        # rot = Rotation.from_euler('zyx', [0, 0, 0], degrees=True)
        null_matr_new = rot.apply(null_matrix)

        null_matr_new = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

        # null_matr_new = np.array([[0, 0, 0],
        # [1, 0, 1],
        # [0, 0, 0]])

        new_base = null_matr_new

        direction0.transform_to_base(new_base=new_base)

        import matplotlib.pyplot as plt

        plt.figure()
        from math import pi

        # Draw circle
        n_points = 50
        angles = np.linspace(0, 2 * np.pi, n_points)

        plt.subplot(1, 2, 1)
        plt.plot(0.5 * pi * np.cos(angles), 0.5 * pi * np.sin(angles), "k")
        plt.plot(0, 0, "ko", label=f"n0={np.round(base0[0])}")
        plt.plot(pi / 2, 0, "ko", label=f"e1={np.round(base0[1])}")
        plt.plot(0, pi / 2, "ko", label=f"e2={np.round(base0[2])}")
        plt.legend()
        plt.axis("equal")
        plt.plot(angle_init[0], angle_init[1], "or")

        plt.subplot(1, 2, 2)
        plt.plot(0.5 * pi * np.cos(angles), 0.5 * pi * np.sin(angles), "k")
        plt.plot(0, 0, "ko", label=f"n0={np.round(new_base[0], 1)}")
        plt.plot(pi / 2, 0, "ko", label=f"e1={np.round(new_base[1])}")
        plt.plot(0, pi / 2, "ko", label=f"e2={np.round(new_base[2])}")
        plt.legend()
        plt.axis("equal")

        angle = direction0.as_angle()
        plt.plot(angle[0], angle[1], "or")

        plt.ion()
        plt.show()

    def visualization_direction_space(self):
        null_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        base0 = null_matrix
        direction0 = UnitDirection(base=base0).from_vector([0, 0, 1])
        angle_init = direction0.as_angle()

        from scipy.spatial.transform import Rotation

        # rot = Rotation.from_euler('zyx', [0, 0, 90], degrees=True)
        rot = Rotation.from_euler("zyx", [180, 90, 0], degrees=True)
        # rot = Rotation.from_euler('zyx', [0, 0, 0], degrees=True)
        null_matr_new = rot.apply(null_matrix)
        new_base = null_matr_new

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)

        from math import pi

        # Draw circle
        n_points = 50
        angles = np.linspace(0, 2 * np.pi, n_points)
        plt.plot(0.5 * pi * np.cos(angles), 0.5 * pi * np.sin(angles), "k")

        plt.plot(0, 0, "ko")
        plt.plot(pi / 2, 0, "ko")
        plt.plot(0, pi / 2, "ko")

        vec_labels = [
            f"n0={np.round(new_base[0])}",
            f"e1={np.round(new_base[1])}",
            f"e2={np.round(new_base[2])}",
        ]
        # UnitDirection = UnitDirection(base0)
        norm_angle = UnitDirection(base0).from_vector(new_base[0]).as_angle()
        # colors = ['

        # for ii in range(3):
        for ii in range(len(vec_labels)):
            # angle = get_angle_from_vector(direction=null_matrix, base=base0)
            angle = UnitDirection(base0).from_vector(new_base[ii]).as_angle()
            plt.plot(angle[0], angle[1], "o", label=vec_labels[ii])

            # angle_hat = angle - norm_angle
            # plt.plot(angle_hat[0], angle_hat[1], 'ko', alpha=0.3, label=vec_labels[ii])
        angle = direction0.as_angle()
        plt.plot(angle[0], angle[1], "ro", label=f"Vector {direction0.as_vector()}")

        plt.axis("equal")
        plt.legend()
        plt.xlim([-2 * pi, 2 * pi])
        plt.ylim([-2 * pi, 2 * pi])
        plt.grid()

        plt.subplot(1, 2, 2)

        direction0.transform_to_base(new_base=new_base)
        angle = direction0.as_angle()

        plt.plot(0.5 * pi * np.cos(angles), 0.5 * pi * np.sin(angles), "k")
        # plt.plot(0, 0, 'ko')
        # plt.plot(pi/2, 0, 'ko')
        # plt.plot(0, pi/2, 'ko')
        plt.axis("equal")
        # plt.legend()
        plt.xlim([-2 * pi, 2 * pi])
        plt.ylim([-2 * pi, 2 * pi])
        plt.grid()

        plt.plot(
            angle[0],
            angle[1],
            "ro",
            label=f"Vector {np.round(direction0.as_angle(), 1)}",
        )

        plt.ion()
        plt.show()

    # def test_directional_convergence_forcing(self):
    # """ Based on Reference direction & normal decomposition force the convergence. """


def test_base_transform_2d():
    """Transform initial vector."""
    # Test for two dimensions
    base = get_orthogonal_basis(np.array([1, 0]))
    dir_0 = UnitDirection(base).from_angle([np.pi / 4])

    dir_new = dir_0._get_unitdirection_relative_to_angle(
        new_base_angle=np.array([-np.pi / 4])
    )

    assert np.allclose(dir_0.as_vector(), dir_new.as_vector())
    assert LA.matrix_rank(dir_new.null_matrix) == len(
        base
    ), "Matrix does not have full rank."

    # No Trafo -> same vector
    base = get_orthogonal_basis(np.array([1, 0]))
    dir_0 = UnitDirection(base).from_angle([np.pi / 4])

    dir_new = dir_0._get_unitdirection_relative_to_angle(new_base_angle=np.array([0]))

    assert np.allclose(dir_0.as_vector(), dir_new.as_vector())
    assert np.allclose(LA.norm(dir_0.as_angle()), LA.norm(dir_new.as_angle()))
    assert LA.matrix_rank(dir_new.null_matrix) == len(
        base
    ), "Matrix does not have full rank."

    # Same Direction
    base = get_orthogonal_basis(np.array([1, 0]))
    dir_0 = UnitDirection(base).from_angle([np.pi / 2])

    dir_new = dir_0._get_unitdirection_relative_to_angle(
        new_base_angle=np.array([np.pi / 4])
    )
    assert np.allclose(dir_0.as_vector(), dir_new.as_vector())
    assert LA.matrix_rank(dir_new.null_matrix) == len(
        base
    ), "Matrix does not have full rank."


def test_base_transform_3d():
    # Test perpendicular
    base = get_orthogonal_basis(np.array([1, 0, 0]))
    dir_0 = UnitDirection(base).from_vector(np.array([0, 1, 0]))

    angle = dir_0.as_angle()
    angle = np.array([(-1) * angle[1], angle[0]])

    dir_new = dir_0._get_unitdirection_relative_to_angle(new_base_angle=angle)

    assert LA.matrix_rank(dir_new.null_matrix) == len(
        base
    ), "Matrix does not have full rank."

    # Test random
    print("Closer to home")
    base = get_orthogonal_basis(np.array([1, 0, 0]))
    dir_0 = UnitDirection(base).from_vector(np.array([0.6, 0.3, 0]))

    angle = dir_0.as_angle()
    angle = np.array([(-1) * angle[1], angle[0]])

    dir_new = dir_0._get_unitdirection_relative_to_angle(new_base_angle=angle)

    assert LA.matrix_rank(dir_new.null_matrix) == len(
        base
    ), "Matrix does not have full rank."

    # Test opposite direction
    base = get_orthogonal_basis(np.array([1, 0, 0]))
    dir_0 = UnitDirection(base).from_vector(np.array([0, 1, 0]))

    dir_new = dir_0._get_unitdirection_relative_to_angle(
        new_base_angle=(-1) * dir_0.as_angle()
    )

    assert np.allclose(dir_0.as_vector(), dir_new.as_vector())
    assert LA.matrix_rank(dir_new.null_matrix) == len(
        base
    ), "Matrix does not have full rank."


if (__name__) == "__main__":
    # Tester = TestDirectionalSpace()
    # Tester.test_special_angle_displacement()
    # Tester.test_check_bijection()

    # test_base_transform_2d()
    test_base_transform_3d()

    # test_base_transform()
    # unittest.main(argv=["first-arg-is-ignored"], exit=False)

    # user_test = False
    # if user_test:
    # Tester = TestDirectionalSpace()
    # Tester.test_base_transform_same_normal()
    # Tester.test_special_angle_displacement()

    # Tester.test_repetitive_nonnorm_influence()
    # Tester.test_base_transform()

    # Tester.visualization_direction_space()
    # Tester.test_90_degree_rotation(visualize=False)

    # Tester.test_check_bijection()
    # Tester.test_check_bijection_rebasing()
    # Tester.test_inversion_and_bijectiveness_3d()

    # Tester.test_comparison_operator_direction_base()
    # Tester.test_mult_operators()
    # Tester.test_comparison_operator_direction_base()

    # Tester.test_inversion_examples()

    # Tester.test_angle_space_distance()
    print("Done")
