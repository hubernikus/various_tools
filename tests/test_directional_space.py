#!/USSR/bin/python3.9
""" Test the directional space. """
# Author: LukasHuber
# Date: 2021-05-18
# Email: lukas.huber@epfl.ch

import unittest
import copy

import numpy as np

from vartools.linalg import get_orthogonal_basis

from vartools.directional_space import get_angle_space
from vartools.directional_space import get_angle_space_inverse

from vartools.directional_space import UnitDirection


class TestDirectionalSpace(unittest.TestCase):
    def test_orthonormality_matrix(self):
        margin = 1e-6
        max_dim = 50
        
        n_repetitions = 100
        for ii in range(n_repetitions):
            dim = np.random.randint(2, max_dim)
            null_direction = np.random.normal(loc=0.0, scale=10.0, size=dim)

            vec = get_orthogonal_basis(null_direction)
            
            for jj in range(dim):
                for kk in range(jj+1, dim):
                    self.assertTrue(vec[:, jj].dot(vec[:, kk]) < margin)

    def test_bijectional_space_2D(self):
        """ Test that forward&inverse directional space is ennaluating.""" 
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

                self.assertTrue(all(np.isclose(vec_init / np.linalg.norm(vec_init), vec_init_rec)))

    def test_construction_angle_value(self):
        pass
    
    def test_special_angle_displacement(self):
        """ Test how the displacement behaves when going from space 1 to space 2. """
        null_direction = np.array([1, 0, 0])
        direction0 = UnitDirection(null_direction=null_direction)
        
        direction0.from_angle([0, 0])
        print('vector 0', direction0.as_vector())
        print('base 0 \n', direction0.null_matrix)

        null_direction = np.array([0, 1, 0])
        null_matrix = np.array([[ 0, 1, 0],
                                [-1, 0, 0],
                                [0,  0, 1]])
        direction1 = UnitDirection(null_matrix=null_matrix)
        print('base 1 \n', direction1.null_matrix)
        
        direction1.transform_to_base(null_matrix)

    
    # def test_directional_convergence_forcing(self):
        # """ Based on Reference direction & normal decomposition force the convergence. """

if __name__ == '__main__':
    # unittest.main()

    user_test = True
    if user_test:
        Tester = TestDirectionalSpace()
        Tester.test_special_angle_displacement()
