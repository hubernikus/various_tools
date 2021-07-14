"""
Directional Space function to use
Helper function for directional & angle evaluations
"""
# Author: LukasHuber
# Created: 2021-05-18
# Email: lukas.huber@epfl.ch

# Use python 3.10 [annotations / typematching]
from __future__ import annotations  # Not needed from python 3.10 onwards

import warnings
from typing import Callable
from math import pi

import numpy as np
from numpy import linalg as LA

from vartools.linalg import get_orthogonal_basis


def get_angle_from_vector(direction: np.ndarray, base: DirectionBase, cos_margin: float = 1e-5) -> np.ndarray:
    """
    Returns a angle evalauted from the direciton & null_matrix
    
    Parameters
    ---------
    vector: unit vector of dimension (dim,)
    base : null matrix of floats of dimension (dim, dim)

    Returns
    ------
    angle : angle-space value of dimension (dim-1,)
    """
    direction_referenceSpace = base.null_matrix.T.dot(direction)
        
    # Make sure to catch numerical error of cosinus calculation
    cos_direction = direction_referenceSpace[0]
    if cos_direction >= (1.0-cos_margin):
        # Trivial solution
        angle = np.zeros(direction_referenceSpace.shape[0] - 1)
        return angle

    elif cos_direction <= -(1.0-cos_margin):
        # This value has to be used with care, since it's close to signularity.
        # Due to the fact that the present transformation can be used to evaluate the total
        # agnle no 'warning' is raised.
        angle = np.zeros(direction_referenceSpace.shape[0] - 1)
        angle = pi
        return angle

    angle = direction_referenceSpace[1:]
    # No zero-check since non-trivial through previous one.
    # angle = (angle / LA.norm(self._angle))
    angle = (angle / LA.norm(angle))
    angle = angle * np.arccos(cos_direction)
    return angle

def get_vector_from_angle(angle: np.ndarray, base: DirectionBase) -> np.ndarray:
    """
    Returns a unit vector transformed back from the angle/direction-space.
    
    Parameters
    ---------
    angle : angle-space value of dimension (dim-1,)
    null_matrix : null matrix of floats of dimension (dim, dim)

    Returns
    ------
    vector: unit vector of dimension (dim,)
    """
    norm_directionSpace = LA.norm(angle)
    if norm_directionSpace:
        vector = base.null_matrix.dot(
            np.hstack((np.cos(norm_directionSpace),
                       np.sin(norm_directionSpace) * dir_angle_space / norm_directionSpace))
            )
    else:
        vector = base.null_matrix[:, 0]
    return vector
    

class UnitDirection():
    """ Direction of the length 1 which can be respresented in angle space.
    Not that this space is not Eucledian but it """
    def __init__(self, base: DirectionBase):
        """
        To create the angle space on of several 'reference angles / directions' have to be
        pass to the function. 

        Properties
        ----------
        _vector : Each element in this space can be equivalently described as a unit-vector
        _angle : The transformation of the angle space
        
        Parameters
        ----------
        base: DirectionBase
        """
        self.base = base

    def __repr__(self):
        return f"<DirectionBase({str(self._matrix)})>"
    # def __iadd__(self, other):
        # pass
    
    # def __add__(self, other: UnitDirection) -> UnitDirection:
    #     self = copy.deepcopy(self)
    #     if not np.allclose(self.null_matrix, other.null_matrix):
    #         other = copy.deepcopy(other)
    #         other.transform_to_base(self)
            
    #     self._anlge = self.as_angle + other.as_angle()
    #     return self

    def __sub__(self, other: UnitDirection) -> UnitDirection:
        return self + (-1)*other

    def __mul__(self, other: float) -> UnitDirection:
        return self._angle * other
            
    def __rmul__(self, other: UnitDirection) -> UnitDirection:
        return self * other

    def get_shortest_angle(self, other):
        """Get shortesst angle distance between points. """
        pass
    
    @property
    def dimension(self) -> int:
        return self.base.null_matrix.shape[0]

    @property
    def magnitude(self) -> float:
        return LA.norm(self.angle)
    
    @property
    def null_matrix(self) -> np.ndarray:
        return self._base.null_matrix

    @property
    def base(self) -> DirectionBase:
        return self._base
    
    @base.setter
    def base(self, value: DirectionBase) -> None:
        if hasattr(self, '_base'):
            # Reset angles / vector
            self._angle = None
            self._vector = None
            
        self._base = value
            
    def from_angle(self, value: np.ndarray) -> None:
        """ Update angle and reset 'equivalent' vector. """
        self._angle = value
        self._vector = None
        return self
    
    def from_vector(self, value: np.ndarray) -> None:
        """ Update vector and reset angle. """
        self._vector = value
        self._angle = None
        return self

    def as_angle(self, cos_margin: float = 1e-5) -> np.ndarray:
        if self._angle is not None:
            return self._angle
        if self._vector is None:
            raise ValueError("Set vector or angle value before evaluating.")

        # Store & return angle
        self._angle = get_angle_from_vector(direction=self._vector, base=self.base, cos_margin=cos_margin)
        return self._angle
            
    def as_vector(self) -> np.ndarray:
        if self._vector is not None:
            return self._vector
        if self._angle is None:
            raise ValueError("Set vector or angle value before evaluating.")

        # Store & return vector
        try:
            self._vector = get_vector_from_angle(angle=self._angle, base=self.base)
        except:
            breakpoint()
        return self._vector

    def transform_to_base(self, new_base: DirectionBase) -> None:
        """ Rebase to new base."""
        if self.base == new_base:
            return

        # Make sure the angle is calculated
        angle = self.angle

       # Transform the base into the new_space
        new_base_as_angle = np.zeros((dim-1, dim))
        for ii in range(dim):
            new_base_as_angle[:, ii] = get_angle_from_vector(
                vector=self.base.null_matrix[:, ii], base=self.base)

            # Do the 'proximity check' (pi-discontuinity) for the vectors
            # by comparing each vector the the normal of the original base.
            if ii > 0:
                base_norm = LA.norm(new_base_as_angle[:, ii] - new_base_as_angle[:, 0]) 
                if base_norm >= pi:
                    warnings.warn("TODO: Throughfully test this case...")
                    dist_new_base = LA.norm(
                        new_base_as_angle[:, ii] - new_base_as_angle[:, 0])

                    # Project accross origin
                    new_base_opposite_angle = new_base_as_angle[:, ii]/base_norm * (base_norm-2*pi)

                    dist_new_base_opposite = LA.norm(
                        new_base_opposite_angle - new_base_as_angle[:, 0])

                    if dist_new_base_opposite < dist_new_base:
                        warnings.warn("Did a transform. Is this the only case where it happens?")
                        new_base_as_angle[:, ii]= new_base_opposite_angle
                        
        # Create direction within directionspace [subsp =^= sub-space]
        dim_subsp = dim-1

        if self.dimension == 2:
            angle_in_frame = self.as_angle()
            self._angle = angle_to_nulldir + angle_in_frame

        elif self.dimension == 3:
            raise NotImplementedError("You can do this...")
        
        elif self.dimension >= 3:
            raise NotImplementedError("How is it defined for d>3")
        
        self._base = new_base



class DirectionBase():
    def __init__(self, vector: np.ndarray = None,
                 matrix: np.ndarray = None,
                 direction_base: DirectionBase = None,
                 ):
        # Should it be a mutable OR immutable object?
        # TODO MAYBE: tests(?)
        if direction_base is not None:
            self._matrix = np.copy(unit_direction.null_matrix)
            
        elif matrix is not None:
            self._matrix = np.copy(matrix)
            
        elif vector is not None:
            self._matrix = get_orthogonal_basis(vector)
            
        else:
            raise ValueError("No input argument as a base of the space.")

    def __repr__(self):
        return f"<DirectionBase({str(self._matrix)})>"
    
    def __eq__(self, other: DirectionBase):
        return np.allclose(self._matrix, other.null_matrix)

    @property
    def null_matrix(self):
        return self._matrix

    # @null_matrix.setter
    # def null_matrix(self, value):
        # self._matrix = value
