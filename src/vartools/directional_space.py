"""
Directional Space function to use
Helper function for directional & angle evaluations
"""
# Author: LukasHuber
# Date: 2021-05-18
# Email: lukas.huber@epfl.ch

import warnings
from math import pi

import numpy as np

from vartools.linalg import get_orthogonal_basis

def logical_xor(value1, value2):
    return bool(value1) ^ bool(value2)

class UnitDirection():
    """ Direction of the length 1 which can be respresented in angle space. """
    def __init__(self, OtherDirection=None, null_matrix=None, unit_direction=None):
        """
        To create the angle space on of several 'reference angles / directions' have to be
        pass to the function. 
        
        Parameters
        ----------
        OtherDirection : Argument of type 'UnitDirection', copy the null_matrix
        unit_direction : Orthonormal float array of size (dimension, dimension)
        null_matrix : Reference vector of size (dimension,)
        """
        if OtherDirection is not None:
            self.null_matrix = np.copy(OtherDirection.null_matrix)
        if null_matrix is not None:
            self.null_matrix = np.copy(null_matrix)
        elif null_direction is not None:
            self.null_matrix = get_orthogonal_basis(null_direction)
        else:
            raise ValueError("No input argument as a base of the space.")

    @property
    def dimension(self):
        return self._null_matrix.shape[0]
    
    @property
    def null_matrix(self):
        return self._null_matrix
    
    @null_matrix.setter
    def null_matrix(self, value):
        if hasattr(self, '_null_matrix'):
            # Reset angles
            self._angle = None
            self._vector = None
        
        self._null_matrix = value
            
    def from_angle(self, value):
        """ Update angle and reset 'equivalent' vector. """
        self._angle = value
        self._vector = None
    
    def from_vector(self, None):
        """ Update vector and reset angle. """
        self._vector = value
        self._angle = None
        
    def as_angle(self, cos_margin=1e-5):
        if self._angle is not None:
            return self._angle
        
        if self._vector is None:
            raise ValueError("Set value before evaluating.")
        
        direction_referenceSpace = null_matrix.T.dot(direction)

        # Make sure to catch numerical error of cosinus calculation
        cos_direction = direction_referenceSpace[0]
        if cos_direction >= (1.0-cos_margin):
            # Trivial solution
            self._angle = np.zeros(direction_referenceSpace.shape[0] - 1)
            return self._angle
        
        elif cos_direction <= -(1.0-cos_margin):
            # This value has to be used with care, since it's close to signularity.
            # Due to the fact that the present transformation can be used to evaluate the total
            # agnle no 'warning' is raised.
            self._angle = np.zeros(direction_referenceSpace.shape[0] - 1)
            self._angle[0] = pi
            return self._angle
        
        self._angle = direction_referenceSpace[1:]
        # No zero-check since non-trivial through previous one.
        self._angle = (self._angle
                       /np.linalg.norm(self._angle))
        self._angle = self._angle * np.arccos(cos_direction)
        
        return self._angle

    def as_vector(self):
        if self._vector is not None:
            return self._vector
        
        if self._angle is None:
            raise ValueError("Set value before evaluating.")
        
        norm_directionSpace = np.linalg.norm(dir_angle_space)
        if norm_directionSpace:
            self._vector = null_matrix.dot(
                np.hstack((np.cos(norm_directionSpace),
                           np.sin(norm_directionSpace) * dir_angle_space / norm_directionSpace))
                )
        else:
            self._vector = null_matrix[:, 0]
        return self._vector

    def transform_to_base(self, null_matrix):
        raise NotImplementedError()
    
    
def get_angle_space_of_array(directions,
                             positions=None, func_vel_default=None,
                             null_direction_abs=None):
    """ Get the angle space for a whole array. """
    dim = directions.shape[0]
    num_samples = directions.shape[1]

    direction_space = np.zeros((dim-1, num_samples))
    for ii in range(num_samples):
        # Nominal Velocity / Null direction is evaluated each time
        if null_direction_abs is None:
            vel_default = func_vel_default(positions[:, ii])
        else:
            vel_default = null_direction_abs
        direction_space[:, ii] = get_angle_space(directions[:, ii], null_direction=vel_default)

    return direction_space


def get_angle_space(direction, null_direction=None, null_matrix=None, normalize=None,
                    OrthogonalBasisMatrix=None):
    """ Get the direction transformed to the angle space with respect to the 'null' direction."""
    if OrthogonalBasisMatrix is not None:
        raise TypeError("OrthogonalBasisMatrix is depreciated, use 'null_matrix' instead.")
    
    if normalize is not None:
        warnings.warn("The use of normalized is depreciated.")

    if len(direction.shape) > 1:
        raise ValueError("No array of direction accepted anymore")
    
    norm_dir = np.linalg.norm(direction)
    if not norm_dir:    
        return np.zeros(direction.shape[0] - 1)
    direction = direction / norm_dir

    if null_matrix is None:
        null_matrix = get_orthogonal_basis(null_direction)

    direction_referenceSpace = null_matrix.T.dot(direction)

    # Make sure to catch numerical error of cosinus calculation
    cos_margin=1e-5
    cos_direction = direction_referenceSpace[0]
    if cos_direction >= (1.0-cos_margin):
        # Trivial solution
        return np.zeros(direction_referenceSpace.shape[0] - 1)
    elif cos_direction <= -(1.0-cos_margin):
        # This value has to be used with care, since it's close to signularity.
        # Due to the fact that the present transformation can be used to evaluate the total
        # agnle no 'warning' is raised.
        default_dir = np.zeros(direction_referenceSpace.shape[0] - 1)
        default_dir[0] = pi
        return default_dir
        
    direction_directionSpace = direction_referenceSpace[1:]
    # No zero-check since non-trivial through previous one.
    direction_directionSpace = direction_directionSpace / np.linalg.norm(direction_directionSpace)
    
    direction_directionSpace = direction_directionSpace * np.arccos(cos_direction)

    if any(np.isnan(direction_directionSpace)):
        breakpoint() # TODO: remove after debugging

    return direction_directionSpace


def get_angle_space_inverse_of_array(vecs_angle_space, positions, func_vel_default):
    """ Get the angle space for a whole array. """
    # breakpoint()
    dim = positions.shape[0]
    num_samples = positions.shape[1]

    directions = np.zeros((dim, num_samples))
    
    for ii in range(num_samples):
        vel_default = func_vel_default(positions[:, ii])
        directions[:, ii] = get_angle_space_inverse(vecs_angle_space[:, ii], null_direction=vel_default)
        
    return directions


def get_angle_space_inverse(dir_angle_space, null_direction=None, null_matrix=None, NullMatrix=None):
    """
    Inverse angle space transformation
    """
    # TODO: currently for one vector. Is multiple vectors desired (?)
    if NullMatrix is not None:
        warnings.warn("'NullMatrix' is depreciated use 'null_matrix' instead.")
        null_matrix = NullMatrix
        
    if null_matrix is None:
        null_matrix = get_orthogonal_basis(null_direction)

    norm_directionSpace = np.linalg.norm(dir_angle_space)
    if norm_directionSpace:
        directions = null_matrix.dot(
            np.hstack((np.cos(norm_directionSpace),
                       np.sin(norm_directionSpace) * dir_angle_space / norm_directionSpace))
            )
    else:
        directions = null_matrix[:, 0]

    return directions


def get_directional_weighted_sum(null_direction, directions, weights,
                                 total_weight=1, normalize=True, normalize_reference=True):
    """ Weighted directional mean for inputs vector ]-pi, pi[ with respect to the null_direction

    Parameters
    ----------
    null_direction: basis direction for the angle-frame
    directions: the directions which the weighted sum is taken from
    weights: used for weighted sum
    total_weight: [<=1] 
    normalize: variable of type Bool to decide if variables should be normalized

    Return
    ------
    summed_velocity: The weighted sum transformed back to the initial space
    """
    ind_nonzero = (weights>0) # non-negative

    null_direction = np.copy(null_direction)
    directions = directions[:, ind_nonzero] 
    weights = weights[ind_nonzero]

    if total_weight<1:
        weights = weights/np.sum(weights) * total_weight

    n_directions = weights.shape[0]
    if (n_directions==1) and total_weight>=1:
        return directions[:, 0]

    dim = np.array(null_direction).shape[0]

    if normalize_reference:
        norm_refDir = np.linalg.norm(null_direction)
        if norm_refDir==0: # nonzero
            raise ValueError("Zero norm direction as input")
        null_direction /= norm_refDir

     # TODO - higher dimensions
    if normalize:
        norm_dir = np.linalg.norm(directions, axis=0)
        ind_nonzero = (norm_dir>0)
        directions[:, ind_nonzero] = directions[:, ind_nonzero]/np.tile(norm_dir[ind_nonzero], (dim, 1))

    null_matrix = get_orthogonal_basis(null_direction)

    directions_referenceSpace = np.zeros(np.shape(directions))
    for ii in range(np.array(directions).shape[1]):
        directions_referenceSpace[:,ii] = null_matrix.T.dot( directions[:,ii])

    directions_directionSpace = directions_referenceSpace[1:, :]

    norm_dirSpace = np.linalg.norm(directions_directionSpace, axis=0)
    ind_nonzero = (norm_dirSpace > 0)

    directions_directionSpace[:,ind_nonzero] = (directions_directionSpace[:, ind_nonzero] /  np.tile(norm_dirSpace[ind_nonzero], (dim-1, 1)))

    cos_directions = directions_referenceSpace[0,:]
    if np.sum(cos_directions > 1) or np.sum(cos_directions < -1):
        # Numerical error correction
        cos_directions = np.min(np.vstack((cos_directions, np.ones(n_directions))), axis=0)
        cos_directions = np.max(np.vstack((cos_directions, -np.ones(n_directions))), axis=0)
        # warnings.warn("Cosinus value out of bound.") 

    directions_directionSpace *= np.tile(np.arccos(cos_directions), (dim-1, 1))

    direction_dirSpace_weightedSum = np.sum(directions_directionSpace* np.tile(weights, (dim-1, 1)), axis=1)

    norm_directionSpace_weightedSum = np.linalg.norm(direction_dirSpace_weightedSum)

    if norm_directionSpace_weightedSum:
        direction_weightedSum = (null_matrix.dot(
                                  np.hstack((np.cos(norm_directionSpace_weightedSum),
                                              np.sin(norm_directionSpace_weightedSum) / norm_directionSpace_weightedSum * direction_dirSpace_weightedSum)) ))
    else:
        direction_weightedSum = null_matrix[:,0]

    return direction_weightedSum


def get_angle_space_array_old(directions, null_direction, null_matrix=None, normalize=True, OrthogonalBasisMatrix=None):
    """ Get the directions transformed to the angle space with respect 
    """
    # TODO: is this still needed or rather depreciated (?)
    dim = np.array(directions).shape[0]
    
    if len(directions.shape)==1:
        num_dirs = None
        directions = directions.reshape(dim, 1)
    else:
        num_dirs = directions.shape[1]
        
    directions = np.copy(directions)

    if normalize:
        norm_dir = np.linalg.norm(directions, axis=0)
        ind_nonzero = (norm_dir>0)
        directions[:, ind_nonzero] = directions[:, ind_nonzero]/np.tile(norm_dir[ind_nonzero], (dim, 1))

    if OrthogonalBasisMatrix is not None:
        warnings.warn("OrthogonalBasisMatrix is depreciated, use 'null_matrix' instead.")
        null_matrix = OrthogonalBasisMatrix
        
    if null_matrix is None:
        null_matrix = get_orthogonal_basis(null_direction)

    directions_referenceSpace = np.zeros(np.shape(directions))
    for ii in range(np.array(directions).shape[1]):
        directions_referenceSpace[:,ii] = null_matrix.T.dot( directions[:,ii])

    # directions_referenceSpace = np.zeros(np.shape(directions))
    # for ii in range(np.array(directions).shape[1]):
        # directions_referenceSpace[:,ii] = null_matrix.T.dot( directions[:,ii])

    directions_directionSpace = directions_referenceSpace[1:, :]

    norm_dirSpace = np.linalg.norm(directions_directionSpace, axis=0)
    ind_nonzero = (norm_dirSpace > 0)

    directions_directionSpace[:,ind_nonzero] = (directions_directionSpace[:, ind_nonzero] /  np.tile(norm_dirSpace[ind_nonzero], (dim-1, 1)))

    cos_directions = directions_referenceSpace[0,:]
    if np.sum(cos_directions > 1) or np.sum(cos_directions < -1):
        cos_directions = np.min(np.vstack((cos_directions, np.ones(directions.shape[1]))), axis=0)
        cos_directions = np.max(np.vstack((cos_directions, -np.ones(directions.shape[1]))), axis=0)
        warnings.warn("Cosinus value out of bound.")

    directions_directionSpace *= np.tile(np.arccos(cos_directions), (dim-1, 1))

    # directions_directionSpace *= (-1) # in 2D for convention 

    if num_dirs is None:
        directions_directionSpace = np.reshape(directions_directionSpace, (dim-1))
        
    return directions_directionSpace
