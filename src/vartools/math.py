""" Various tools to help / and speed up."""

from typing import Callable
import numpy as np

def get_numerical_gradient(position: np.ndarray, function: Callable[[np.ndarray], float], 
                           delta_magnitude: float = 1e-6) -> np.ndarray:
    """ Returns the numerical derivative of an input function at the specified position."""
    dimension = position.shape[0]
    
    vec_low = np.zeros(dimension)
    vec_high = np.zeros(dimension)
    for ii in range(dimension):
        delta_vec = np.zeros(dimension)
        delta_vec[ii] = delta_magnitude/2.0
        
        vec_low[ii] = function(position-delta_vec)
        vec_high[ii] = function(position+delta_vec)
    return (vec_high-vec_low)/delta_magnitude

def get_numerical_hessian_fast(position: np.ndarray, function,
                               # function: Callable[[np.ndarray], float],
                               delta_magnitude: float = 1e-6) -> np.ndarray:
    """ Returns (numerical) Hessian-Matrix of 'function' at 'position'.
    Calculation is speed up, by going in positive-delta direction only. """

    dimension = position.shape[0]

    hessian = np.zeros((dimension, dimension))
    f_eval = function(position)
    f_dx_eval = np.zeros(dimension)

    pos_deltas = np.eye(dimension)*delta_magnitude
    
    for ix in range(dimension):
        f_dx_eval[ix] = function(position + pos_deltas[:, ix])
        for iy in range(0, ix+1):
            f_dx_dx_eval = function( position + pos_deltas[:, ix] + pos_deltas[:, iy])

            hessian[ix, iy] = ((f_dx_dx_eval - f_dx_eval[ix] - f_dx_eval[iy] + f_eval)
                               / (delta_magnitude*delta_magnitude))
            if ix != iy:
                hessian[iy, ix] = hessian[ix, iy]
    return hessian

def get_numerical_hessian(position: np.ndarray, function,
                          # function: Callable[[np.ndarray], float],
                          delta_magnitude: float = 1e-6) -> np.ndarray:
    """ Returns (numerical) Hessian-Matrix of 'function' at 'position'.
    Calculation is centered around position."""
    dimension = position.shape[0]

    hessian = np.zeros((dimension, dimension))
    pos_deltas = np.eye(dimension)*(delta_magnitude*0.5)
    for ix in range(dimension):
        for iy in range(0, ix+1):
            f_dx_dy_eval = function(position + pos_deltas[:, ix] + pos_deltas[:, iy])
            f_ndx_dy_eval = function(position - pos_deltas[:, ix] + pos_deltas[:, iy])
            f_dx_ndy_eval = function(position + pos_deltas[:, ix] - pos_deltas[:, iy])
            f_ndx_ndy_eval = function(position - pos_deltas[:, ix] - pos_deltas[:, iy])

            hessian[ix, iy] = ((f_dx_dy_eval - f_ndx_dy_eval - f_dx_ndy_eval + f_ndx_ndy_eval)
                               / (delta_magnitude*delta_magnitude))
            if ix != iy:
                hessian[iy, ix] = hessian[ix, iy]
    return hessian

def get_scaled_orthogonal_projection(vector):
    """ Returns scaled orthogonal projection of the form  P_v = ||v||^2 I_n − v v^T.
    
    It has following properties:
    (i) P_v = P_v^T (symmetry)
    (ii) P_v 2 = ||v||2 P_v 
    (iii) the spectrum of P_v is composed of 0 and ||v||^2
          with algebraic multiplicity 1 and n − 1, respectively
    (iv) P_v z = ||v||^2 z for all z ∈ R n on the projective subspace defined by v∈R_n 
    (v) P_v w = 0 for all w ∈ R n such that vw;
    (vi) 12 w^T Ṗ_v w = v^T P_w v̇."""
    return LA.norm(vector)*np.eye(vector.shape[0]) - vector @ vector.T


