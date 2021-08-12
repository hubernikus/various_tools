""" Various tools to help / and speed up """

from typing import Callable

import numpy as np

def get_numerical_gradient(position: np.ndarray, function: Callable[[np.ndarray], float], 
                           delta_magnitude: float = 1e-6):
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
