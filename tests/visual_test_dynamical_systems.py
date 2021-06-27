#!/usr/bin/python3
"""
Visual test to evaluate the dynamicals systems.
"""

__author__ = "LukasHuber"
__date__ = "2021-05-18"
__email__ = "lukas.huber@epfl.ch"

import numpy as np
import matplotlib.pyplot as plt

from vartools.dynamicalsys.closedform import evaluate_linear_dynamical_system
from vartools.dynamicalsys.closedform import evaluate_stable_circle_dynamical_system




if (__name__) == "__main__":
    plt.close('all')
    plt.ion()  # Interactive plotting
    
    plot_dynamical_system(evaluate_linear_dynamical_system)

    plot_dynamical_system(
        func=evaluate_stable_circle_dynamical_system,
        func_kwargs={'radius': 8})
