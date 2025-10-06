import numpy as np


def vandermonde(x, degree):
    """Create Vandermonde matrix for polynomial basis up to given degree"""
    x = np.asarray(x).reshape(-1, 1)
    powers = np.arange(degree + 1)
    return x ** powers


