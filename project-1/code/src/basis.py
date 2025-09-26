import numpy as np


def vandermonde(x, degree):
    """Create Vandermonde matrix for polynomial basis up to given degree"""
    x = np.asarray(x).reshape(-1, 1)
    powers = np.arange(degree + 1)
    return x ** powers


def chebyshev(x, degree):
    """Create Chebyshev polynomial basis matrix up to given degree"""
    x = np.asarray(x)
    n_points = len(x)
    T = np.zeros((n_points, degree + 1))

    # T_0(x) = 1
    T[:, 0] = 1

    if degree > 0:
        # T_1(x) = x
        T[:, 1] = x

        # T_n(x) = 2x * T_{n-1}(x) - T_{n-2}(x)
        for n in range(2, degree + 1):
            T[:, n] = 2 * x * T[:, n-1] - T[:, n-2]

    return T


def legendre(x, degree):
    """Create Legendre polynomial basis matrix up to given degree"""
    x = np.asarray(x)
    n_points = len(x)
    P = np.zeros((n_points, degree + 1))

    # P_0(x) = 1
    P[:, 0] = 1

    if degree > 0:
        # P_1(x) = x
        P[:, 1] = x

        # P_n(x) = ((2n-1) * x * P_{n-1}(x) - (n-1) * P_{n-2}(x)) / n
        for n in range(2, degree + 1):
            P[:, n] = ((2*n - 1) * x * P[:, n-1] - (n - 1) * P[:, n-2]) / n

    return P

