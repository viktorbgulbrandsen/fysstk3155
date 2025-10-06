import numpy as np


def runge_function(x):
    """
    Evaluate the Runge function.

    The Runge function is defined as:
        f(x) = 1 / (1 + 25x^2)

    Parameters
    ----------
    x : array_like or float
        Input value(s).

    Returns
    -------
    ndarray or float
        The value(s) of the Runge function at `x`.
    """
    return 1 / (1 + 25 * x**2)


def equispaced_points(n):
    """
    Generate equispaced points in the interval [-1, 1].

    Parameters
    ----------
    n : int
        Number of points.

    Returns
    -------
    ndarray
        Array of shape (n,) with evenly spaced points in [-1, 1].
    """
    return np.linspace(-1, 1, n)

