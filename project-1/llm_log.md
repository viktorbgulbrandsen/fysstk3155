User: 
Create nice docstrings for these functions [all functions in src]

ChatGPT:
Here’s your code with precise, clean docstrings added:

```python
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


def random_points(n, seed=314):
    """
    Generate random points uniformly distributed in [-1, 1].

    Parameters
    ----------
    n : int
        Number of points.
    seed : int, optional
        Random seed for reproducibility (default is 314).

    Returns
    -------
    ndarray
        Array of shape (n,) with random points in [-1, 1].
    """
    np.random.seed(seed)
    return np.random.uniform(-1, 1, n)


def chebyshev_points(n):
    """
    Generate Chebyshev nodes in the interval [-1, 1].

    Chebyshev nodes are defined as:
        x_k = cos((2k - 1)π / (2n)),  for k = 1,...,n

    Parameters
    ----------
    n : int
        Number of nodes.

    Returns
    -------
    ndarray
        Array of shape (n,) with Chebyshev nodes in [-1, 1].
    """
    k = np.arange(1, n + 1)
    return np.cos((2*k - 1) * np.pi / (2*n))


def generate_runge_data(n, method=equispaced_points, seed=314):
    """
    Generate sample data (x, y) from the Runge function.

    Parameters
    ----------
    n : int
        Number of sample points.
    method : callable, optional
        Function to generate sample points. Should accept `n` as an argument.
        Defaults to `equispaced_points`. If `random_points` is used, the
        `seed` parameter will also be passed.
    seed : int, optional
        Random seed (only used if `method` is `random_points`).

    Returns
    -------
    tuple of ndarray
        (x, y) where:
        - x : shape (n,), sample points
        - y : shape (n,), Runge function values at x
    """
    if method == random_points:
        x = method(n, seed)
    else:
        x = method(n)

    y = runge_function(x)
    return x, y


def train_test_split(x, y, test_ratio=0.2, seed=314):
    """
    Split data into training and test sets.

    Parameters
    ----------
    x : ndarray
        Input features, shape (n,).
    y : ndarray
        Target values, shape (n,).
    test_ratio : float, optional
        Fraction of data to use as test set (default is 0.2).
    seed : int or None, optional
        Random seed for reproducibility. If None, randomness is uncontrolled.

    Returns
    -------
    tuple of ndarray
        (x_train, x_test, y_train, y_test) where each is a 1D array.
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(x)
    indices = np.random.permutation(n)
    split_idx = int(n * (1 - test_ratio))

    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]
```

Do you want me to also make a **module-level docstring** (for the whole file), summarizing that this is a Runge function dataset generator with sampling methods and splitting utilities?
