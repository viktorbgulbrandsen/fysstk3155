help me write about the notebooks "#### Group Members:  Viktor Braaten Gulbrandsen  #### Project Description  In this project, we look at regression for the oscilliatory behavior of the Runge function at $[-1,1]$. OLS, Ridge, Lasso and Gradient Descent are applied, and the nuance of their behavior are explored through the lens of statistical learning, numerical approximation and linear algebra.  #### Installation  In order to install the project, simply run:  ```bash git clone https://github.com/viktorbgulbrandsen/fysstk3155 cd project-1 pip install -r requirements.txt #we suggest making a venv first ```   #### Notebooks:  Currently, in code/notebooks/uke39 the notebook for week39.ipynb can be found. This will be updated with notebooks replicating and exploring the results of the project tasks.   " in my README.md 

**`conditioning.ipynb`** – Explores Vandermonde matrix conditioning and spectral degradation. Visualizes how condition numbers and singular values collapse as polynomial degree increases, plus regularization effects.
**`Least_Square_validation.ipynb`** – Compares OLS solvers (Normal Equations, QR, SVD) and gradient descent variants (Batch GD, SGD, Momentum, Adagrad, RMSProp, Adam) using training loss, holdout validation, k-fold CV, and bootstrap. Includes spectral tail energy analysis showing how different optimizers distribute coefficient energy across singular vectors at varying iteration counts.
**`OLS_bias_variance_plot.ipynb`** – Analyzes bias–variance tradeoff in polynomial regression on the Runge function over ([-1,1]). Fits OLS models of varying degrees and visualizes bias², variance, and total error, showing how increasing complexity leads to overfitting.
**`Ridge_bias_variance_plot.ipynb`** – Extends the bias–variance analysis to Ridge regression. Demonstrates how the regularization parameter stabilizes high-degree polynomials and suppresses variance without drastically increasing bias.
**`Ridge_Validation.ipynb`** – Evaluates Ridge regression through validation curves, comparing test error across polynomial degrees and regularization strengths. Illustrates the balance between underfitting and overfitting as λ varies.

**`basis.py`** – Generates polynomial design matrices (Vandermonde basis) up to a specified degree.
**`bias_variance.py`** – Performs Monte Carlo bias–variance decomposition via bootstrap resampling.
**`data.py`** – Provides the Runge function and utilities for generating equispaced sample points.
**`gd.py`** – Implements gradient-based solvers: batch, stochastic, momentum, adaptive (Adam, RMSProp, Adagrad) and LASSO-specific proximal variants.
**`regression.py`** – Contains closed-form and decomposition-based OLS and Ridge regressors (Normal, QR, SVD, sklearn).
**`validation.py`** – Includes validation and resampling methods: bootstrap, k-fold CV, residual bootstrap, and deterministic holdout.


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
