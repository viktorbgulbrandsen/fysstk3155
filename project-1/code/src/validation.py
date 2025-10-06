# validation.py
import numpy as np
from .basis import vandermonde


def k_fold_split(n, k=5, seed=314):
    """Generate k-fold cross-validation indices"""
    np.random.seed(seed)
    indices = np.random.permutation(n)
    fold_sizes = np.full(k, n // k)
    fold_sizes[:n % k] += 1
    
    current = 0
    folds = []
    for fold_size in fold_sizes:
        test_idx = indices[current:current + fold_size]
        train_idx = np.concatenate([indices[:current], indices[current + fold_size:]])
        folds.append((train_idx, test_idx))
        current += fold_size
    
    return folds


def bootstrap_indices(n, n_samples=1000, seed=314):
    """Generate bootstrap sample indices"""
    np.random.seed(seed)
    return [np.random.choice(n, size=n, replace=True) for _ in range(n_samples)]


def residual_bootstrap_samples(X, y, theta_fit, n_samples=30, seed=314):
    """Generate residual bootstrap samples - keeps X fixed"""
    np.random.seed(seed)
    residuals = y - X @ theta_fit
    samples = []
    for _ in range(n_samples):
        resampled_residuals = np.random.choice(residuals, size=len(y), replace=True)
        y_boot = X @ theta_fit + resampled_residuals
        samples.append(y_boot)
    return samples


def bootstrap_coefficients(X, y, solver, n_samples=1000, seed=314, **solver_kwargs):
    """Bootstrap coefficient estimates"""
    bootstrap_coefs = []
    indices_list = bootstrap_indices(len(X), n_samples, seed)
    
    for indices in indices_list:
        X_boot = X[indices]
        y_boot = y[indices]
        coef, _, _ = solver(X_boot, y_boot, **solver_kwargs)
        bootstrap_coefs.append(coef)
    
    return np.array(bootstrap_coefs)


def bootstrap_predictions(X, y, X_test, solver, n_samples=1000, seed=314, **solver_kwargs):
    """Bootstrap prediction estimates"""
    bootstrap_preds = []
    indices_list = bootstrap_indices(len(X), n_samples, seed)
    
    for indices in indices_list:
        X_boot = X[indices]
        y_boot = y[indices]
        coef, predict_fn, _ = solver(X_boot, y_boot, **solver_kwargs)
        y_pred = predict_fn(X_test, coef)
        bootstrap_preds.append(y_pred)
    
    return np.array(bootstrap_preds)




def residual_bootstrap(x, y, d, fit, B=50):
    """Residual bootstrap estimate of generalization error."""
    X = vandermonde(x, d)
    theta, predict, _ = fit(X, y)
    residuals = y - predict(X, theta)

    errs = []
    for _ in range(B):
        y_boot = predict(X, theta) + np.random.choice(residuals, size=len(y), replace=True)
        theta_b, predict, _ = fit(X, y_boot)
        errs.append(np.mean((y - predict(X, theta_b))**2))
    return np.mean(errs)


def kfold_cv(x, y, d, fit, K=5):
    """K-fold CV error estimate."""
    X = vandermonde(x, d)
    n = len(y)
    idx = np.arange(n)
    np.random.shuffle(idx)
    folds = np.array_split(idx, K)

    errs = []
    for k in range(K):
        val_idx = folds[k]
        train_idx = np.hstack(folds[:k] + folds[k+1:])
        theta, predict, _ = fit(X[train_idx], y[train_idx])
        errs.append(np.mean((y[val_idx] - predict(X[val_idx], theta))**2))
    return np.mean(errs)




def deterministic_holdout(x, y_true, f_true, d, fit, B=50, sigma=0.1):
    """Train on noisy responses, test on clean grid."""
    X = vandermonde(x, d)
    xg = np.linspace(-1, 1, 400)
    Xg = vandermonde(xg, d)
    yg = f_true(xg)

    errs = []
    for _ in range(B):
        y_noisy = y_true + np.random.normal(0, sigma, size=len(y_true))
        theta, predict, _ = fit(X, y_noisy)
        errs.append(np.mean((yg - predict(Xg, theta))**2))
    return np.mean(errs)
