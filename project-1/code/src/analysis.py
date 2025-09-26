import numpy as np
from src.validation import bootstrap_coefficients, bootstrap_predictions


def condition_number(X):
    """Compute condition number of matrix X"""
    return np.linalg.cond(X)


def effective_dof(X, alpha=0):
    """Effective degrees of freedom for ridge regression"""
    if alpha == 0:
        return X.shape[1]
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    return np.sum(s**2 / (s**2 + alpha))




def bias_variance_decomp(X, y, X_test, y_test_true, solver, n_trials=100, seed=314, **solver_kwargs):
    """Bias-variance decomposition via bootstrap"""
    bootstrap_preds = bootstrap_predictions(X, y, X_test, solver, n_trials, seed, **solver_kwargs)

    # Mean prediction across bootstrap samples
    mean_pred = np.mean(bootstrap_preds, axis=0)

    # Bias squared
    bias_sq = np.mean((mean_pred - y_test_true) ** 2)

    # Variance
    variance = np.mean(np.var(bootstrap_preds, axis=0))

    # Noise (irreducible error) - estimated as residual
    total_error = np.mean((mean_pred - y_test_true) ** 2)
    noise = total_error - bias_sq - variance

    return {
        'bias_squared': bias_sq,
        'variance': variance,
        'noise': max(0, noise),  # Ensure non-negative
        'total_error': bias_sq + variance + max(0, noise),
        'predictions': bootstrap_preds,
        'mean_prediction': mean_pred
    }


def bootstrap_confidence_intervals(bootstrap_samples, confidence=0.95):
    """Compute confidence intervals from bootstrap samples"""
    alpha = 1 - confidence
    lower_p = (alpha / 2) * 100
    upper_p = (1 - alpha / 2) * 100

    if bootstrap_samples.ndim == 1:
        return np.percentile(bootstrap_samples, [lower_p, upper_p])
    else:
        return np.percentile(bootstrap_samples, [lower_p, upper_p], axis=0)


def coefficient_stability(X, y, solver, n_trials=100, seed=314, **solver_kwargs):
    """Analyze coefficient stability via bootstrap"""
    bootstrap_coefs = bootstrap_coefficients(X, y, solver, n_trials, seed, **solver_kwargs)

    mean_coef = np.mean(bootstrap_coefs, axis=0)
    std_coef = np.std(bootstrap_coefs, axis=0)
    ci_coef = bootstrap_confidence_intervals(bootstrap_coefs)

    return {
        'mean_coefficients': mean_coef,
        'std_coefficients': std_coef,
        'confidence_intervals': ci_coef,
        'bootstrap_coefficients': bootstrap_coefs
    }

