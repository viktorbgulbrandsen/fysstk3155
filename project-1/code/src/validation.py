import numpy as np


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


def bootstrap_coefficients(X, y, solver, n_samples=1000, seed=314, **solver_kwargs):
    """Bootstrap coefficient estimates"""
    bootstrap_coefs = []
    indices_list = bootstrap_indices(len(X), n_samples, seed)

    for indices in indices_list:
        X_boot = X[indices]
        y_boot = y[indices]
        coef, _, _, _ = solver(X_boot, y_boot, **solver_kwargs)
        bootstrap_coefs.append(coef)

    return np.array(bootstrap_coefs)


def bootstrap_predictions(X, y, X_test, solver, n_samples=1000, seed=314, **solver_kwargs):
    """Bootstrap prediction estimates"""
    bootstrap_preds = []
    indices_list = bootstrap_indices(len(X), n_samples, seed)

    for indices in indices_list:
        X_boot = X[indices]
        y_boot = y[indices]
        coef, predict_fn, _, _ = solver(X_boot, y_boot, **solver_kwargs)
        y_pred = predict_fn(X_test, coef)
        bootstrap_preds.append(y_pred)

    return np.array(bootstrap_preds)


def cross_validate_scores(X, y, solver, k=5, seed=314, **solver_kwargs):
    """Cross-validation scores"""
    folds = k_fold_split(len(X), k, seed)
    scores = []

    for train_idx, test_idx in folds:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        coef, predict_fn, _, _ = solver(X_train, y_train, **solver_kwargs)
        y_pred = predict_fn(X_test, coef)
        score = mse(y_test, y_pred)
        scores.append(score)

    return np.array(scores)


def grid_search_cv(X, y, solver, param_grid, k=5, seed=314):
    """Grid search with cross-validation"""
    best_score = float('inf')
    best_params = None
    results = []

    for params in param_grid:
        scores = cross_validate_scores(X, y, solver, k, seed, **params)
        mean_score = np.mean(scores)
        results.append({'params': params, 'scores': scores, 'mean_score': mean_score})

        if mean_score < best_score:
            best_score = mean_score
            best_params = params

    return best_params, best_score, results


# Metrics
def mse(y_true, y_pred):
    """Mean squared error"""
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    """Root mean squared error"""
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    """Mean absolute error"""
    return np.mean(np.abs(y_true - y_pred))


def r2(y_true, y_pred):
    """R-squared"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def adjusted_r2(y_true, y_pred, n_features):
    """Adjusted R-squared"""
    n = len(y_true)
    r2_val = r2(y_true, y_pred)
    return 1 - (1 - r2_val) * (n - 1) / (n - n_features - 1)


def condition_number(X):
    """Calculate condition number of matrix X"""
    return np.linalg.cond(X)


