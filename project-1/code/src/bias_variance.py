import numpy as np
from .validation import bootstrap_predictions

def bias_variance_decomp(X, y, X_test, y_test_true, solver, n_trials=200, seed=314, **solver_kwargs):
    """
    Monte Carlo bias-variance decomposition.
    Approximates the expected generalization error by resampling training sets.
    """
    preds = bootstrap_predictions(X, y, X_test, solver, n_trials, seed, **solver_kwargs)

    # Average prediction across resamples
    mean_pred = np.mean(preds, axis=0)

    # Bias squared (average across test points)
    bias_sq = np.mean((mean_pred - y_test_true) ** 2)

    # Variance (average predictive variance across test points)
    variance = np.mean(np.var(preds, axis=0))

    # Total error = bias² + variance
    total_error = bias_sq + variance

    return {
        "bias_squared": bias_sq,
        "variance": variance,
        "total_error": total_error,
        "predictions": preds,
        "mean_prediction": mean_pred,
    }


def compute_bias_variance(x, y, x_grid, y_true, degrees, solver,
                          trials=200, seed=314, **solver_kwargs):
    """Run bias–variance decomposition across polynomial degrees."""
    from .basis import vandermonde

    bias2_list, var_list, tot_list = [], [], []

    for d in degrees:
        X = vandermonde(x, d)
        Xg = vandermonde(x_grid, d)

        res = bias_variance_decomp(
            X, y, Xg, y_true,
            solver,
            n_trials=trials,
            seed=seed,
            **solver_kwargs   # <- forward here
        )
        bias2_list.append(res["bias_squared"])
        var_list.append(res["variance"])
        tot_list.append(res["total_error"])

    return np.array(bias2_list), np.array(var_list), np.array(tot_list)
