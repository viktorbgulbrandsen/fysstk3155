import numpy as np

from sklearn.linear_model import LinearRegression, Ridge as SkRidge, Lasso as SkLasso

def predict(X, coef):
    """Simple prediction function"""
    return X @ coef


# OLS Implementations
def ols_normal(X, y):
    """OLS using normal equations with safe pseudoinverse fallback."""
    XtX, Xty = X.T @ X, X.T @ y
    try:
        coef = np.linalg.solve(XtX, Xty)
        method, used_pinv = "normal", False
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(XtX) @ Xty
        method, used_pinv = "pinv", True

    info = {
        "method": method,
        "used_pinv": used_pinv,
        "condition_number": np.linalg.cond(XtX)
    }
    return coef, predict, info



def ols_pinv(X, y):
    """OLS using pseudoinverse"""
    coef = np.linalg.pinv(X) @ y


    info = {'method': 'pinv', 'condition_number': np.linalg.cond(X)}
    return coef, predict, info


def ols_qr(X, y):
    """OLS using QR decomposition"""
    Q, R = np.linalg.qr(X)
    coef = np.linalg.solve(R, Q.T @ y)


    info = {'method': 'qr', 'condition_number': np.linalg.cond(R)}
    return coef, predict, info


def ols_svd(X, y):
    """OLS using SVD decomposition"""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    coef = Vt.T @ (np.diag(1/s) @ (U.T @ y))


    info = {'method': 'svd', 'condition_number': s[0]/s[-1]}
    return coef, predict, info


def ols_sklearn(X, y):
    """OLS using sklearn"""
    reg = LinearRegression(fit_intercept=False).fit(X, y)
    coef = reg.coef_

    info = {'method': 'sklearn'}
    return coef, predict, info


# Ridge Implementations
def ridge_normal(X, y, alpha):
    """Ridge using normal equations"""
    XtX = X.T @ X
    I = np.eye(XtX.shape[0])
    Xty = X.T @ y
    coef = np.linalg.solve(XtX + alpha * I, Xty)

    info = {'method': 'ridge_normal', 'alpha': alpha, 'condition_number': np.linalg.cond(XtX + alpha * I)}
    return coef, predict, info


def ridge_svd(X, y, alpha, **kwargs):
    """Ridge using SVD"""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    d = s / (s**2 + alpha)
    coef = Vt.T @ (d * (U.T @ y))


    info = {'method': 'ridge_svd', 'alpha': alpha}
    return coef, predict, info


def ridge_sklearn(X, y, alpha):
    """Ridge using sklearn"""
    reg = SkRidge(alpha=alpha, fit_intercept=False).fit(X, y)
    coef = reg.coef_


    info = {'method': 'ridge_sklearn', 'alpha': alpha}
    return coef, predict, info
