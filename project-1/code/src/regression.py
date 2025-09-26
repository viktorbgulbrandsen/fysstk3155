import numpy as np

from sklearn.linear_model import LinearRegression, Ridge as SkRidge, Lasso as SkLasso

def predict(X, coef):
    """Simple prediction function"""
    return X @ coef


# OLS Implementations
def ols_normal(X, y):
    """OLS using normal equations"""
    XtX = X.T @ X
    Xty = X.T @ y
    coef = np.linalg.solve(XtX, Xty)

    def gradient(X_new):
        return X_new

    info = {'method': 'normal', 'condition_number': np.linalg.cond(XtX)}
    return coef, predict, gradient, info


def ols_pinv(X, y):
    """OLS using pseudoinverse"""
    coef = np.linalg.pinv(X) @ y

    def gradient(X_new):
        return X_new

    info = {'method': 'pinv', 'condition_number': np.linalg.cond(X)}
    return coef, predict, gradient, info


def ols_qr(X, y):
    """OLS using QR decomposition"""
    Q, R = np.linalg.qr(X)
    coef = np.linalg.solve(R, Q.T @ y)

    def gradient(X_new):
        return X_new

    info = {'method': 'qr', 'condition_number': np.linalg.cond(R)}
    return coef, predict, gradient, info


def ols_svd(X, y):
    """OLS using SVD decomposition"""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    coef = Vt.T @ (np.diag(1/s) @ (U.T @ y))

    def gradient(X_new):
        return X_new

    info = {'method': 'svd', 'condition_number': s[0]/s[-1]}
    return coef, predict, gradient, info


def ols_sklearn(X, y):
    """OLS using sklearn"""
    reg = LinearRegression(fit_intercept=False).fit(X, y)
    coef = reg.coef_

    def gradient(X_new):
        return X_new

    info = {'method': 'sklearn'}
    return coef, predict, gradient, info


# Ridge Implementations
def ridge_normal(X, y, alpha):
    """Ridge using normal equations"""
    XtX = X.T @ X
    I = np.eye(XtX.shape[0])
    Xty = X.T @ y
    coef = np.linalg.solve(XtX + alpha * I, Xty)

    def gradient(X_new):
        return X_new

    info = {'method': 'ridge_normal', 'alpha': alpha, 'condition_number': np.linalg.cond(XtX + alpha * I)}
    return coef, predict, gradient, info


def ridge_svd(X, y, alpha):
    """Ridge using SVD"""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    d = s / (s**2 + alpha)
    coef = Vt.T @ (d * (U.T @ y))

    def gradient(X_new):
        return X_new

    info = {'method': 'ridge_svd', 'alpha': alpha}
    return coef, predict, gradient, info


def ridge_sklearn(X, y, alpha):
    """Ridge using sklearn"""
    reg = SkRidge(alpha=alpha, fit_intercept=False).fit(X, y)
    coef = reg.coef_

    def gradient(X_new):
        return X_new

    info = {'method': 'ridge_sklearn', 'alpha': alpha}
    return coef, predict, gradient, info


