import numpy as np


class OLSRegression:
    def __init__(self):
        self.theta = None
        self.fitted = False
    
    def _svd_solve(self, X, y):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        return Vt.T @ np.diag(1/s) @ U.T @ y
        
    def fit(self, X, y, method='pinv'):
        methods = {
            'pinv': lambda: np.linalg.pinv(X) @ y,
            'normal': lambda: np.linalg.inv(X.T @ X) @ X.T @ y,
            'svd': lambda: self._svd_solve(X, y),
            'lstsq': lambda: np.linalg.lstsq(X, y, rcond=None)[0]
        }
        
        if method not in methods:
            raise ValueError(f"Method must be one of {list(methods.keys())}")
            
        self.theta = methods[method]()
        self.fitted = True
        return self
    
    def predict(self, X):
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        return X @ self.theta
    
    def cost_function(self, X, y, theta):
        return np.mean((y - X @ theta)**2)
    
    def gradient(self, X, y, theta):
        return 2/len(y) * X.T @ (X @ theta - y)


class RidgeRegression:
    def __init__(self):
        self.theta = None
        self.fitted = False
    
    def _analytical_solve(self, X, y, lmbda):
        I = np.eye(X.shape[1])
        return np.linalg.inv(X.T @ X + lmbda * I) @ X.T @ y
        
    def _cholesky_solve(self, X, y, lmbda):
        I = np.eye(X.shape[1])
        A = X.T @ X + lmbda * I
        b = X.T @ y
        return np.linalg.solve(A, b)
        
    def fit(self, X, y, lmbda=1.0, method='analytical'):
        methods = {
            'analytical': lambda: self._analytical_solve(X, y, lmbda),
            'cholesky': lambda: self._cholesky_solve(X, y, lmbda)
        }
        
        if method not in methods:
            raise ValueError(f"Method must be one of {list(methods.keys())}")
            
        self.theta = methods[method]()
        self.fitted = True
        return self
    
    def predict(self, X):
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        return X @ self.theta
    
    def cost_function(self, X, y, theta, lmbda):
        mse = np.mean((y - X @ theta)**2)
        penalty = lmbda * np.sum(theta**2)
        return mse + penalty
    
    def gradient(self, X, y, theta, lmbda):
        return 2/len(y) * X.T @ (X @ theta - y) + 2 * lmbda * theta


class LassoRegression:
    def __init__(self, lmbda=1.0):
        self.lmbda = lmbda
        self.theta = None
    
    def cost_function(self, X, y, theta):
        mse = np.mean((y - X @ theta)**2)
        l1_penalty = self.lmbda * np.sum(np.abs(theta))
        return mse + l1_penalty
    
    def gradient(self, X, y, theta):
        n = len(y)
        mse_gradient = 2/n * X.T @ (X @ theta - y)
        l1_gradient = self.lmbda * np.sign(theta)
        return mse_gradient + l1_gradient
    
    def predict(self, X):
        if self.theta is None:
            raise ValueError("Model must be optimized before making predictions")
        return X @ self.theta


