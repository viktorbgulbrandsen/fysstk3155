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
    def __init__(self, lmbda=1.0):
        self.theta = None
        self.fitted = False
        self.lmbda = lmbda
    
    def _analytical_solve(self, X, y, lmbda):
        I = np.eye(X.shape[1])
        return np.linalg.inv(X.T @ X + lmbda * I) @ X.T @ y
        
    def _cholesky_solve(self, X, y, lmbda):
        I = np.eye(X.shape[1])
        A = X.T @ X + lmbda * I
        b = X.T @ y
        return np.linalg.solve(A, b)
        
    def fit(self, X, y, lmbda=None, method='analytical'):
        if lmbda is not None:
            self.lmbda = lmbda
            
        methods = {
            'analytical': lambda: self._analytical_solve(X, y, self.lmbda),
            'cholesky': lambda: self._cholesky_solve(X, y, self.lmbda)
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
        mse = np.mean((y - X @ theta)**2)
        penalty = self.lmbda * np.sum(theta**2)
        return mse + penalty
    
    def gradient(self, X, y, theta):
        return 2/len(y) * X.T @ (X @ theta - y) + 2 * self.lmbda * theta


class LassoRegression:
    def __init__(self, lmbda=1.0):
        self.lmbda = lmbda
        self.theta = None
        self.fitted = False
    
    def _soft_threshold(self, x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def fit(self, X, y, lmbda=None, max_iter=1000, tolerance=1e-6):
        if lmbda is not None:
            self.lmbda = lmbda
            
        n, p = X.shape
        self.theta = np.zeros(p)
        
        XTX_diag = np.sum(X**2, axis=0)
        
        for iteration in range(max_iter):
            theta_old = self.theta.copy()
            
            for j in range(p):
                residual = y - X @ self.theta + X[:, j] * self.theta[j]
                
                rho = X[:, j] @ residual
                if XTX_diag[j] != 0:
                    self.theta[j] = self._soft_threshold(rho, self.lmbda * n) / XTX_diag[j]
                else:
                    self.theta[j] = 0
            
            if np.linalg.norm(self.theta - theta_old) < tolerance:
                break
        
        self.fitted = True
        return self
    
    def cost_function(self, X, y, theta):
        mse = np.mean((y - X @ theta)**2)
        l1_penalty = self.lmbda * np.sum(np.abs(theta))
        return mse + l1_penalty
    
    def gradient(self, X, y, theta):
        n = len(y)
        mse_gradient = 2/n * X.T @ (X @ theta - y)
        l1_subgradient = self.lmbda * np.sign(theta)
        return mse_gradient + l1_subgradient
    
    def predict(self, X):
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        return X @ self.theta


