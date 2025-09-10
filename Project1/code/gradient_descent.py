import numpy as np
import autograd.numpy as anp
from autograd import grad
import jax.numpy as jnp
import jax
from jax import grad as jax_grad


class GradientDescent:
    def __init__(self):
        pass
    
    def optimize(self, model, X, y, method='gd', learning_rate=0.01, max_iter=1000, tolerance=1e-6, epochs=None, auto_diff='numpy', lr_schedule=None, **kwargs):
        methods = {
            'gd': self._gradient_descent,
            'sgd': self._stochastic_gd,
            'momentum': self._momentum_gd,
            'adagrad': self._adagrad,
            'rmsprop': self._rmsprop,
            'adam': self._adam
        }
        
        if method not in methods:
            raise ValueError(f"Method must be one of {list(methods.keys())}")
        
            
        iterations = epochs if epochs else max_iter
        theta = methods[method](model, X, y, learning_rate, iterations, tolerance, auto_diff, lr_schedule, **kwargs)
        model.theta = theta
        return theta
    
    def _compute_gradient(self, model, X, y, theta, auto_diff):
        if auto_diff == 'autograd':
            cost_fn = lambda params: model.cost_function(X, y, params)
            return grad(cost_fn)(theta)
        elif auto_diff == 'jax':
            cost_fn = lambda params: model.cost_function(X, y, params)
            return jax_grad(cost_fn)(theta)
        else:
            return model.gradient(X, y, theta)
    
    def _apply_learning_rate_schedule(self, learning_rate, lr_schedule, epoch):
        if lr_schedule is None:
            return learning_rate
        elif lr_schedule == 'exponential':
            decay_rate = 0.95
            return learning_rate * (decay_rate ** epoch)
        elif lr_schedule == 'step':
            step_size = 100
            decay_factor = 0.5
            return learning_rate * (decay_factor ** (epoch // step_size))
        elif callable(lr_schedule):
            return lr_schedule(learning_rate, epoch)
        else:
            return learning_rate
    
    def _gradient_descent(self, model, X, y, learning_rate, max_iter, tolerance, auto_diff, lr_schedule, **kwargs):
        n, p = X.shape
        theta = np.random.randn(p) * 0.01
        
        for iteration in range(max_iter):
            current_lr = self._apply_learning_rate_schedule(learning_rate, lr_schedule, iteration)
            gradient = self._compute_gradient(model, X, y, theta, auto_diff)
            theta_new = theta - current_lr * gradient
            
            if np.linalg.norm(theta_new - theta) < tolerance:
                break
                
            theta = theta_new
        
        return theta
    
    def _stochastic_gd(self, model, X, y, learning_rate, max_iter, tolerance, auto_diff, lr_schedule, batch_size=32, shuffle=True, **kwargs):
        n, p = X.shape
        theta = np.random.randn(p) * 0.01
        
        for epoch in range(max_iter):
            if shuffle:
                indices = np.random.permutation(n)
                X_shuffled, y_shuffled = X[indices], y[indices]
            else:
                X_shuffled, y_shuffled = X, y
            
            theta_old = theta.copy()
            current_lr = self._apply_learning_rate_schedule(learning_rate, lr_schedule, epoch)
            
            for i in range(0, n, batch_size):
                end_idx = min(i + batch_size, n)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]
                
                gradient = self._compute_gradient(model, X_batch, y_batch, theta, auto_diff)
                theta = theta - current_lr * gradient
            
            if np.linalg.norm(theta - theta_old) < tolerance:
                break
        
        return theta
    
    def _momentum_gd(self, model, X, y, learning_rate, max_iter, tolerance, auto_diff, lr_schedule, gamma=0.9, **kwargs):
        n, p = X.shape
        theta = np.random.randn(p) * 0.01
        velocity = np.zeros(p)
        
        for epoch in range(max_iter):
            current_lr = self._apply_learning_rate_schedule(learning_rate, lr_schedule, epoch)
            gradient = self._compute_gradient(model, X, y, theta, auto_diff)
            velocity = gamma * velocity + current_lr * gradient
            theta_new = theta - velocity
            
            if np.linalg.norm(theta_new - theta) < tolerance:
                break
                
            theta = theta_new
        
        return theta
    
    def _adagrad(self, model, X, y, learning_rate, max_iter, tolerance, auto_diff, lr_schedule, epsilon=1e-8, **kwargs):
        n, p = X.shape
        theta = np.random.randn(p) * 0.01
        G = np.zeros(p)
        
        for epoch in range(max_iter):
            current_lr = self._apply_learning_rate_schedule(learning_rate, lr_schedule, epoch)
            gradient = self._compute_gradient(model, X, y, theta, auto_diff)
            G += gradient**2
            theta_new = theta - current_lr * gradient / (np.sqrt(G) + epsilon)
            
            if np.linalg.norm(theta_new - theta) < tolerance:
                break
                
            theta = theta_new
        
        return theta
    
    def _rmsprop(self, model, X, y, learning_rate, max_iter, tolerance, auto_diff, lr_schedule, beta=0.9, epsilon=1e-8, **kwargs):
        n, p = X.shape
        theta = np.random.randn(p) * 0.01
        v = np.zeros(p)
        
        for epoch in range(max_iter):
            current_lr = self._apply_learning_rate_schedule(learning_rate, lr_schedule, epoch)
            gradient = self._compute_gradient(model, X, y, theta, auto_diff)
            v = beta * v + (1 - beta) * gradient**2
            theta_new = theta - current_lr * gradient / (np.sqrt(v) + epsilon)
            
            if np.linalg.norm(theta_new - theta) < tolerance:
                break
                
            theta = theta_new
        
        return theta
    
    def _adam(self, model, X, y, learning_rate, max_iter, tolerance, auto_diff, lr_schedule, beta1=0.9, beta2=0.999, epsilon=1e-8, **kwargs):
        n, p = X.shape
        theta = np.random.randn(p) * 0.01
        m = np.zeros(p)
        v = np.zeros(p)
        
        for t in range(1, max_iter + 1):
            current_lr = self._apply_learning_rate_schedule(learning_rate, lr_schedule, t-1)
            gradient = self._compute_gradient(model, X, y, theta, auto_diff)
            
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * gradient**2
            
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            
            theta_new = theta - current_lr * m_hat / (np.sqrt(v_hat) + epsilon)
            
            if np.linalg.norm(theta_new - theta) < tolerance:
                break
                
            theta = theta_new
        
        return theta