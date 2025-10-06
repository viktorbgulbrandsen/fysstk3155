import numpy as np
import torch
from sklearn.linear_model import SGDRegressor
from .regression import predict

# -------------------------
# Gradient helpers
# -------------------------
def ls_gradient(X, y):
    return lambda theta: X.T @ (X @ theta - y)

def ridge_gradient(X, y, alpha):
    return lambda theta: X.T @ (X @ theta - y) + alpha * theta

def lasso_gradient(X, y, alpha):
    return lambda theta: X.T @ (X @ theta - y) + alpha * np.sign(theta)


# -------------------------
# Batch gradient descent
# -------------------------
def batch_gradient_descent(X, y, eta=1e-3, max_iter=1000, tol=1e-8, grad_fn=None, verbose=False, **kwargs):
    """
    Batch gradient descent in unified format.
    Pass grad_fn for custom objectives (Ridge, Lasso, etc.)
    Returns (coef, predict_fn, info)
    """
    p = X.shape[1]
    theta = np.zeros(p)
    history, thetas = [], []

    if grad_fn is None:
        grad_fn = ls_gradient(X, y)

    for t in range(max_iter):
        grad = grad_fn(theta)
        theta_new = theta - eta * grad

        history.append(np.linalg.norm(grad))
        thetas.append(theta_new.copy())

        if np.linalg.norm(theta_new - theta) < tol:
            theta = theta_new
            break

        theta = theta_new

    info = {
        "method": "gd",
        "eta": eta,
        "max_iter": max_iter,
        "n_iter": t + 1,
        "grad_norm_history": history,
        "trajectory": np.array(thetas),
    }

    return theta, predict, info


# -------------------------
# Stochastic gradient descent (mini-batch or single-sample)
# -------------------------
def sgd_sklearn(X, y, eta=1e-3, max_iter=2000, batch_size=None, alpha=0.0):
    """SKLearn SGD - supports OLS (alpha=0) and Ridge (alpha>0)"""
    reg = SGDRegressor(
        penalty='l2' if alpha > 0 else None,
        alpha=alpha if alpha > 0 else 0.0,
        learning_rate="constant",
        eta0=eta,
        max_iter=max_iter,
        tol=1e-6,
        fit_intercept=False,
        shuffle=True,
        average=False,
        random_state=314,
    )
    reg.fit(X, y)
    coef = reg.coef_
    info = {
        "method": "sklearn_SGD",
        "alpha": alpha,
        "eta": eta,
        "max_iter": max_iter,
        "n_iter_": reg.n_iter_,
    }
    return coef, predict, info

# -------------------------
# Momentum gradient descent
# -------------------------
def momentum_gradient_descent(X, y, eta=1e-3, beta=0.9, max_iter=1000, tol=1e-8, grad_fn=None, verbose=False, **kwargs):
    """
    Gradient descent with classical momentum.
    Pass grad_fn for custom objectives.
    Returns (coef, predict_fn, info) in unified format.
    """
    p = X.shape[1]
    theta = np.zeros(p)
    v = np.zeros(p)
    
    if grad_fn is None:
        grad_fn = ls_gradient(X, y)

    history, thetas = [], []

    for t in range(max_iter):
        grad = grad_fn(theta)
        v = beta * v + (1 - beta) * grad
        theta_new = theta - eta * v

        history.append(np.linalg.norm(grad))
        thetas.append(theta_new.copy())

        if np.linalg.norm(theta_new - theta) < tol:
            theta = theta_new
            break

        theta = theta_new

    info = {
        "method": "momentum_gd",
        "eta": eta,
        "beta": beta,
        "max_iter": max_iter,
        "n_iter": t + 1,
        "grad_norm_history": history,
        "trajectory": np.array(thetas),
    }

    return theta, predict, info


# -------------------------
# Adaptive Gradient Methods (via PyTorch)
# -------------------------
def torch_optimizer_solver(
    X, y,
    optimizer_cls,
    lr=1e-3,
    max_iter=2000,
    loss_fn=None,
    weight_decay=0.0,
    tol=1e-8,
    verbose=False,
    **kwargs
):
    """
    Unified wrapper for PyTorch optimizers (Adam, AdamW, RMSProp, Adagrad).
    Pass custom loss_fn(X_t, y_t, theta) for Ridge/Lasso/etc.
    Returns (coef, predict_fn, info).
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    p = X.shape[1]
    theta = torch.zeros((p, 1), dtype=torch.float32, requires_grad=True)
    optimizer = optimizer_cls(
        [{"params": [theta], "lr": lr, "weight_decay": weight_decay}]
    )

    # Default to MSE
    if loss_fn is None:
        mse_fn = torch.nn.MSELoss()
        loss_fn = lambda X, y, th: mse_fn(X @ th, y)

    history = []

    for t in range(max_iter):
        optimizer.zero_grad()
        loss = loss_fn(X_t, y_t, theta)
        loss.backward()
        optimizer.step()

        history.append(loss.item())
        if loss.item() < tol:
            break

    coef = theta.detach().numpy().ravel()
    info = {
        "method": optimizer_cls.__name__,
        "lr": lr,
        "weight_decay": weight_decay,
        "max_iter": max_iter,
        "n_iter": t + 1,
        "loss_history": history,
    }

    return coef, predict, info


# --- Specific wrappers for clarity ---
def adam_solver(X, y, lr=1e-3, max_iter=2000, loss_fn=None, **kwargs):
    return torch_optimizer_solver(X, y, torch.optim.Adam, lr=lr, max_iter=max_iter, loss_fn=loss_fn, **kwargs)

def adamw_solver(X, y, lr=1e-3, max_iter=2000, weight_decay=1e-2, loss_fn=None, **kwargs):
    return torch_optimizer_solver(X, y, torch.optim.AdamW, lr=lr, max_iter=max_iter, weight_decay=weight_decay, loss_fn=loss_fn, **kwargs)

def rmsprop_solver(X, y, lr=1e-3, max_iter=2000, loss_fn=None, **kwargs):
    return torch_optimizer_solver(X, y, torch.optim.RMSprop, lr=lr, max_iter=max_iter, loss_fn=loss_fn, **kwargs)

def adagrad_solver(X, y, lr=1e-2, max_iter=2000, loss_fn=None, **kwargs):
    return torch_optimizer_solver(X, y, torch.optim.Adagrad, lr=lr, max_iter=max_iter, loss_fn=loss_fn, **kwargs)


# Add soft-thresholding helper
def soft_threshold(x, threshold):
    """Soft-thresholding operator for LASSO proximal step"""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

# Batch GD for LASSO
def batch_gradient_descent_lasso(X, y, alpha, eta=1e-3, max_iter=1000, tol=1e-8, **kwargs):
    """Proximal gradient descent for LASSO"""
    p = X.shape[1]
    theta = np.zeros(p)
    history = []
    
    for t in range(max_iter):
        grad = X.T @ (X @ theta - y)
        theta_new = theta - eta * grad
        theta_new = soft_threshold(theta_new, eta * alpha)  # Proximal step
        
        history.append(np.linalg.norm(grad))
        if np.linalg.norm(theta_new - theta) < tol:
            theta = theta_new
            break
        theta = theta_new
    
    info = {"method": "lasso_batch_gd", "alpha": alpha, "eta": eta, 
            "n_iter": t + 1, "grad_norm_history": history}
    return theta, predict, info

# Momentum for LASSO
def momentum_gradient_descent_lasso(X, y, alpha, eta=1e-3, beta=0.9, max_iter=1000, tol=1e-8, **kwargs):
    """Momentum with proximal step for LASSO"""
    p = X.shape[1]
    theta = np.zeros(p)
    v = np.zeros(p)
    history = []
    
    for t in range(max_iter):
        grad = X.T @ (X @ theta - y)
        v = beta * v + (1 - beta) * grad
        theta_new = theta - eta * v
        theta_new = soft_threshold(theta_new, eta * alpha)
        
        history.append(np.linalg.norm(grad))
        if np.linalg.norm(theta_new - theta) < tol:
            theta = theta_new
            break
        theta = theta_new
    
    info = {"method": "lasso_momentum", "alpha": alpha, "eta": eta, "beta": beta,
            "n_iter": t + 1, "grad_norm_history": history}
    return theta, predict, info


# PyTorch optimizers for LASSO
def torch_optimizer_solver_lasso(X, y, alpha, optimizer_cls, lr=1e-3, max_iter=2000, tol=1e-8, **kwargs):
    """PyTorch optimizers with proximal step for LASSO"""
    import torch
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    p = X.shape[1]
    theta = torch.zeros((p, 1), dtype=torch.float32, requires_grad=True)
    optimizer = optimizer_cls([theta], lr=lr)
    
    mse_fn = torch.nn.MSELoss()
    history = []
    
    for t in range(max_iter):
        optimizer.zero_grad()
        loss = mse_fn(X_t @ theta, y_t)
        loss.backward()
        optimizer.step()
        
        # Proximal step
        with torch.no_grad():
            theta.data = torch.sign(theta.data) * torch.maximum(
                torch.abs(theta.data) - lr * alpha, torch.tensor(0.0)
            )
        
        history.append(loss.item())
        if loss.item() < tol:
            break
    
    coef = theta.detach().numpy().ravel()
    info = {"method": f"lasso_{optimizer_cls.__name__}", "alpha": alpha, 
            "lr": lr, "n_iter": t + 1, "loss_history": history}
    return coef, predict, info

# Wrappers
def adam_solver_lasso(X, y, alpha, lr=1e-3, max_iter=2000, **kwargs):
    return torch_optimizer_solver_lasso(X, y, alpha, torch.optim.Adam, lr=lr, max_iter=max_iter, **kwargs)

def rmsprop_solver_lasso(X, y, alpha, lr=1e-3, max_iter=2000, **kwargs):
    return torch_optimizer_solver_lasso(X, y, alpha, torch.optim.RMSprop, lr=lr, max_iter=max_iter, **kwargs)

def adagrad_solver_lasso(X, y, alpha, lr=1e-2, max_iter=2000, **kwargs):
    return torch_optimizer_solver_lasso(X, y, alpha, torch.optim.Adagrad, lr=lr, max_iter=max_iter, **kwargs)
