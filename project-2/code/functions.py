import numpy as np

def runge_function(x):
    return 1 / (1 + 25 * x**2)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def ReLU(z):
    return np.maximum(0, z)

def ReLU_derivative(z):
    return (z > 0).astype(float)

def identity(z):
    return z

def identity_derivative(z):
    return np.ones_like(z)

def leaky_ReLU(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_ReLU_derivative(z, alpha=0.01):
    dz = np.ones_like(z)
    dz[z < 0] = alpha
    return dz
