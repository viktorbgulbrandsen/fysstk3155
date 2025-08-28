
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# -----------------------------
# Exercise 3: Income, children, spending regression
# -----------------------------


n = 20
income = np.array([116., 161., 167., 118., 172., 163., 179., 173., 162., 116., 101.,
                   176., 178., 172., 143., 135., 160., 101., 149., 125.])
children = np.array([5, 3, 0, 4, 5, 3, 0, 4, 4, 3, 3, 5, 1, 0, 2, 3, 2, 1, 5, 4])
spending = np.array([152., 141., 102., 136., 161., 129.,  99., 159., 160., 107.,  98.,
                     164., 121.,  93., 112., 127., 117.,  69., 156., 131.])

# feature matrix
X = np.zeros((n, 3))
X[:, 0] = 1.0          # intercept
X[:, 1] = income
X[:, 2] = children

def OLS_parameters(X, y):
    return np.linalg.inv(X.T @ X) @ (X.T @ y)

beta = OLS_parameters(X, spending)
print("intercept, income coefficient, children coefficient:")
print(beta)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, spending, test_size=0.2, random_state=314
)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

beta_train = OLS_parameters(X_train, y_train)
y_pred_train = X_train @ beta_train
y_pred_test = X_test @ beta_train

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

print("MSE train:", mse_train)
print("MSE test:", mse_test)
