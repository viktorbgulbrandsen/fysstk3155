import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# -----------------------------
# Exercise 4: Polynomial regression experiment
# -----------------------------

n = 100
x = np.linspace(-3, 3, n)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1, size=n)

def polynomial_features(x, p):
    n = len(x)
    X = np.zeros((n, p+1))
    for j in range(p+1):
        X[:, j] = x**j
    return X

degrees = range(2, 11)
mse_train_list = []
mse_test_list = []

from ex3 import OLS_parameters, mean_squared_error

for d in degrees:
    X_poly = polynomial_features(x, d)
    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size=0.2, random_state=314
    )
    beta = OLS_parameters(X_train, y_train)
    y_pred_train = X_train @ beta
    y_pred_test = X_test @ beta
    mse_train_list.append(mean_squared_error(y_train, y_pred_train))
    mse_test_list.append(mean_squared_error(y_test, y_pred_test))

best_degree, best_mse = min(zip(degrees, mse_test_list), key=lambda t: t[1])
print("lowest test MSE:", best_mse, "at degree", best_degree)

plt.plot(degrees, mse_train_list, label="train MSE")
plt.plot(degrees, mse_test_list, label="test MSE")
plt.xlabel("polynomial degree")
plt.ylabel("MSE")
plt.legend()
plt.show()

# -----------------------------
# Exercise 5: Checks against sklearn
# -----------------------------

for d in range(1, 11):
    X_poly_ours = polynomial_features(x, d)
    X_poly_skl = PolynomialFeatures(degree=d, include_bias=True).fit_transform(x.reshape(-1, 1))
    assert np.allclose(X_poly_ours, X_poly_skl, rtol=1e-8, atol=1e-10), f"feature matrix mismatch at degree {d}"
print("feature matrices match")

for d in range(1, 11):
    X_poly = polynomial_features(x, d)
    beta_ours = OLS_parameters(X_poly, y)
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X_poly, y)
    beta_skl = lr.coef_
    assert np.allclose(beta_ours, beta_skl, rtol=1e-8, atol=1e-10), f"beta mismatch at degree {d}"
print("ols parameters match")
