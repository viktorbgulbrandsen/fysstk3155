#scaling the data

#imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os

from uke35.ex3 import OLS_parameters, mean_squared_error


#dataset
n = 100
x = np.linspace(-3, 3, n)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1)
# -----------------------------
# Exercise 3a: adapt your function from last week to only include the intercept column if the boolean argument intercept is True
# -----------------------------

def adapted_OLS(x_data, y_data, intercept=True):
    if intercept:
        X = np.c_[np.ones((x_data.shape[0], 1)), x_data]
    else:
        if len(x_data.shape) == 1:
            X = x_data[:, np.newaxis]
        else:
            X = x_data

    # Call the inherited function
    return OLS_parameters(X, y_data)

def test_adapted_OLS():
    # Test with intercept
    beta_with_intercept = adapted_OLS(x, y, intercept=True)
    print(f"Parameters with intercept: {beta_with_intercept}")

    # Test without intercept
    beta_without_intercept = adapted_OLS(x, y, intercept=False)
    print(f"Parameters without intercept: {beta_without_intercept}")
    assert beta_with_intercept.shape[0] == 2, "With intercept, should have 2 parameters"
    assert beta_without_intercept.shape[0] == 1, "Without intercept, should have 1 parameter"   
    X_with_intercept = np.c_[np.ones((x.shape[0], 1)), x]
    y_pred_with_intercept = X_with_intercept @ beta_with_intercept
    plt.scatter(x, y, label='Data')
    plt.plot(x, y_pred_with_intercept, color='red', label='OLS with intercept')
    plt.legend()
    plt.show() 



# test_adapted_OLS()

# -----------------------------
# Exercise 3b: split your data
# -----------------------------

from uke35.ex4 import polynomial_features 

x_raw = x.reshape(-1,1)
X = polynomial_features(x, p=3)
X_train, X_test, y_train, y_test, x_train, x_test = train_test_split(
    X, y, x_raw, test_size=0.2, random_state=314
)


# -----------------------------
# Exercise 3c: scale your design matrix with sklearn standardscaler
# -----------------------------

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_offset = np.mean(y_train)

# -----------------------------
# Exercise 4a - implementing ridge regression
# -----------------------------


def ridge_parameters(X, y, lam):
    #assumes X is scaled and has no incercept column
    n, p = X.shape
    I = np.eye(p)
    return np.linalg.inv(X.T @ X + lam * I) @ (X.T @ y)

# ------------------------------
# Exercise 4b - fit the model to the data, and plot the prediction using both training and test x-values extracted before scaling, and the y-offset
# ------------------------------
def plot_ridge_regression(lam):
    beta_ridge = ridge_parameters(X_train_scaled, y_train - y_offset, lam)
    y_pred_train = X_train_scaled @ beta_ridge + y_offset
    y_pred_test = X_test_scaled @ beta_ridge + y_offset

    plt.scatter(x_train, y_train, label='Train Data', color='blue')
    plt.scatter(x_test, y_test, label='Test Data', color='green')
    plt.plot(np.sort(x_train, axis=0),
             y_pred_train[np.argsort(x_train[:,0])], color='cyan', label='Train Prediction')
    plt.plot(np.sort(x_test, axis=0),
             y_pred_test[np.argsort(x_test[:,0])], color='orange', label='Test Prediction')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Ridge Regression (lambda={lam})')
    plt.legend()
    plt.show()

# plot_ridge_regression(lam=1.0)

# ------------------------------
# Exercise 5 - experiment with different values of lambda
# ------------------------------

# -----------------------------
# Exercise 5a
# -----------------------------

def mse_vs_degree_ridge(x, y, degrees, lam, plot=True):
    mse_test_list = []
    for d in degrees:
        X = polynomial_features(x, p=d)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=314)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        y_offset = np.mean(y_train)

        beta_ridge = ridge_parameters(X_train_scaled, y_train - y_offset, lam)
        y_pred_test = X_test_scaled @ beta_ridge + y_offset

        mse_test_list.append(mean_squared_error(y_test, y_pred_test))

    if plot:
        plt.figure()
        plt.plot(degrees, mse_test_list, 'o-')
        plt.xlabel("Polynomial Degree")
        plt.ylabel("Test MSE")
        plt.title(f"Ridge Regression MSE vs. Polynomial Degree (lambda={lam})")
        plt.show()

    return degrees, mse_test_list

# -----------------------------
# Exercise 5b
# -----------------------------

def mse_vs_lambda_ridge(x, y, degree, lambdas, plot=True):
    X = polynomial_features(x, p=degree)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=314)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_offset = np.mean(y_train)

    mse_test_list = []
    for lam in lambdas:
        beta_ridge = ridge_parameters(X_train_scaled, y_train - y_offset, lam)
        y_pred_test = X_test_scaled @ beta_ridge + y_offset

        mse_test_list.append(mean_squared_error(y_test, y_pred_test))

    if plot:
        plt.figure()
        plt.plot(np.log10(lambdas), mse_test_list, 'o-')
        plt.xlabel("log10(lambda)")
        plt.ylabel("Test MSE")
        plt.title(f"Ridge Regression MSE vs. Lambda (degree={degree})")
        plt.show()

    return lambdas, mse_test_list

# -----------------------------
# Exercise 5c
# -----------------------------

def mse_heatmap_ridge(x, y, degrees, lambdas):
    mse_heatmap = np.zeros((len(degrees), len(lambdas)))

    for i, d in enumerate(degrees):
        X = polynomial_features(x, p=d)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=314)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        y_offset = np.mean(y_train)

        for j, lam in enumerate(lambdas):
            beta_ridge = ridge_parameters(X_train_scaled, y_train - y_offset, lam)
            y_pred_test = X_test_scaled @ beta_ridge + y_offset

            mse_heatmap[i, j] = mean_squared_error(y_test, y_pred_test)
    
    #this ax design was generated with help of chatgpt
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mse_heatmap, cmap="viridis")

    ax.set_xticks(np.arange(len(lambdas)))
    ax.set_yticks(np.arange(len(degrees)))
    ax.set_xticklabels([f"{l:.1e}" for l in lambdas])
    ax.set_yticklabels(degrees)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    for i in range(len(degrees)):
        for j in range(len(lambdas)):
            text = ax.text(j, i, f"{mse_heatmap[i, j]:.2f}",
                           ha="center", va="center", color="w")

    ax.set_title("Ridge Regression Test MSE")
    fig.tight_layout()
    plt.xlabel("Lambda")
    plt.ylabel("Polynomial Degree")
    plt.show()

    return mse_heatmap


# --- testing ---
print("--- running ex 5a ---")
degrees_5a = range(1, 6)
mse_vs_degree_ridge(x, y, degrees_5a, lam=0.1)

print("--- running ex 5b ---")
lambdas_5b = np.logspace(-5, -1, 100)
mse_vs_lambda_ridge(x, y, degree=4, lambdas=lambdas_5b)

print("--- running ex  5c ---")
degrees_5c = range(1, 6)
lambdas_5c = np.logspace(-5, -1, 5)
mse_heatmap_ridge(x, y, degrees_5c, lambdas_5c)