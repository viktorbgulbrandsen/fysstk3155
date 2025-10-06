#### Group Members:

Viktor Braaten Gulbrandsen

#### Project Description

In this project, we look at regression for the oscilliatory behavior of the Runge function at $[-1,1]$. OLS, Ridge, Lasso and Gradient Descent are applied, and the nuance of their behavior are explored through the lens of statistical learning, numerical approximation and linear algebra.

#### Installation

In order to install the project, simply run:

```bash
git clone https://github.com/viktorbgulbrandsen/fysstk3155
cd project-1
pip install -r requirements.txt #we suggest making a venv first
```


#### Notebooks: 
**`conditioning.ipynb`** - Explores Vandermonde matrix conditioning and spectral degradation. Visualizes how condition numbers and singular values collapse as polynomial degree increases, plus regularization effects.Retry
**`Least_Square_validation.ipynb`** - Compares OLS solvers (Normal Equations, QR, SVD) and gradient descent variants (Batch GD, SGD, Momentum, Adagrad, RMSProp, Adam) using training loss, holdout validation, k-fold CV, and bootstrap. Includes spectral tail energy analysis,
**`OLS_bias_variance_plot.ipynb`** – Analyzes bias–variance tradeoff in polynomial regression on the Runge function over ([-1,1]). Fits OLS models of varying degrees and visualizes bias², variance, and total error, showing how increasing complexity leads to overfitting.
**`Ridge_bias_variance_plot.ipynb`** – Bias–variance decomposition for Ridge regression across polynomial degrees.
**`Ridge_Validation.ipynb`** – Validation and model selection for Ridge regression over degrees and lambda values.
**`LASSO_Validation.ipynb`** – Validation and model selection for LASSO regression across polynomial degrees and alpha values.

#### src: 
**`basis.py`** – Generates polynomial design matrices (Vandermonde basis) up to a specified degree.
**`bias_variance.py`** – Performs Monte Carlo bias–variance decomposition via bootstrap resampling.
**`data.py`** – Provides the Runge function and utilities for generating equispaced sample points.
**`gd.py`** – Implements gradient-based solvers: batch, stochastic, momentum, adaptive (Adam, RMSProp, Adagrad) and LASSO-specific proximal variants.
**`regression.py`** – Contains closed-form and decomposition-based OLS and Ridge regressors (Normal, QR, SVD, sklearn).
**`validation.py`** – Includes validation and resampling methods: bootstrap, k-fold CV, residual bootstrap, and deterministic holdout.