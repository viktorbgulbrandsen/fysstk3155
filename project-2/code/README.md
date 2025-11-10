````markdown
#### Group Members

Viktor Braaten Gulbrandsen  

---

#### Project Description

This project implements a small feed–forward neural network (FFNN) framework from scratch and applies it to

- regression on the Runge function on \([-1,1]\)  
- multiclass classification on the MNIST handwritten digits

I study how **optimization algorithms** (batch GD, SGD, Adam, RMSProp), **activation functions** (sigmoid, ReLU, Leaky ReLU) and **regularization** (\(L_1, L_2\)) affect training and generalization.  
For regression the goal is to approximate the Runge function using both raw and polynomially expanded inputs; for classification the same framework is extended.  
Results are qualitatively compared to `scikit-learn`’s `MLPRegressor`.

---

#### Installation

```bash
git clone https://github.com/viktorbgulbrandsen/fysstk3155
cd project-2
pip install -r requirements.txt
````

---

#### Project Structure

```text
project-2/
├── code/
│   ├── FFNN.py
│   ├── optimizers.py
│   ├── functions.py
│   └── notebooks/
│       └── Figures/
```

---

#### Notebooks (`code/notebooks/`)

* **Runge regression – basic FFNN**
  Single–hidden–layer FFNN on raw Runge data with batch GD; compares 50 vs 100 hidden nodes.

* **Runge regression – polynomial features and optimizers**
  Degree–10 Vandermonde features; compares batch GD, SGD, Adam and RMSProp on standardized inputs.

* **Activations and depth**
  Two hidden layers with Sigmoid–Sigmoid–ReLU vs Sigmoid–Sigmoid–Leaky ReLU

* **Regularization ((L_1), (L_2))**
  Adds (L_1)/(L_2) penalties in backprop for a 50-node hidden layer

* **Comparison with `scikit-learn`**
  Uses `MLPRegressor` on the polynomial Runge setup; compares SGD/Adam loss curves to the custom code.

* **MNIST classification**
  One hidden layer (64 sigmoid units) with Softmax output, trained with mini–batch SGD

All figures are saved to `Figures/` for use in the LaTeX report.

---

#### Source Code (`code/`)

* **`FFNN.py`**
  Core fully connected FFNN:

  * arbitrary layer sizes
  * explicit forward pass storing ((z, a))
  * backprop with user–supplied activations
  * MSE loss for regression (classification via Softmax + cross–entropy in notebooks)

* **`optimizers.py`**
  FFNN subclasses with custom `backward_pass`:
  * Adam 
  * RMSProp

* **`functions.py`**
  Helper functions:
  * Runge function 
  * activations and derivatives (sigmoid, ReLU, Leaky ReLU, identity, Softmax)
