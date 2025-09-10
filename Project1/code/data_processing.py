import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


class RungeData:
    def __init__(self, n_points, random_seed=314, x_min=-1, x_max=1, random_x=True):
        np.random.seed(random_seed)
        
        if random_x:
            self.x = np.random.uniform(x_min, x_max, n_points)
        else:
            self.x = np.linspace(x_min, x_max, n_points)
        
        self.y = 1 / (1 + 25 * self.x**2)
    
    def add_noise(self, noise_std=1.0):
        noise = np.random.normal(0, noise_std, len(self.x))
        self.y += noise
        return self
    
    def scale(self, method = "StandardScaler"):
        if method == "StandardScaler":
            scaler = StandardScaler()
            self.x = scaler.fit_transform(self.x.reshape(-1, 1)).flatten()
            self.y = scaler.fit_transform(self.y.reshape(-1, 1)).flatten()
        else:
            assert ValueError("other methods not implemented")
        return self
    
    def center(self):
        self.x = self.x - np.mean(self.x)
        self.y = self.y - np.mean(self.y)
        return self
    
    def split(self, test_size=0.2):
        return train_test_split(self.x, self.y, test_size=test_size, random_state=314)


def create_polynomial_features(x, degree, include_bias=True):
    x = x.reshape(-1, 1)
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    return poly.fit_transform(x)