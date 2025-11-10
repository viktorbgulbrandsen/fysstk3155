from FFNN import FFNN
import numpy as np

class FFNN_Adam(FFNN):
    def __init__(self, input_size, layer_output_sizes, beta1=0.9, beta2=0.999):
        super().__init__(input_size, layer_output_sizes)
        self.m = [(np.zeros_like(W), np.zeros_like(b)) for W, b in self.layers]
        self.v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in self.layers]
        self.t = 0
        self.beta1 = beta1
        self.beta2 = beta2

    def update_weights(self, layer_idx, W, b, dLdW, dLdb, learning_rate):
        if layer_idx == len(self.layers) - 1:
            self.t += 1
        
        m_W, m_b = self.m[layer_idx]
        v_W, v_b = self.v[layer_idx]
        
        m_W = self.beta1 * m_W + (1 - self.beta1) * dLdW
        m_b = self.beta1 * m_b + (1 - self.beta1) * dLdb
        
        v_W = self.beta2 * v_W + (1 - self.beta2) * dLdW**2
        v_b = self.beta2 * v_b + (1 - self.beta2) * dLdb**2
        
        m_W_hat = m_W / (1 - self.beta1**self.t)
        m_b_hat = m_b / (1 - self.beta1**self.t)
        
        v_W_hat = v_W / (1 - self.beta2**self.t)
        v_b_hat = v_b / (1 - self.beta2**self.t)
        
        W_update = learning_rate * m_W_hat / (np.sqrt(v_W_hat) + 1e-8)
        b_update = learning_rate * m_b_hat / (np.sqrt(v_b_hat) + 1e-8)
        
        self.m[layer_idx] = (m_W, m_b)
        self.v[layer_idx] = (v_W, v_b)
        
        self.layers[layer_idx] = (W - W_update, b - b_update)


class FFNN_RMSProp(FFNN):
    def __init__(self, input_size, layer_output_sizes, beta=0.9):
        super().__init__(input_size, layer_output_sizes)
        self.cache = [(np.zeros_like(W), np.zeros_like(b)) for W, b in self.layers]
        self.beta = beta

    def update_weights(self, layer_idx, W, b, dLdW, dLdb, learning_rate):
        cache_W, cache_b = self.cache[layer_idx]
        
        cache_W = self.beta * cache_W + (1 - self.beta) * dLdW**2
        cache_b = self.beta * cache_b + (1 - self.beta) * dLdb**2
        
        W_update = learning_rate * dLdW / (np.sqrt(cache_W) + 1e-8)
        b_update = learning_rate * dLdb / (np.sqrt(cache_b) + 1e-8)
        
        self.cache[layer_idx] = (cache_W, cache_b)
        self.layers[layer_idx] = (W - W_update, b - b_update)
