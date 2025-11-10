import numpy as np
class FFNN:
    def __init__(self, input_size, layer_output_sizes):
        self.forward_pass_values = []
        self.output = None
        self.input_size = input_size
        self.layer_output_sizes = layer_output_sizes
        layers = []

        i_size = input_size
        for layer_output_size in layer_output_sizes:
            W = np.random.randn(i_size, layer_output_size) 
            b = np.random.randn(layer_output_size)
            layers.append((W, b))
            i_size = layer_output_size
        self.layers = layers

    def forward_pass(self, input, activation_funcs):
        if not len(activation_funcs) == len(self.layer_output_sizes):
            raise IndexError("Should be same size")
        
        self.forward_pass_values = []
        self.forward_pass_values.append({'a': input})
        
        a = input
        for (W, b), activation_func in zip(self.layers, activation_funcs):
            z = a @ W + b
            a = activation_func(z)
            self.forward_pass_values.append({'z': z, 'a': a})
        self.output = a
        return a
        
    def backward_pass(self, y_ground, dactivation_funcs, learning_rate=0.01):
        a_final = self.forward_pass_values[-1]['a']
        dLda = 2 * (a_final - y_ground) / y_ground.shape[0]
        
        for i, ((W, b), dactivation_func) in enumerate(reversed(list(zip(self.layers, dactivation_funcs)))):
            layer_idx = len(self.layers) - 1 - i
            
            z = self.forward_pass_values[layer_idx + 1]['z']
            a_prev = self.forward_pass_values[layer_idx]['a']
            
            dLdz = dLda * dactivation_func(z)
            dLdW = a_prev.T @ dLdz
            dLdb = np.sum(dLdz, axis=0)
            
            self.update_weights(layer_idx, W, b, dLdW, dLdb, learning_rate)
            
            dLda = dLdz @ W.T

    def update_weights(self, layer_idx, W, b, dLdW, dLdb, learning_rate): # plain gradient descent
        self.layers[layer_idx] = (W - learning_rate * dLdW, b - learning_rate * dLdb)


