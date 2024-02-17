import numpy as np

class Layer_Dense:

    def __init__(self , n_inputs , n_neurons) :
        sigma_w = 0.01

        #initialize the weights and biases
        self.weights = sigma_w*np.random.randn(n_inputs , n_neurons)
        self.biases = np.zeros((1,n_neurons))

        #create momentum arrays for sgd optimizer with momentum or for adam optimizer
        self.weights_momentum = np.zeros_like(self.weights)
        self.biases_momentum = np.zeros_like(self.biases)

        #create cache arrays for adam optimizer
        self.weights_cache = np.zeros_like(self.weights)
        self.biases_cache = np.zeros_like(self.biases)

    def forward(self , inputs) :
        #perform the forward pass but return the linear part
        self.inputs = inputs
        self.output = np.dot(inputs , self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
