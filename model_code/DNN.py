# All the imports we will be needing
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import seaborn as sns

# We first define a class for our neural network
# Then we will define all the functions which it will need one by one
class DNN():
    # First, we have to write an init function to initialize values which the model needs
    def __init__(self, sizes: list, activation='sigmoid'):
        self.sizes = sizes
        self.num_layers = len(sizes)

        if activation == 'sigmoid':
            self.activation = self.sigmoid
        elif activation == 'relu':
            self.activation = self.relu
        else:
            raise ValueError("Activation can either be 'sigmoid' or 'relu'. Given neither")
        
        self.params = self.initialize()
        self.cache = {} # for saving the activations
    # Defines the functionality of the ReLU activation function
    def relu(self, x, grad=False):
        if grad:
            x = np.where(x<0, 0, x)
            x = np.where(x>=0, 1, x)
            return x
        return np.maximum(0,x)

    # Defines the functionality of the sigmoid activation function
    def sigmoid(self, x, grad=False):
        if grad:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

   
    # Defines the functionality of the softmax activation function
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    # Here, we initialize the weights and biases for all the layers
    # We will be using the He initialization
    def initialize(self):
        params = {}
        for i in range(1, self.num_layers):
            # He init
            scale = np.sqrt(1./self.sizes[i-1])
            params[f"W{i}"] = np.random.randn(self.sizes[i], self.sizes[i-1]) * scale #Return a sample (or samples) from the “standard normal” distribution.
            params[f"b{i}"] = np.zeros((self.sizes[i], 1))

        return params
    
    # Defines the functionality to move forward through the neural network
    # We use the cache which we defined the init function
    def forward(self, x):
        self.cache["X"] = x
        self.cache["A0"] = x.T

        for i in range(1, self.num_layers):
            self.cache[f"Z{i}"] = self.params[f"W{i}"] @ self.cache[f"A{i-1}"] + self.params[f"b{i}"] # The same W*A[i-1] + b 
            # We apply the selected activation function to all layers other than the last layer
            # Last layer uses softmax so we get probabilities for each class
            if (i < self.num_layers-1):
                self.cache[f"A{i}"] = self.activation(self.cache[f"Z{i}"])
            else:
                self.cache[f"A{i}"] = self.softmax(self.cache[f"Z{i}"])
        
        return self.cache[f"A{self.num_layers-1}"]
        # Defines the functionality to move back through the neural net, calculating all the necessary gradients as we go backward
    def backprop(self, y, y_hat):
        batch_size = y.shape[0]
        self.grads = {}

        l = self.num_layers - 1
        dZ = y_hat - y.T
        self.grads[f"W{l}"] = (1./batch_size) * (dZ @ self.cache[f"A{l-1}"].T) # Same as (dZ*A[l-1])/total
        self.grads[f"b{l}"] = (1./batch_size) * np.sum(dZ, axis=1, keepdims=True) # Same as (dZ/total)
        dA = self.params[f"W{l}"].T @ dZ # Same as W*dZ

        for i in range(l-1, 0, -1):
            dZ = dA * self.activation(self.cache[f"Z{i}"], grad=True)
            self.grads[f"W{i}"] = (1./batch_size) * (dZ @ self.cache[f"A{i-1}"].T)
            self.grads[f"b{i}"] = (1./batch_size) * np.sum(dZ, axis=1, keepdims=True)
            if i>1:
                dA = self.params[f"W{i}"].T @ dZ
        
        return self.grads
    
    # This actually updates the weights and biases
    # This is where the model "learns"
    def optimize(self, lr=0.001):
        for key in self.params:
            self.params[key] = self.params[key] - lr*self.grads[key]
    