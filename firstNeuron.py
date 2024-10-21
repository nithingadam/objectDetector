import sys
import numpy as np
import matplotlib 
from nnfs.datasets import spiral_data
import math


# nnfs.init()
np.random.seed(0)
inputs = [
    [0.2249, 0.8401, 0.5464],
    [0.6197, 0.0812, 0.3812],
    [0.7616, 0.9732, 0.4202],
    [0.3916, 0.1041, 0.6440]
]

weights = [
    [0.25, 0.75, 0.50, 0.60],
    [0.15, 0.45, 0.85, 0.90],
    [0.40, 0.35, 0.20, 0.55]
]
bias = [0.10, 0.20, 0.30]
output = 0

# for i in range(len(inputs)):
#     output+=(inputs[i] * weights[i])
# output+=bias


output = np.dot(weights, inputs) + bias;

print(output)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# Define the ReLU activation class
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims=True)
        self.output = probabilities


X,y  = spiral_data(samples= 100, classes = 3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])



