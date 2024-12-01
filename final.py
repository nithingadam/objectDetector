import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle

# Function to process and flatten an image
def image_to_array(image_path):
    image = Image.open(image_path)
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((120, 120))  # Resize image
    image_array = np.array(image) / 255.0  # Normalize pixel values
    return np.expand_dims(image_array, axis=0)  # Add channel dimension

# Convolution Layer
class ConvLayer:
    def __init__(self, num_filters, kernel_size, stride=1, padding=0):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.filters = np.random.randn(num_filters, kernel_size, kernel_size) * 0.1

    def _apply_padding(self, input_image):
        if self.padding == 0:
            return input_image
        else:
            return np.pad(input_image, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

    def forward(self, input_image):
        self.input_image = input_image
        input_image = self._apply_padding(input_image)
        num_channels, h, w = input_image.shape
        output_dim = ((h - self.kernel_size) // self.stride) + 1
        output = np.zeros((self.num_filters, output_dim, output_dim))

        for f in range(self.num_filters):
            filter = self.filters[f]
            for i in range(0, output_dim):
                for j in range(0, output_dim):
                    region = input_image[:, i * self.stride:i * self.stride + self.kernel_size, j * self.stride:j * self.stride + self.kernel_size]
                    output[f, i, j] = np.sum(region * filter)
        return output

    def backward(self, dvalues, learning_rate):
        for i in range(self.num_filters):
            self.filters[i] -= learning_rate * dvalues[i]

# Max Pooling Layer
class MaxPoolingLayer:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input_image):
        num_filters, h, w = input_image.shape
        output_dim = ((h - self.pool_size) // self.stride) + 1
        output = np.zeros((num_filters, output_dim, output_dim))

        for f in range(num_filters):
            for i in range(0, h - self.pool_size + 1, self.stride):
                for j in range(0, w - self.pool_size + 1, self.stride):
                    region = input_image[f, i:i + self.pool_size, j:j + self.pool_size]
                    output[f, i // self.stride, j // self.stride] = np.max(region)

        return output

# ReLU Activation
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# Flatten Layer
class FlattenLayer:
    def forward(self, inputs):
        self.output = inputs.flatten().reshape(1, -1)

# Dense Layer with Xavier Initialization and Gradient Clipping
class Layer_dense:
    def __init__(self, inputs, neurons):
        self.weights = np.random.randn(inputs, neurons) * np.sqrt(2. / (inputs + neurons)).astype(np.float32)
        self.biases = np.zeros((1, neurons), dtype=np.float32)

    def forward(self, inputs):
        self.input = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues, learning_rate, max_grad_norm=5.0):
        dweights = np.dot(self.input.T, dvalues)
        dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Clip gradients to avoid explosion
        dweights = np.clip(dweights, -max_grad_norm, max_grad_norm)
        dbiases = np.clip(dbiases, -max_grad_norm, max_grad_norm)

        self.weights -= learning_rate * dweights
        self.biases -= learning_rate * dbiases

# Output Layer
class OutputLayer:
    def __init__(self, inputs):
        self.weights = np.random.randn(inputs, 1) * 0.01
        self.biases = np.zeros((1, 1), dtype=np.float32)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues, learning_rate):
        dweights = np.dot(self.output.T, dvalues) # self.output -> self.input
        dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.weights -= learning_rate * dweights
        self.biases -= learning_rate * dbiases

# Mean Squared Error Loss
class MeanSquaredError:
    def forward(self, y_pred, y_true):
        print(y_pred, y_true)
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.size

# Functions to save and load weights



def save_weights(epoch):
    weights = {
        "conv1": conv1.filters,
        "conv2": conv2.filters,
        "dense1_weights": dense1.weights,
        "dense1_biases": dense1.biases,
        "output_weights": output_layer.weights,
        "output_biases": output_layer.biases,
    }
    filename = f'weights_epoch_{epoch}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(weights, f)
    print(f"Weights saved to {filename}")

def load_weights(epoch):
    filename = f'weights_epoch_{epoch}.pkl'
    try:
        with open(filename, 'rb') as f:
            weights = pickle.load(f)
        conv1.filters = weights["conv1"]
        conv2.filters = weights["conv2"]
        dense1.weights = weights["dense1_weights"]
        dense1.biases = weights["dense1_biases"]
        output_layer.weights = weights["output_weights"]
        output_layer.biases = weights["output_biases"]
        print(f"Weights loaded from {filename}")
    except FileNotFoundError:
        print(f"No weights found for epoch {epoch}. Starting fresh.")


image_path = 'img_33.jpeg'
image_array = image_to_array(image_path)
actual_count = 0.33
target_output = np.array([[actual_count]])  # True count

# Calculate flattened input size dynamically
input_image_size = 120
conv1_output_size = (input_image_size - 3 + 2 * 1) // 1 + 1  # After Conv1
pool1_output_size = (conv1_output_size - 2) // 2 + 1         # After Pool1
conv2_output_size = (pool1_output_size - 3 + 2 * 1) // 1 + 1 # After Conv2
pool2_output_size = (conv2_output_size - 2) // 2 + 1         # After Pool2
flatten_input_size = 64 * pool2_output_size * pool2_output_size  # 64 filters

# Initialize Layers
conv1 = ConvLayer(num_filters=32, kernel_size=3, stride=1, padding=1)
pool1 = MaxPoolingLayer(pool_size=2, stride=2)
activation1 = Activation_ReLU()
conv2 = ConvLayer(num_filters=64, kernel_size=3, stride=1, padding=1)
pool2 = MaxPoolingLayer(pool_size=2, stride=2)
activation2 = Activation_ReLU()
flatten = FlattenLayer()

dense1 = Layer_dense(flatten_input_size, 128)
output_layer = OutputLayer(128)
mse_loss = MeanSquaredError()


# Check for saved weights and load them if available
start_epoch = 0
saved_files = [f for f in os.listdir() if f.startswith("weights_epoch_")]
if saved_files:
    latest_file = max(saved_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    start_epoch = int(latest_file.split('_')[-1].split('.')[0])
    load_weights(start_epoch)


# Training function with weights saved after each epoch
def train_model(image_array, target_output, epochs=50, learning_rate=1e-5):
    for epoch in range(epochs):
        # Forward pass
        
        conv_output1 = conv1.forward(image_array)
        activation1.forward(conv_output1)
        pool_output1 = pool1.forward(activation1.output)
        conv_output2 = conv2.forward(pool_output1)
        activation2.forward(conv_output2)
        pool_output2 = pool2.forward(activation2.output)
        flatten.forward(pool_output2)
        dense1.forward(flatten.output)
        output_layer.forward(dense1.output)

        # Calculate loss
        loss = mse_loss.forward(output_layer.output, target_output)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

        predicted_count = output_layer.output[0][0]  # Since output is a single value, we take the first element
        print(f"Predicted count: {round(predicted_count*100, 2)}")
        # if(predicted_count>actual_count):
        #     break
        # Backward pass
        dvalues = mse_loss.backward(output_layer.output, target_output)
        output_layer.backward(dvalues, learning_rate)
        dense1.backward(dvalues, learning_rate)

        # Save weights after each epoch
        save_weights(epoch + 1)


# Input Image Processing


# Train the model
train_model(image_array, target_output, epochs=25, learning_rate=1e-4)
