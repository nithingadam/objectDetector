import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Function to process and flatten an image
def image_to_array(image_path):
    # Load and process the image
    image = Image.open(image_path)
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((240, 240))  # Resize image
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
            # Correctly apply padding for each channel
            return np.pad(input_image, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

    def forward(self, input_image):
        input_image = self._apply_padding(input_image)
        num_channels, h, w = input_image.shape  # Adjusting to unpack the number of channels
        output_dim = ((h - self.kernel_size) // self.stride) + 1
        output = np.zeros((self.num_filters, output_dim, output_dim))

        # Apply convolution
        for f in range(self.num_filters):
            filter = self.filters[f]
            for i in range(0, output_dim):
                for j in range(0, output_dim):
                    # Sum over all channels
                    region = input_image[
                        :,  # All channels
                        i * self.stride:i * self.stride + self.kernel_size,
                        j * self.stride:j * self.stride + self.kernel_size
                    ]
                    output[f, i, j] = np.sum(region * filter)
        return output

# Max Pooling Layer
class MaxPoolingLayer:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input_image):
        num_filters, h, w = input_image.shape  # Unpack the shape
        output_dim = ((h - self.pool_size) // self.stride) + 1  # Correct output dimension calculation
        output = np.zeros((num_filters, output_dim, output_dim))

        for f in range(num_filters):  # Loop through each filter output
            for i in range(0, h - self.pool_size + 1, self.stride):  # Adjusted loop to avoid out-of-bounds
                for j in range(0, w - self.pool_size + 1, self.stride):  # Adjusted loop to avoid out-of-bounds
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
        self.output = inputs.flatten().reshape(1, -1)  # Flatten and add batch dimension

# Dense Layer
class Layer_dense:
    def __init__(self, inputs, neurons):
        self.weights = 0.01 * np.random.randn(inputs, neurons).astype(np.float32)
        self.biases = np.zeros((1, neurons), dtype=np.float32)
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# Output Layer
class OutputLayer:
    def __init__(self, inputs):
        self.weights = 0.01 * np.random.randn(inputs, 1).astype(np.float32)  # Single output
        self.biases = np.zeros((1, 1), dtype=np.float32)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# Load and process image
image_path = 'img_count_18.jpeg'
image_array = image_to_array(image_path)
print("Original Image Shape:", image_array.shape)  # Should be (1, 240, 240)

# Apply Conv Layer 1
conv1 = ConvLayer(num_filters=32, kernel_size=3, stride=1, padding=1)
conv_output1 = conv1.forward(image_array)
print("Conv Layer 1 Output Shape:", conv_output1.shape)  # Output shape: (32, 240, 240)

# Continue with the rest of your network...

# Apply ReLU activation after Conv Layer 1
activation1 = Activation_ReLU()
activation1.forward(conv_output1)
conv_output1_activated = activation1.output

# Apply Max Pooling after Conv Layer 1
pool1 = MaxPoolingLayer(pool_size=2, stride=2)
pool_output1 = pool1.forward(conv_output1_activated)
print("Max Pooling Layer 1 Output Shape:", pool_output1.shape)  # Output shape: (32, 120, 120)

# Apply Conv Layer 2
conv2 = ConvLayer(num_filters=64, kernel_size=3, stride=1, padding=1)
conv_output2 = conv2.forward(pool_output1)
print("Conv Layer 2 Output Shape:", conv_output2.shape)  # Output shape: (64, 120, 120)

# Apply ReLU activation after Conv Layer 2
activation2 = Activation_ReLU()
activation2.forward(conv_output2)
conv_output2_activated = activation2.output

# Apply Max Pooling after Conv Layer 2
pool2 = MaxPoolingLayer(pool_size=2, stride=2)
pool_output2 = pool2.forward(conv_output2_activated)
print("Max Pooling Layer 2 Output Shape:", pool_output2.shape)  # Output shape: (64, 60, 60)

# Apply Conv Layer 3
conv3 = ConvLayer(num_filters=128, kernel_size=3, stride=1, padding=1)
conv_output3 = conv3.forward(pool_output2)
print("Conv Layer 3 Output Shape:", conv_output3.shape)  # Output shape: (128, 60, 60)

# Apply ReLU activation after Conv Layer 3
activation3 = Activation_ReLU()
activation3.forward(conv_output3)
conv_output3_activated = activation3.output

# Apply Max Pooling after Conv Layer 3
pool3 = MaxPoolingLayer(pool_size=2, stride=2)
pool_output3 = pool3.forward(conv_output3_activated)
print("Max Pooling Layer 3 Output Shape:", pool_output3.shape)  # Output shape: (128, 30, 30)

# Apply Conv Layer 4
conv4 = ConvLayer(num_filters=256, kernel_size=3, stride=1, padding=1)
conv_output4 = conv4.forward(pool_output3)
print("Conv Layer 4 Output Shape:", conv_output4.shape)  # Output shape: (256, 30, 30)

# Apply ReLU activation after Conv Layer 4
activation4 = Activation_ReLU()
activation4.forward(conv_output4)
conv_output4_activated = activation4.output

# Apply Max Pooling after Conv Layer 4
pool4 = MaxPoolingLayer(pool_size=2, stride=2)
pool_output4 = pool4.forward(conv_output4_activated)
print("Max Pooling Layer 4 Output Shape:", pool_output4.shape)  # Output shape: (256, 15, 15)

# Apply Conv Layer 5
conv5 = ConvLayer(num_filters=512, kernel_size=3, stride=1, padding=1)
conv_output5 = conv5.forward(pool_output4)
print("Conv Layer 5 Output Shape:", conv_output5.shape)  # Output shape: (512, 15, 15)

# Apply ReLU activation after Conv Layer 5
activation5 = Activation_ReLU()
activation5.forward(conv_output5)
conv_output5_activated = activation5.output

# Apply Max Pooling after Conv Layer 5
pool5 = MaxPoolingLayer(pool_size=2, stride=2)
pool_output5 = pool5.forward(conv_output5_activated)
print("Max Pooling Layer 5 Output Shape:", pool_output5.shape)  # Output shape: (512, 7, 7)

# Flatten the output for Dense layer input
flatten = FlattenLayer()
flatten.forward(pool_output5)
print("Flattened Output Shape:", flatten.output.shape)  # Output shape after flattening

# Instantiate Dense Layer 1 and apply it
dense1 = Layer_dense(flatten.output.shape[1], 128)
dense1.forward(flatten.output)
print("Dense Layer 1 Output Shape:", dense1.output.shape)  # Shape after Dense Layer 1

# Apply ReLU activation after Dense Layer 1
activation6 = Activation_ReLU()
activation6.forward(dense1.output)
print("Activation after Dense Layer 1:", activation6.output.shape)

# Instantiate Dense Layer 2 and apply it
dense2 = Layer_dense(128, 64)
dense2.forward(activation6.output)
print("Dense Layer 2 Output Shape:", dense2.output.shape)  # Shape after Dense Layer 2

# Apply ReLU activation after Dense Layer 2
activation7 = Activation_ReLU()
activation7.forward(dense2.output)
print("Activation after Dense Layer 2:", activation7.output.shape)

# Instantiate Dense Layer 3 and apply it
dense3 = Layer_dense(64, 32)
dense3.forward(activation7.output)
print("Dense Layer 3 Output Shape:", dense3.output.shape)  # Shape after Dense Layer 3

# Apply ReLU activation after Dense Layer 3
activation8 = Activation_ReLU()
activation8.forward(dense3.output)
print("Final Output after Dense Layer 3 Activation:", activation8.output.shape)

# Instantiate Output Layer and apply it
output_layer = OutputLayer(32)  # 32 is the output shape from Dense Layer 3
output_layer.forward(activation8.output)
print("Output Layer Output Shape:", output_layer.output.shape)  # Shape after Output Layer
print("Predicted Number of People:", 100 * output_layer.output.flatten()[0])  # Flatten to get the single value

# Optional: Visualize the original image
plt.imshow(image_array[0], cmap='gray')  # Use image_array[0] to remove the first dimension
plt.axis('off')
plt.show()
