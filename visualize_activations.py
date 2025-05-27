# visualize_activations.py
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.datasets import cifar10

model = load_model("cnn_model.h5")
activation_model = Model(inputs=model.input, outputs=[layer.output for layer in model.layers[:3]])

(_, _), (x_test, _) = cifar10.load_data()
x_test = x_test / 255.0
img = x_test[0:1]

activations = activation_model.predict(img)

for layer_activation in activations:
    num_filters = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    display_grid = np.zeros((size, size * num_filters))
    for i in range(num_filters):
        activation = layer_activation[0, :, :, i]
        display_grid[:, i * size: (i + 1) * size] = activation
    plt.figure(figsize=(15, 5))
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.axis('off')
    plt.show()
