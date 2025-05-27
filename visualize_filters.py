# visualize_filters.py
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("cnn_model.h5")

# First conv layer filters
filters, biases = model.layers[0].get_weights()
filters = (filters - filters.min()) / (filters.max() - filters.min())

fig, axs = plt.subplots(4, 8, figsize=(8, 4))
for i in range(32):
    f = filters[:, :, :, i]
    axs[i // 8, i % 8].imshow(f[:, :, 0], cmap='gray')
    axs[i // 8, i % 8].axis('off')
plt.suptitle("First Conv Layer Filters")
plt.tight_layout()
plt.show()
