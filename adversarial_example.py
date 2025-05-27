# adversarial_example.py
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10

model = load_model("cnn_model.h5")
(_, _), (x_test, y_test) = cifar10.load_data()
x_test = x_test / 255.0

img = tf.convert_to_tensor(x_test[1:2])
label = tf.convert_to_tensor(y_test[1:2])

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

with tf.GradientTape() as tape:
    tape.watch(img)
    prediction = model(img)
    loss = loss_object(label, prediction)

gradient = tape.gradient(loss, img)
signed_grad = tf.sign(gradient)
eps = 0.01
adv_image = img + eps * signed_grad

pred_orig = np.argmax(model.predict(img))
pred_adv = np.argmax(model.predict(adv_image))

plt.subplot(1, 2, 1)
plt.title(f"Original ({pred_orig})")
plt.imshow(img[0])
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Adversarial ({pred_adv})")
plt.imshow(adv_image[0])
plt.axis('off')
plt.show()
