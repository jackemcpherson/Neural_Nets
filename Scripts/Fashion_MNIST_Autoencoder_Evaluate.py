import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import fashion_mnist

net = load_model("Auto Encoder")

def get_random_inputs(xs, ys, n=1):
    indicies = np.random.choice(xs.shape[0], size=n)
    sample_x = xs[indicies]
    sample_y = ys[indicies]

    return sample_x, sample_y

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.astype(np.float32) / 255.0
x_test  = x_test.astype(np.float32) / 255.0

x, _ = get_random_inputs(x_test, y_test, n=1)

out = net.predict(x)

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.set_title("Original image")
ax1.imshow(x.squeeze(), cmap="gray")

ax2.set_title("Output image")
ax2.imshow(out.squeeze(), cmap="gray")

plt.show()