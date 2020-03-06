from tensorflow.keras.layers import Flatten, Dense, Reshape
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.astype(np.float32) / 255.0
x_test  = x_test.astype(np.float32) / 255.0

def loss (y_true, y_pred):
  result     = (y_true - y_pred) ** 2
  loss_batch = tf.math.reduce_sum(result, axis=-1)
  mean_loss  = tf.math.reduce_mean(loss_batch)
  return mean_loss

net = Sequential()
net.add(Flatten(input_shape=(28, 28)))   # our input was (28,28)
net.add(Dense(256,   activation="tanh"))
net.add(Dense(128,   activation="tanh"))
net.add(Dense(10,    activation="tanh", name="z"))
net.add(Dense(128,   activation="tanh"))
net.add(Dense(256,   activation="tanh"))
net.add(Dense(28*28, activation="tanh")) # 784
net.add(Reshape((28,28)))
net.compile(loss='MSE', optimizer=Adam(0.001), metrics=["accuracy"])
net.fit(x=x_train, y=x_train, validation_data=(x_test, x_test), epochs=10, batch_size=100)
net.save("Auto Encoder")