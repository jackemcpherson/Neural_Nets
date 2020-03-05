import os
import io
import datetime
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import TensorBoard, LambdaCallback
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

callbacks = []

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(logdir, histogram_freq=1)

callbacks.append(tensorboard_callback)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train.astype(np.float32) / 255.0
x_test  = x_test.astype(np.float32) / 255.0

LABELS = \
  [ "T-shirt/top"
  , "Trouser"
  , "Pullover"
  , "Dress"
  , "Coat"
  , "Sandal"
  , "Shirt"
  , "Sneaker"
  , "Bag"
  , "Ankle boot"
  ]

def CustomLoss(y_true, y_pred):
  return tf.math.reduce_mean(tf.math.reduce_sum(((y_pred - y_true) ** 2),axis=-1))

def to_onehot(num, size):
  x = np.zeros(size)
  x[num] = 1
  return x

y_train_onehot = np.array([to_onehot(x, 10) for x in y_train])
y_test_onehot = np.array([to_onehot(x, 10) for x in y_test])

y_train_mean = [tf.math.reduce_mean(x) for x in y_train_onehot]

net = Sequential()

net.add(Flatten(input_shape=(28, 28)))
net.add(Dense(30, activation="tanh"))
net.add(Dense(20, activation="tanh"))
net.add(Dense(10, activation="softmax"))
net.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy","mse"])
net.fit(x_train, y_train, epochs=10, batch_size=25, callbacks=callbacks)

score = net.evaluate(x_test, y_test)[1]
print(f"Model is {round(score*100,2)}% accurate")