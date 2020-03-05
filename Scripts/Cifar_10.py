import glob
import pickle
import tensorflow as tf
import numpy as np
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


nows   = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = f"./logs/{nows}"

tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

print(f"logdir={logdir}")

batches = glob.glob("/content/cifar-10-batches-py/*data_batch_*")
labels = {0:"an airplane", 1:"an automobile", 2:"a bird", 3:"a cat", 4:"a deer", 5:"a dog", 6:"a frog", 7:"a horse", 8:"a ship", 9:"a truck"}

for x in batches:
  with open(x, 'rb') as f:
    d = pickle.load(f, encoding="latin1")
  X = d['data'].reshape((len(d['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
  Y = d['labels']

X_train = np.array([x.astype("float32") / 255.0 for x in X])
Y_train = np.array([np.float32(y) for y in Y])

with open("/content/cifar-10-batches-py/test_batch", 'rb') as f:
  d = pickle.load(f, encoding='latin1')
  X_test = d["data"].reshape((len(d['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
  X_test = np.array([x.astype("float32") / 255.0 for x in X_test])
  Y_test = np.array([np.float32(y) for y in d["labels"]])

net = Sequential()
net.add(Flatten(input_shape=(32, 32, 3)))
net.add(Dense(30, activation="tanh"))
net.add(Dense(20, activation="tanh"))
net.add(Dense(10, activation="softmax"))
net.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer=tf.keras.optimizers.Adam(0.001))
net.fit(X_train, Y_train, epochs=25, batch_size=128, callbacks=[tensorboard_callback])
net.evaluate(X_test, Y_test)