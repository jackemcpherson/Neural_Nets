import glob
import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D

batches = glob.glob(
    '/Data_Files/cifar-10-batches-py/*data_batch_*')
labels = {0: "an airplane", 1: "an automobile", 2: "a bird", 3: "a cat", 4: "a deer", 5: "a dog", 6: "a frog",
          7: "a horse", 8: "a ship", 9: "a truck"}

for x in batches:
    with open(x, 'rb') as f:
        d = pickle.load(f, encoding="latin1")
    X = d['data'].reshape((len(d['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    Y = d['labels']

X_train = np.array([x.astype("float32") / 255.0 for x in X])
Y_train = np.array([np.float32(y) for y in Y])

with open("/Data_Files/cifar-10-batches-pycifar-10-batches-py/test_batch", 'rb') as f:
    d = pickle.load(f, encoding='latin1')
    X_test = d["data"].reshape((len(d['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    X_test = np.array([x.astype("float32") / 255.0 for x in X_test])
    Y_test = np.array([np.float32(y) for y in d["labels"]])

net = Sequential()
net.add(Conv2D(16, (3, 3), activation="relu", padding="same"))
net.add(MaxPool2D())
net.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
net.add(MaxPool2D())
net.add(Flatten(input_shape=(32, 32, 3)))
net.add(Dense(30, activation="tanh"))
net.add(Dense(20, activation="tanh"))
net.add(Dense(10, activation="softmax"))
net.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer=tf.keras.optimizers.Adam(0.001))
net.fit(X_train, Y_train, epochs=10, batch_size=50, validation_data=(X_test, Y_test))
