import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf
import os, datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

iris = sns.load_dataset("iris")
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

LE = OneHotEncoder(sparse=False)

def SoftMax(x):
  return np.exp(x) /np.exp(x).sum()

x = iris.iloc[:,:4]
y = pd.get_dummies(iris["species"])
X_train, x_test, Y_train, y_test = train_test_split(x,y)

net = Sequential()
net.add(Dense(10, input_shape=(X_train.shape[1],), activation="relu"))
net.add(Dense(3, activation="softmax"))
net.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(0.03), metrics=["accuracy"])
net.fit(X_train, Y_train, epochs=150, batch_size=5, verbose=True, callbacks=[tensorboard_callback])
net.evaluate(x_test, y_test)