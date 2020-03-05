from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, UpSampling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = (x_train.astype('float32') / 255.0).reshape((-1,28,28,1))
x_test = (x_test.astype('float32') / 255.0).reshape((-1,28,28,1))

net = Sequential()
net.add(Conv2D(filters=16, kernel_size=(3,3), activation="tanh", padding="same"))
net.add(MaxPool2D(pool_size=(2,2), strides=None, padding="same"))
net.add(Conv2D(filters=32, kernel_size=(3,3), activation="tanh", padding="same"))
net.add(MaxPool2D(pool_size=(2,2), strides=None, padding="same"))
net.add(Flatten())
net.add(Dense(30, activation="tanh"))
net.add(Dense(20, activation="tanh"))
net.add(Dense(10, activation="softmax"))
net.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
net.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), batch_size=250)