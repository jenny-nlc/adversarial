'''Trains a simple convnet on the Manifold MNIST dataset.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import h5py
import src.utilities as U

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
mmnist = h5py.File('manifold_mnist/manifold_mnist.h5','r')
x_train = mmnist['x_train']
x_test  = mmnist['x_test']

y_train = mmnist['y_train']
y_test  = mmnist['y_test']


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test), shuffle='batch')
score = model.evaluate(x_test, y_test, verbose=0)
fname = U.gen_save_name('save/manifold_mnist_cnn_run.h5')
model.save(fname)
mmnist.close()
