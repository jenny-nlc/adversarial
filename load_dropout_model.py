"""
This script trains a model on mnist using concrete dropout. Other than the
inclusion of concrete dropout, it is identical to the default keras mnist
example.
"""
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
from keras.models import save_model
from keras import backend as K
import src.utilities as U
from src.concrete_dropout import ConcreteDropout

x_train, y_train, x_test, y_test = U.get_mnist()
# mnist, scaled to the range 0,1.

model = Sequential()

act_fn = 'relu'
input_shape = (28,28,1)
epochs=12
batch_size = 128
num_classes = 10


model.add(ConcreteDropout(Conv2D(32, kernel_size=(3,3),
    activation=act_fn),
    input_shape=input_shape))
model.add(ConcreteDropout(Conv2D(64, (3,3), activation=act_fn)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(ConcreteDropout(Dense(128, activation=act_fn)))
model.add(ConcreteDropout(Dense(num_classes, activation='softmax')))

model.load_weights('mnist_cdrop_cnn.h5')

x = K.placeholder(shape=[None] + list(x_test.shape[1:]))
preds_tensor = U.mc_dropout_preds(model, x, 50)

#evaluate in batches to avoid crashing tensorflow

batches = U.batches_generator(x_test, y_test, batch_size=200)

#preds = preds_tensor.eval(session=K.get_session(), feed_dict={x: x_test}).mean(axis=0)
accs = []
for bx, by in batches:
    bpreds = preds_tensor.eval(session=K.get_session(), feed_dict={x: bx}).mean(axis=0)
    bacc = np.mean(np.equal(bpreds.argmax(axis=1), by.argmax(axis=1)))
    accs.append(bacc)
acc = np.mean(np.array(accs))
print(acc)
