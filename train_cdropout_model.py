"""
This script trains a model on mnist using concrete dropout. Other than the
inclusion of concrete dropout, it is identical to the default keras mnist
example.
"""
import keras
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


class TrackConcreteDropoutP(keras.callbacks.Callback):
    def __init__(self, model: keras.models.Model):
        self.model = model
    def on_train_begin(self, logs={}):
        self.ps = []
        self.losses = []
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        ps_tensor = [x.p for x in self.model.layers if 'concrete_dropout' in x.name]
        get_ps = K.function([], ps_tensor)
        p = get_ps([])
        self.ps.append(p)
        print(" - concrete ps: ", p)
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return

model.add(ConcreteDropout(Conv2D(32, kernel_size=(3,3),
    activation=act_fn),
    input_shape=input_shape))
model.add(ConcreteDropout(Conv2D(64, (3,3), activation=act_fn)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(ConcreteDropout(Dense(128, activation=act_fn)))
model.add(ConcreteDropout(Dense(num_classes, activation='softmax')))

model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=['accuracy'])

model.fit(x_train[:20], y_train[:20],
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks= [TrackConcreteDropoutP(model)]) #check the p values of the c d are converging.

save_model(model, 'mnist_cdrop_cnn.h5')
