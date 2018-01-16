"""
This is a model trained only on the 3 / 7s mnist problem.
"""
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
from keras.models import save_model
from keras import backend as K
import src.utilities as U
from src.concrete_dropout import ConcreteDropout
import numpy as np

def mnist_to_3s_and_7s(mnist):
    x_train, y_train, x_test, y_test = mnist
    train_labels = y_train.argmax(axis=1)
    test_labels  = y_test.argmax(axis=1)

    x_train_37 = x_train[np.logical_or(train_labels == 3,train_labels == 7)]
    x_test_37  = x_test[np.logical_or(test_labels == 3,test_labels == 7)]

    y_train_37 = keras.utils.to_categorical(
        train_labels[np.logical_or(train_labels == 3,train_labels == 7)] == 7)
    y_test_37 = keras.utils.to_categorical(
        test_labels[np.logical_or(test_labels == 3,test_labels == 7)] == 7)
    
    return x_train_37, y_train_37, x_test_37, y_test_37

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

def define_cdropout_3s_7s(N_DATA=12396):
    LENGTH_SCALE = 0.25 #setting a low length scale encourages uncertainty to be higher.
    MODEL_PRECISION = 1 #classification problem: see Gal's Thesis
    WEIGHT_REGULARIZER =  LENGTH_SCALE ** 2 / (N_DATA * MODEL_PRECISION)
    DROPOUT_REGULARIZER = 1 / (MODEL_PRECISION * N_DATA)
    N_MC = 50
    N_CLASSES = 2
    input_shape = (28,28,1)
    act_fn = 'elu'    
    model = Sequential()
    model.add(ConcreteDropout(Conv2D(32, kernel_size=(3,3),
        activation=act_fn),
                              input_shape=input_shape,
                              weight_regularizer=WEIGHT_REGULARIZER,
                              dropout_regularizer=DROPOUT_REGULARIZER,
    ))
    model.add(ConcreteDropout(Conv2D(64, (3,3), activation=act_fn),
                              weight_regularizer=WEIGHT_REGULARIZER,
                              dropout_regularizer=DROPOUT_REGULARIZER,
    ))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(ConcreteDropout(Dense(128, activation=act_fn),
                              weight_regularizer=WEIGHT_REGULARIZER,
                              dropout_regularizer=DROPOUT_REGULARIZER,
                              ))
    model.add(ConcreteDropout(Dense(N_CLASSES, activation='softmax'),
                              weight_regularizer=WEIGHT_REGULARIZER,
                              dropout_regularizer=DROPOUT_REGULARIZER,
    ))
    return model

if __name__ == "__main__":

    x_train, y_train, x_test, y_test = mnist_to_3s_and_7s(U.get_mnist())
    # mnist, scaled to the range 0,1.


    epochs=50
    batch_size = 128
    model = define_cdropout_3s_7s()
    model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy'])

    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks= [TrackConcreteDropoutP(model)]) #check the p values of the c d are converging.

    model.save_weights('mnist_cdrop_3s_7s.h5')
