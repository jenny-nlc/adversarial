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


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = U.get_mnist()
    # mnist, scaled to the range 0,1.


    act_fn = 'elu'
    input_shape = (28,28,1)
    epochs=50
    batch_size = 128
    num_classes = 10

    N_DATA = x_train.shape[0]
    LENGTH_SCALE = 0.25 #setting a low length scale encourages uncertainty to be higher.
    MODEL_PRECISION = 1 #classification problem: see Gal's Thesis
    WEIGHT_REGULARIZER =  LENGTH_SCALE ** 2 / (N_DATA * MODEL_PRECISION)
    DROPOUT_REGULARIZER = 1 / (MODEL_PRECISION * N_DATA)
    N_CLASSES = 2


    model = Sequential()
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
                            input_shape=input_shape,
                            weight_regularizer=WEIGHT_REGULARIZER,
                            dropout_regularizer=DROPOUT_REGULARIZER,
    ))
    model.add(ConcreteDropout(Conv2D(64, (3,3),
                                    activation=act_fn),
                            weight_regularizer=WEIGHT_REGULARIZER,
                            dropout_regularizer=DROPOUT_REGULARIZER,
    ))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(ConcreteDropout(Dense(128, activation=act_fn),
                            weight_regularizer=WEIGHT_REGULARIZER,
                            dropout_regularizer=DROPOUT_REGULARIZER,
    ))
    model.add(ConcreteDropout(Dense(num_classes, activation='softmax'),
                            weight_regularizer=WEIGHT_REGULARIZER,
                            dropout_regularizer=DROPOUT_REGULARIZER,
    ))

    model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy'])

    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks= [TrackConcreteDropoutP(model)]) #check the p values of the c d are converging.
    fname = U.gen_save_name('save/mnist_cdrop_cnn_run.h5')
    save_model(model, fname)
