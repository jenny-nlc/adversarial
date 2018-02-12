import keras
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.layers import Dropout, Dense
import src.utilities as U
import os
import h5py
from keras import backend as K

H5PATH='/data-local/lsgs/cats_dogs.h5'

def load_or_create_dataset():
    if not os.path.exists(H5PATH):
        cats = U.load_jpgs('/data-local/lsgs/PetImages/Cat')
        catlabel = np.zeros(cats.shape[0])

        dogs = U.load_jpgs('/data-local/lsgs/PetImages/Dog')
        doglabel = np.ones(dogs.shape[0])

        data = np.concatenate([cats, dogs])
        labels = np.concatenate([catlabel, doglabel])
        
        inds = np.random.permutation(data.shape[0])

        X = preprocess_input(data.astype(np.float))
        Y = keras.utils.to_categorical(labels)

        #shuffle data
        X = X[inds]
        Y = Y[inds]
        
        N = X.shape[0]
        split = int(0.8 * N) 
        
        X_train = X[:split]
        Y_train = Y[:split]

        X_test = X[split:]
        Y_test = Y[split:]
       
        #write to database file to avoid this crap later
        with h5py.File(H5PATH,'w') as f:
            tr = f.create_group('train')
            te = f.create_group('test')
            tr.create_dataset('X', data=X_train)
            tr.create_dataset('Y', data=Y_train)

            te.create_dataset('X', data=X_test)
            te.create_dataset('Y', data=Y_test)
        return X_train, Y_train, X_test, Y_test
    else:
        with h5py.File(H5PATH,'r') as f:
            X_train = f['train']['X'].value
            Y_train = f['train']['Y'].value

            X_test = f['test']['X'].value
            Y_test = f['test']['Y'].value
        return X_train, Y_train, X_test, Y_test

def define_model():
    K.set_learning_phase(True)
    vgg = VGG16(weights='imagenet')
    #re-insert VGG16's dropout layers
    fc1 = vgg.layers[-3]
    fc2 = vgg.layers[-2]
    fc3 = vgg.layers[-1] # final layer

    a = fc1.output
    a = Dropout(rate=0.5)(a) 
    a = fc2(a)
    a = Dropout(rate=0.5)(a)
    a = Dense(2, activation='softmax')(a) #replace fc3 with a two class classifier

    model = keras.models.Model(inputs = vgg.input, outputs = a)

    #freeze vgg layers
    for layer in vgg.layers[:-1]:
        layer.trainable = False
    
    return model

def define_model_resnet():
    K.set_learning_phase(True)
    rn50 = ResNet50(weights = 'imagenet', include_top='False')
    a = Dropout(rate=0.5)(rn50.output)
    a = Dense(2, activation='softmax')(a)
    
    model = keras.models.Model(inputs = rn50.input, outputs = a)
    
    #freeze resnet layers
    for layer in rn50.layers:
        layer.trainable = False
    return model

def train(X, Y, X_val, Y_val):
    model = define_model()
    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],optimizer='adam')

    #sanity check; just train for now
    model.fit(X, Y, epochs=10,  validation_data=(X_val, Y_val))
    return model

if __name__ == '__main__':
    mode = 'rnet'
    if mode == 'vgg':

        X,Y, X_val, Y_val = load_or_create_dataset()
        model = define_model()
        model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],optimizer='adam')
        model.fit(X, Y, epochs=10,  validation_data=(X_val, Y_val))
        model.save_weights('save/cats_dogs_vgg_w.h5')

    elif mode == 'rnet':

        X,Y, X_val, Y_val = load_or_create_dataset()
        model = define_model_resnet()
        model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],optimizer='adam')
        model.fit(X, Y, epochs=10,  validation_data=(X_val, Y_val))
        model.save_weights('save/cats_dogs_rn50_w.h5')
