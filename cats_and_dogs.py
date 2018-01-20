import keras
import numpy as np
from matplotlib import pyplot as plt
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.layers import Dropout, Dense
import src.utilities as U

K.set_learning_phase(True)

cats = U.load_jpgs('datasets/PetImages/Cat')
catlabel = np.zeros(cats.shape[0])

dogs = U.load_jpgs('datasets/PetImages/Dog')
doglabel = np.ones(dogs.shape[0])

vgg = VGG16(weights='imagenet')
#re-insert VGG16's dropout layers
fc1 = vgg.layers[-3]
fc2 = vgg.layers[-2]
fc3 = vgg.layers[-1] # final layer

x = fc1.output
x = Dropout(rate=0.5)(x) 
x = fc2(x)
x = Dropout(rate=0.5)(x)
x = Dense(2, activation='softmax')(x) #replace fc3 with a two class classifier

model = keras.models.Model(inputs = vgg.input, outputs = x)

#freeze vgg layers
for layer in vgg.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy', optimizer='adam')

#sanity check; just train for now

minidata = np.concatenate([cats[:200], dogs[:200]])
minilabels = np.concatenate([catlabel[:200], doglabel[:200]])
        
miniX = preprocess_input(minidata.astype(np.float))
miniY = keras.utils.to_categorical(minilabels)

model.fit(miniX, miniY, epochs=10)
