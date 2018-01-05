
"""
Compare the predictions of concrete dropout to mcmc integration over network uncertainty
"""
import keras
import numpy as np
from keras import backend as K
import sys
sys.path.append('..')
import src.utilities as U
import scipy.special
from src.concrete_dropout import ConcreteDropout
from keras.layers import Dense
from sklearn.datasets import make_classification
import src.mcmc as mcmc
import pickle        
import os


H_ACT = 'relu'  # Might not be optimal, but a standard choice.
N_HIDDEN_UNITS = 500
N_DATA = 100
LENGTH_SCALE = 1e-2 #setting a low length scale encourages uncertainty to be higher.
MODEL_PRECISION = 1e1
WEIGHT_DECAY = LENGTH_SCALE ** 2 / (2 * N_DATA * MODEL_PRECISION)
WEIGHT_REGULARIZER =  LENGTH_SCALE ** 2 / (N_DATA * MODEL_PRECISION)
DROPOUT_REGULARIZER = 2 / (MODEL_PRECISION * N_DATA)
N_MC = 50
N_CLASSES = 2  # number of classes

#create a unique folder for the output of this script

PATH_NAME = U.create_unique_folder('save/round_')
def define_cdropout_model():
    """
    Define the cdropout model. This is written as a function for easier loading
    from the plotting script
    """
    inputs = keras.layers.Input(shape=(2,))

    h1 = Dense(N_HIDDEN_UNITS, activation=H_ACT)(inputs)
    h2 = ConcreteDropout(Dense(N_HIDDEN_UNITS, activation=H_ACT),
                         weight_regularizer=WEIGHT_REGULARIZER,
                         dropout_regularizer=DROPOUT_REGULARIZER)(h1)
    predictions = ConcreteDropout(Dense(N_CLASSES, activation='softmax'),
                                  weight_regularizer=WEIGHT_REGULARIZER,
                                  dropout_regularizer=DROPOUT_REGULARIZER)(h2)
    model = keras.models.Model(inputs=inputs, outputs=predictions)
    return model, inputs

def define_standard_model():
    """
    Define a basic model using the standard architecture and the same learning rate,
    but no dropout, only weight decay
    TODO: make sure the weight regularizer matches the dropout one in a sensible way
    """

    model = keras.models.Sequential()
    model.add(Dense(500, activation=H_ACT, input_shape=(2,)))
    model.add(
        Dense(
            500,
            activation=H_ACT,
            kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY)))
    model.add(
        Dense(
            N_CLASSES,
            activation='softmax',
            kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY)))
    return model



def train_cdropout_model(x, y):
    """
    Define and train the cdropout model, also saveing it for later use
    """
    K.set_learning_phase(True)
    model,_ = define_cdropout_model()

    model.compile(
        optimizer='sgd',
        loss=keras.losses.categorical_crossentropy,
        metrics=['accuracy'])
    model.fit(x,y,epochs=500)
    
    #save model weights
    fname = os.path.join(PATH_NAME,'cdropout_toy_model_weights.h5')
    model.save_weights(fname)
    return

def train_hmc_model(x,y):
    model = define_standard_model()
    model.compile(optimizer='sgd', loss = keras.losses.categorical_crossentropy,metrics=['accuracy'])
     # N.B: this line does basically nothing because the hmc code doesn't use the optimizer, it's
    # hand written in numpy. However, setting the loss function like this avoids passing it into
    # the HMC function
    ensemble_weights = mcmc.HMC_ensemble(model,
                x,
                y,
                N_mc=N_MC, #number of models in the ensemble
                ep=5e-3, #the step size epsilon
                tau=1, #the number of steps before a metropolis step. I found just one is the fastest (Langevin)
                burn_in=4000, #the burn in. The normal network converges in <500 epochs so this should be ok.
                samples_per_init=3, #number of times to sample the chain before re-initialising.
                sample_every=500) #amount of time to run the network before re-sampling
    fname = os.path.join(PATH_NAME,'hmc_ensemble_weights.pickle')
    with open(fname, 'wb') as f:
        pickle.dump(ensemble_weights, f)
    return 

if __name__=="__main__":
    n_informative = np.random.randint(1,3) #one or two
    n_redundant = 2 - n_informative
    data, labels = make_classification(n_samples=N_DATA, n_classes=N_CLASSES, n_features=2,
                                       n_redundant=n_redundant, n_informative=n_informative,
                                       n_clusters_per_class=1, class_sep=1)
    if n_informative==1:
        data[:, 0] = scipy.special.j1(data[:, 0])  # this could be any function really.
    data -= data.mean(axis=0)
    data /= data.std(axis=0)
    x = data
    y = keras.utils.to_categorical(labels)
    
    train_cdropout_model(x,y)
    train_hmc_model(x,y)
    
    #save toy data.
    fname = os.path.join(PATH_NAME,'toy_dataset.pickle')
    with open(fname, 'wb') as f:
        pickle.dump((x,y), f)
