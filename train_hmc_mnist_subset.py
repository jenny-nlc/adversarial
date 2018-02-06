import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
import src.utilities as U
import src.mcmc as mcmc

def test_run(model, x_train, y_train, N_mc, ep, tau, burn_in, sample_every):
    """
    This is a demo function; used to test the parameters of the HMC by returning detailed information about each run
    """

    ensemble, losses, weights, accept_ratio = mcmc.HMC_ensemble_run(model,x_train, y_train, N_mc, ep, tau, burn_in, sample_every,
    return_extra=True, verbose=True)
    #the sampled weights, the entire loss history (so you can look at the autocorrelation), the entire weight history, and the ratio of accepted steps
    return ensemble, losses, weights, accept_ratio

if __name__ == '__main__':

    input_shape = (28, 28, 1)
    num_classes = 10
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.SGD(),
                metrics=['accuracy'])

    x_train, y_train, x_test, y_test = U.get_mnist()

    #train on a subset only
    x_train = x_train[:1000]
    y_train = y_train[:1000]

    epsilon = 1e-4 # step size
    tau = 5 # number of steps to take before the reject/accept step
    burn_in = 1000 
    sample_every = 1000
    samples_per_init = 1 #samples to take before restarting 
    N_ensemble = 20 #number of models to create

    ensemble, losses, weights, accept_ratio = test_run(model,x_train, y_train, 1, epsilon, tau, burn_in,sample_every)
    print('Accept ratio:', accept_ratio) 
    with open('save/tmp/losses.dat','w') as f:
        for l in losses:
            print(l, file=f)


#   ensemble = mcmc.HMC_ensemble(model,
#                           x_train,
#                           y_train,
#                           N_mc = N_ensemble,
#                           ep = epsilon,
#                           tau = tau,
#                           burn_in = burn_in,
#                           sample_every = sample_every,
#                           samples_per_init = samples_per_init,
#                           verbose=True):

 
