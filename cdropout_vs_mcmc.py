
"""
Compare the predictions of concrete dropout to mcmc integration over network uncertainty
"""
import tensorflow as tf
import keras
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model, save_model
from keras import backend as K
import itertools as itr
from cleverhans.attacks_tf import fgm
import os
import argparse
import sys
sys.path.append('..')
from src.utilities import *
import scipy.special
from src.concrete_dropout import ConcreteDropout
from keras.layers import Dense
from sklearn.datasets import make_classification
from src.mcmc import HMC


        
    
N_c = 2  # number of classes
data, labels = make_classification(
    n_classes=N_c, n_features=2, n_redundant=1, n_informative=1, n_clusters_per_class=1, class_sep=1)
data[:, 0] = scipy.special.j1(data[:, 0])  # this could be any function really.
data -= data.mean(axis=0)
data /= data.std(axis=0)
# plt.figure()
# plt.scatter(data[:, 0], data[:, 1], c=labels)


K.set_learning_phase(True)
weight_decay = 1e-3
h_act = 'relu'  # anything here really
inputs = keras.layers.Input(shape=(2,))

dropout_regularizer = 1e0
h1 = Dense(500, activation=h_act)(inputs)
h2 = ConcreteDropout(Dense(500, activation=h_act),
                     weight_regularizer=weight_decay,
                     dropout_regularizer=dropout_regularizer)(h1)
predictions = ConcreteDropout(Dense(N_c, activation='softmax'),
                              weight_regularizer=weight_decay,
                              dropout_regularizer=dropout_regularizer)(h2)
model = keras.models.Model(inputs=inputs, outputs=predictions)

model.compile(
    optimizer='sgd',
    loss=keras.losses.categorical_crossentropy,
    metrics=['accuracy'])


x = data
y = keras.utils.to_categorical(labels)
model.fit(x, y, epochs=500)

n_mc = 50

mc_preds_tensor = mc_dropout_preds(model, inputs, n_mc=n_mc)
entropy_mean_tensor = predictive_entropy(mc_preds_tensor)
expected_entropy_tensor = expected_entropy(mc_preds_tensor)
bald_tensor = entropy_mean_tensor - expected_entropy_tensor

get_output = K.function([inputs], [K.mean(mc_preds_tensor, axis=0),
                                   entropy_mean_tensor,
                                   bald_tensor])


plt.rcParams['figure.figsize'] = 8, 8
around = 5
xx, yy = np.meshgrid(np.linspace(data[:, 0].min() -
                                 around, data[:, 0].max() +
                                 around, 100), np.linspace(data[:, 1].min() -
                                                           around, data[:, 1].max() +
                                                           around, 100))

plot_x = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
plot_probs, plot_entropy, plot_bald = get_output([plot_x])
plot_preds = np.argmax(plot_probs, axis=1).reshape(xx.shape)


def mk_plots(xx, yy, x, y, probs, entropy, bald):
    decision = probs.argmax(axis=1)

    f, ax = plt.subplots(2, 2)
    backgrounds = [decision, entropy, bald, probs[:, 1]]
    titles = [
        'Decision Boundaries',
        'Mean Predictive Entropy',
        'BALD',
        'Probabilty of First Class']
    backcols = [plt.cm.Spectral, plt.cm.gray, plt.cm.gray, plt.cm.viridis]
    axlist = [a for a in ax.flatten()]
    for (ax, field, c, title) in zip(axlist, backgrounds, backcols, titles):
        ax.contourf(xx, yy, field.reshape(xx.shape), cmap=c)
        ax.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Set1_r)
        ax.set_title(title)


mk_plots(xx, yy, data, labels, plot_probs, plot_entropy, plot_bald)
fname = gen_save_name('output/dropout_plots.png')
plt.savefig(fname)


# now make the sample plots, but using hamiltonian monte carlo samples instead.
# we re-define a model with the same architecture but no concrete dropout
# applied,

model = keras.models.Sequential()
model.add(Dense(500, activation=h_act, input_shape=(2,)))
model.add(
    Dense(
        500,
        activation=h_act,
        kernel_regularizer=keras.regularizers.l2(weight_decay)))
model.add(
    Dense(
        N_c,
        activation='softmax',
        kernel_regularizer=keras.regularizers.l2(weight_decay)))


def entropy(p):
    """
    p is a n x n_classes array; return the entropy
    for each point
    """
    return np.sum(- p * np.log(p + 1e-8), axis=-1)


mc_preds = HMC(model,
               keras.losses.categorical_crossentropy,
               x,
               y,
               plot_x,
               15,
               4000,
               5e-3,
               1,
               3000,
               100)

plot_entropy = entropy(mc_preds.mean(axis=0))
plot_expected_entropy = entropy(mc_preds).mean(axis=0)
plot_bald = plot_entropy - plot_expected_entropy
mk_plots(xx, yy, data, labels, mc_preds.mean(axis=0), plot_entropy, plot_bald)

fname = gen_save_name('output/mcmc_plots.png') 
plt.savefig(fname)
