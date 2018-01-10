"""
Plot the results of running the script cdropout_train.py, in order to save costly recomputation.
"""
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
import src.utilities as U
from src import mcmc
import pickle
import cdropout_vs_mcmc_train_models as C
import sys
import os

if len(sys.argv) != 2:
    sys.exit("Please provide a path to load models from")

LOAD_PATH = sys.argv[1]
SAVE_PATH = os.path.join('output',LOAD_PATH.replace('/','_')) 
if not os.path.isdir(SAVE_PATH):
    os.mkdir(SAVE_PATH)
plt.rcParams['figure.figsize'] = 8, 8
#use true type fonts only
plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42 
def entropy(p):
    """
    p is a n x n_classes array; return the entropy
    for each point. This function works on numpy arrays
    (the one in utils works on tensors)
    """
    return np.sum(- p * np.log(p + 1e-8), axis=-1)

def mk_plots(xx, yy, x, y, probs, entropy, bald):
    decision = probs.argmax(axis=1)

    f, ax = plt.subplots(3, 2)
    backgrounds = [decision, entropy, bald, probs[:, 1], np.log(bald + 1e-6)]
    titles = [
        'Decision Boundaries',
        'Predictive Entropy',
        'BALD',
        'Probabilty of First Class',
        'Log of BALD',
    ]
    backcols = [plt.cm.Spectral, plt.cm.gray, plt.cm.gray, plt.cm.viridis, plt.cm.gray]
    axlist = [a for a in ax.flatten()]
    extent = [xx.min(), xx.max(), yy.min(), yy.max()]
    for (ax, field, c, title) in zip(axlist, backgrounds, backcols, titles):
        ax.imshow(field.reshape(xx.shape), cmap=c, origin='lower', extent=extent)
        ax.scatter(x[:, 0], x[:, 1], c=y.argmax(axis=1), cmap=plt.cm.Set1_r)
        ax.set_title(title)


with open(os.path.join(LOAD_PATH,'toy_dataset.pickle'), 'rb') as f:
    x, y = pickle.load(f)


around = 3
xx, yy = np.meshgrid(np.linspace(x[:, 0].min() -
                                 around, x[:, 0].max() +
                                 around, 100), np.linspace(x[:, 1].min() -
                                                           around, x[:, 1].max() +
                                                           around, 100))

plot_x = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)

def make_cdropout_plots(save=False):
    """
    Make plots for the concrete dropout model.
    """
    K.set_learning_phase(True)
    model, inputs = C.define_cdropout_model()
    model.load_weights(os.path.join(LOAD_PATH,'cdropout_toy_model_weights.h5'))


    mc_preds_tensor = U.mc_dropout_preds(model, inputs, n_mc=C.N_MC)
    entropy_mean_tensor = U.predictive_entropy(mc_preds_tensor)
    expected_entropy_tensor = U.expected_entropy(mc_preds_tensor)
    bald_tensor = entropy_mean_tensor - expected_entropy_tensor

    get_output = K.function([inputs], [K.mean(mc_preds_tensor, axis=0),
                                       entropy_mean_tensor,
                                       bald_tensor])



    plot_probs, plot_entropy, plot_bald = get_output([plot_x])

    mk_plots(xx, yy, x, y, plot_probs, plot_entropy, plot_bald)
    
    if save:
        fname = os.path.join(SAVE_PATH,'cdropout_plots.png')
        plt.savefig(fname)


def make_hmc_plots(save=False):
    with open(os.path.join(LOAD_PATH,'hmc_ensemble_weights.pickle'), 'rb') as f:
        hmc_ensemble_weights = pickle.load(f)

    hmc_model = C.define_standard_model()
    mc_preds = mcmc.HMC_ensemble_predict(hmc_model, hmc_ensemble_weights, plot_x)
    preds = mc_preds.mean(axis=0)
    plot_entropy = entropy(preds)
    plot_expected_entropy = entropy(mc_preds).mean(axis=0)
    plot_bald = plot_entropy - plot_expected_entropy
    mk_plots(xx, yy, x, y, preds, plot_entropy, plot_bald)
    if save:
        fname = os.path.join(SAVE_PATH,'mcmc_plots.png') 
        plt.savefig(fname)

if __name__ == "__main__":
   make_cdropout_plots(save=True) 
   make_hmc_plots(save=True)
