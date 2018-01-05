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


plt.rcParams['figure.figsize'] = 8, 8
#use true type fonts only
plt.rcParams['pdf.plottype'] = 42 
plt.rcParams['ps.plottype'] = 42 
def entropy(p):
    """
    p is a n x n_classes array; return the entropy
    for each point. This function works on numpy arrays
    (the one in utils works on tensors)
    """
    return np.sum(- p * np.log(p + 1e-8), axis=-1)

def mk_plots(xx, yy, x, y, probs, entropy, bald):
    decision = probs.argmax(axis=1)

    f, ax = plt.subplots(2, 2)
    backgrounds = [decision, entropy, bald, probs[:, 1]]
    titles = [
        'Decision Boundaries',
        'Predictive Entropy',
        'log of BALD',
        'Probabilty of First Class']
    backcols = [plt.cm.Spectral, plt.cm.gray, plt.cm.gray, plt.cm.viridis]
    axlist = [a for a in ax.flatten()]
    extent = [xx.min(), xx.max(), yy.min(), yy.max()]
    for (ax, field, c, title) in zip(axlist, backgrounds, backcols, titles):
        ax.imshow(field.reshape(xx.shape).T, cmap=c, origin='lower', extent=extent)
        ax.scatter(x[:, 0], x[:, 1], c=y.argmax(axis=1), cmap=plt.cm.Set1_r)
        ax.set_title(title)


with open('save/toy_dataset.pickle', 'rb') as f:
    x, y = pickle.load(f)


around = 5
xx, yy = np.meshgrid(np.linspace(x[:, 0].min() -
                                 around, x[:, 0].max() +
                                 around, 100), np.linspace(x[:, 1].min() -
                                                           around, x[:, 1].max() +
                                                           around, 100))

plot_x = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)

def make_cdropout_plots(save=True):
    """
    Make plots for the concrete dropout model.
    """
    K.set_learning_phase(True)
    model, inputs = C.define_cdropout_model()
    model.load_weights('save/cdropout_toy_model_weights.h5')


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
        fname = U.gen_save_name('output/cdrop_vs_mcmc/cdropout_plots.png')
        plt.savefig(fname)


def make_hmc_plots(save=False):
    with open('save/hmc_ensemble_weights.pickle', 'rb') as f:
        hmc_ensemble_weights = pickle.load(f)

    hmc_model = C.define_standard_model()
    preds = mcmc.HMC_ensemble_predict(hmc_model, hmc_ensemble_weights, plot_x)
    plot_entropy = entropy(preds)
    plot_expected_entropy = entropy(preds).mean(axis=0)
    plot_bald = plot_entropy - plot_expected_entropy
    mk_plots(xx, yy, x, y, preds, plot_entropy, plot_bald)
    if save:
        fname = U.gen_save_name('output/cdrop_vs_mcmc/mcmc_plots.png') 
        plt.savefig(fname)

if __name__ == "__main__":
   make_cdropout_plots(save=True) 
   make_hmc_plot(save=True)
