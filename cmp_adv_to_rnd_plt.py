"""
TODO: make this script work on both mnist and cats
"""
from matplotlib import pyplot as plt
import pickle
import sys
import os
import numpy as np
import src.plot_utils as pu
import src.utilities as U
import argparse



plt.rcParams['figure.figsize'] = 8, 8
#use true type fonts only
plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42 
def get_nn_dists(X, xs):
    return np.array([U.calc_nn_dist(X, x) for x in xs])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('load_path', help='path to lead from')
    parser.add_argument('--norm','-n', help='norm used', default='inf')
    parser.add_argument('--cats','-c', help='whether to use cats or mnist mode', action='store_true')


    args = parser.parse_args() 
    load_path = args.load_path
    ORD       = args.norm


    if args.cats:
        with open(os.path.join(load_path, 'cats_and_dogs_pertub_data.pickle'), 'rb') as f:
            epsilons, adv_preds, adv_entropies, adv_balds, adv_vars,rnd_preds, rnd_entropies, rnd_balds, rnd_vars = pickle.load(f)
        with open(os.path.join(load_path, 'cats_and_dogs_labels.pickle'), 'rb') as f:
            y_test = pickle.load(f)
    else:
        with open(os.path.join(load_path, 'mnist_pertub_data.pickle'), 'rb') as f:
            epsilons, adv_preds, adv_entropies, adv_balds, adv_vars,rnd_preds, rnd_entropies, rnd_balds, rnd_vars = pickle.load(f)
        with open(os.path.join(load_path, 'mnist_sample.pickle'), 'rb') as f:
            x_test,y_test = pickle.load(f)



    # with open(os.path.join(load_path, 'perturbations.pickle'), 'rb') as f:
    #     advs, perturbs = pickle.load(f)

    fig, axes = plt.subplots(3,1)

    # x_train, y_train, _, _ = U.get_mnist()   

    pu.var_fill_plot(axes[0], epsilons,adv_balds, c='b', label='Adversarial Direction')
    pu.var_fill_plot(axes[0], epsilons,rnd_balds, c='r', label='Random Direction')
    axes[0].set_xlabel('Step size ({} norm)'.format(ORD))
    axes[0].set_ylabel('Average BALD')
    axes[0].legend()

    pu.var_fill_plot(axes[1], epsilons,adv_entropies, c='b', label='Adversarial Direction')
    pu.var_fill_plot(axes[1], epsilons,rnd_entropies, c='r', label='Random Direction')
    axes[1].set_xlabel('Step size ({} norm)'.format(ORD))
    axes[1].set_ylabel('Average Entropy')
    axes[1].legend()

    #calculate accuracy
    adv_accs = np.equal(adv_preds.argmax(axis=-1), y_test.argmax(axis=1)[None,:]).astype(np.float).mean(axis=1)
    rnd_accs = np.equal(rnd_preds.argmax(axis=-1), y_test.argmax(axis=1)[None,:]).astype(np.float).mean(axis=1)

    axes[2].plot(epsilons,adv_accs, c='b', label='Adversarial Direction')
    axes[2].plot(epsilons,rnd_accs, c='r', label='Random Direction')
    axes[2].set_xlabel('Step size ({} norm)'.format(ORD))
    axes[2].legend()
    axes[2].set_ylabel('Average Accuracy')

    f, ax = plt.subplots()
    pu.var_fill_plot(ax, epsilons,adv_vars, c='b', label='Adversarial Direction')
    pu.var_fill_plot(ax, epsilons,rnd_vars, c='r', label='Random Direction')
    ax.set_xlabel('Step size ({} norm)'.format(ORD))
    ax.set_ylabel('Variance Score')
    ax.legend()

    # f, ax = plt.subplots()
    # #this takes too long; look at sklearns nn function?
    # advdists = np.array([get_nn_dists(x_train, adv) for adv in advs])
    # rnddists = np.array([get_nn_dists(x_train, rnd) for rnd in perturbs])
    # pu.var_fill_plot(ax, epsilons, advdists, c='b', label='Adversarial Direction')
    # pu.var_fill_plot(ax, epsilons, rnddists, c='r', label='Random Direction')
    # ax.legend()

    plt.show()
