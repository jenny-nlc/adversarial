import tensorflow as tf
import keras
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from keras.models import load_model, save_model
from keras import backend as K
import itertools as itr
from cleverhans.attacks_tf import fgm
import os

from sklearn.metrics import roc_curve, roc_auc_score
from src.utilities import *
import argparse

parser = arparse.ArgumentParser()
parser.add_argument('eps_min', type = float, default = 0.1,
                    description = "Minimum value of epsilon to generate \
                    adverserial examples with FGM")
parser.add_argument('eps_max', type = float, default = 1,
                    description = "Max value of epsilon to generate \
                    adverserial examples with")
parser.add_argument('N_eps', type = float, default = 10,
                    description = "Number of values of epsilon to use \
                    (linspace eps_min eps_max)")
parser.add_argument('N_data', type = int, default = 100,
                    description = "Number of examples of adverserial and \
                    non-adverserial examples to use. If 0 will use the entire \
                    dataset")
parser.add_argument('norm', default = 'inf',
                    description = "which norm to use: currently <- {1,2,inf}")
parser.add_argument('N_mc', default = 50, type = int,
                    description = "Number of MC forward passes to use.")

args = parser.parse_args()

if args.norm == 'inf':
    norm = np.inf
elif args.norm = '1':
    norm = 1
elif args.norm = '2':
    norm = 2
else:
    raise NotImplementedError("Norms other than 1,2, inf not implemented")


eps = np.linspace(args.eps_min,args.eps_max,args.N_eps) #some random values of epsilon
SYNTH_DATA_SIZE = args.N_data #actually twice this but whatever# %%



x_test, y_test, x_train, y_train = get_mnist()


K.set_learning_phase(True)
#load the pre-trained model (trained by another file)
model = load_model('mnist_cnn.h5')

n_mc = 50

x = K.placeholder(shape = [None] +  list(x_test.shape[1:]))
mc_preds_tensor = mc_dropout_preds(model, x, n_mc)
mean_entropy_tensor  = m_entropy(mc_preds_tensor)
bald_tensor     = BALD(mc_preds_tensor)
get_output = K.function([x], [mc_preds_tensor,
                              mean_entropy_tensor,
                              bald_tensor])


preds_tensor = K.mean(mc_preds_tensor, axis = 0)

#create a synthetic training set at various epsilons, and evaluate the ROC curves on it

x_real = x_test[np.random.randint(x_test.shape[0], size=SYNTH_DATA_SIZE)]
x_to_adv = x_test[np.random.randint(x_test.shape[0], size=SYNTH_DATA_SIZE)]

x_advs_plot = [tile_images([x_to_adv[i] for i in range(10)], horizontal = False) ]

x_real_label = [0 for _ in range(SYNTH_DATA_SIZE)] #label zero for non adverserial input
x_adv_label = [1 for _ in range(SYNTH_DATA_SIZE)]


adv_distances = []
bald_aucs = []
H_aucs = []

plt.figure()
for ep in eps:
    adv_tensor = fgm(x, preds_tensor, eps = ep, ord = norm, clip_min = 0, clip_max = 1)

    #choose a random sample from the test set


    x_adv = adv_tensor.eval(session = K.get_session(),
                            feed_dict = {x: x_to_adv})


    #calculate the L-norm distances between the adv examples and the originals
    dists = batch_L_norm_distances(x_to_adv, x_adv, ord = norm)
    adv_distances.append(dists.mean())

    #we now have some random samples of real and adverserial data.
    #shuffle them up

    x_synth = np.concatenate([x_real, x_adv])
    y_synth = np.array(x_real_label + x_adv_label)


    #save the adverserial examples to plot
    x_advs_plot.append(tile_images([x_adv[i] for i in range(10)],
                        horizontal = False))


    #get the entropy and bald on this task

    _, entropy, bald = get_output([x_synth])

    fpr_entropy, tpr_entropy,_ = roc_curve(y_synth, entropy, pos_label = 1)
    fpr_bald, tpr_bald, _      = roc_curve(y_synth, bald,    pos_label = 1)

    AUC_entropy = roc_auc_score(y_synth, entropy)
    AUC_bald    = roc_auc_score(y_synth, bald)

    H_aucs.append(AUC_entropy)
    bald_aucs.append(AUC_bald)

    plt.clf()
    plt.plot(fpr_entropy, tpr_entropy,
             label = "Entropy, AUC: {}".format(AUC_entropy))
    plt.plot(fpr_bald, tpr_bald,
             label = "BALD, AUC: {}".format(AUC_bald))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig(os.path.join("output", "ROC_eps_{}.png".format(ep)))

#plot the adverserial images
plt.figure()
tile = tile_images(x_advs_plot, horizontal = True)
plt.imshow(tile, cmap = 'gray_r')
plt.savefig(os.path.join("output", "adv_images_ep_{}_to_{}.png".format(eps.min(), eps.max())))

plt.figure()
plt.plot(eps, adv_distances)
plt.xlabel("FGSM Epsilon")
plt.ylabel("Average L{} distance of advererial images".format(norm))
plt.savefig(os.path.join("output", "{}_norm_eps_vs_avg_dist.png".format(norm)))

plt.figure()
plt.plot(eps, H_aucs, label = "Entropy")
plt.plot(eps, bald_aucs, label = "BALD")
plt.xlabel('FGM Epsilon')
plt.ylabel('AUC')
plt.savefig(os.path.join("output", "{}_norm_eps_vs_auc.png".format(norm)))
