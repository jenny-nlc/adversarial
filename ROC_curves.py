import tensorflow as tf
import keras
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from keras.models import load_model, save_model
from keras import backend as K
import itertools as itr
from cleverhans.attacks_tf import fgsm
import os

from sklearn.metrics import roc_curve, roc_auc_score
from src.utilities import *



eps = np.linspace(0.1,1,10) #some random values of epsilon
SYNTH_DATA_SIZE = 2000 #actually twice this but whatever# %%




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


for ep in eps:
    adv_tensor = fgsm(x, preds_tensor, eps = ep, clip_min = 0, clip_max = 1)

    #choose a random sample from the test set
    x_real = x_test[np.random.randint(x_test.shape[0], size=SYNTH_DATA_SIZE)]
    x_real_label = [0 for _ in range(SYNTH_DATA_SIZE)] #label zero for non adverserial input

    x_to_adv = x_test[np.random.randint(x_test.shape[0], size=SYNTH_DATA_SIZE)]
    x_adv = adv_tensor.eval(session = K.get_session(),
                            feed_dict = {x: x_to_adv})
    x_adv_label = [1 for _ in range(SYNTH_DATA_SIZE)]

    #we now have some random samples of real and adverserial data.
    #shuffle them up

    x_synth = np.concatenate([x_real, x_adv])
    y_synth = np.array(x_real_label + x_adv_label)
    shuffle = np.random.permutation(x_synth.shape[0])
    x_synth = x_synth[shuffle]
    y_synth = y_synth[shuffle]

    #get the entropy and bald on this task

    _, entropy, bald = get_output([x_synth])

    fpr_entropy, tpr_entropy,_ = roc_curve(y_synth, entropy, pos_label = 1)
    fpr_bald, tpr_bald, _      = roc_curve(y_synth, bald,    pos_label = 1)

    AUC_entropy = roc_auc_score(y_synth, entropy)
    AUC_bald    = roc_auc_score(y_synth, bald)

    plt.figure()
    plt.plot(fpr_entropy, tpr_entropy,
             label = "Entropy, AUC: {}".format(AUC_entropy))
    plt.plot(fpr_bald, tpr_bald,
             label = "BALD, AUC: {}".format(AUC_bald))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig(os.path.join("output", "ROC_eps_{}.png".format(ep)))
