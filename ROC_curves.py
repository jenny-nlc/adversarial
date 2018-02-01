import argparse
import os
import sys

import numpy as np
from keras import backend as K
from keras.models import load_model
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from matplotlib import pyplot as plt
from keras.utils import to_categorical
import src.utilities as U
from load_dropout_model import load_drop_model
from cleverhans import attacks
from cleverhans.model import CallableModelWrapper

"""
This script calculates the ROC for various models for the basic iterative method.
TODO: use CW attack? but this has a non-straightforward generalisation...
"""

def shuffle_dataset(x, y):
    inds = np.random.permutation(x.shape[0])
    return x[inds], y[inds]

def make_random_targets(y, n_classes=10):
    """
    Return one hot vectors that differ from the labels in y
    """
    labels = y.argmax(axis=1)
    new = (labels + np.random.randint(1, n_classes - 1) ) % n_classes
    return to_categorical(new, num_classes=n_classes)
def get_models():

    models = []
    K.set_learning_phase(True)
    model = load_model('save/mnist_cnn_run_3.h5')
    models.append(('MC Dropout cnn',U.MCModel(model, model.input, n_mc=30)))

    K.set_learning_phase(False)
    model = load_model('save/mnist_cnn_run_3.h5') 
    models.append(('Deterministic CNN (dropout)',model))

    model = load_model('save/mnist_cnn_no_drop_run.h5')
    models.append(('Deterministic CNN (no dropout)', model))
    # model = load_drop_model('save/mnist_cdrop_cnn_run.h5')
    # model = U.MCModel(model, model.input,n_mc=30)
    # models.append(('Concrete Dropout CNN', model))

    # ms = []
    # for name in filter(lambda x: 'mnist_cdrop_cnn_run' in x, os.listdir('save')):
    #     print('loading model {}'.format(name))
    #     model = load_drop_model('save/' + name)
    #     ms.append(model)
    # model = U.MCEnsembleWrapper(ms, n_mc=20)
    # models.append(('Ensemble CNN (cdropout)', model))
 
    ms = []
    for name in filter(lambda x: 'mnist_cnn' in x, os.listdir('save')):
        print('loading model {}'.format(name))
        model = load_model('save/' + name)
        ms.append(model)
    model = U.MCEnsembleWrapper(ms, n_mc=20)
    models.append(('Ensemble CNN', model))

    return models

def plot(models, fpr_entropies, tpr_entropies, fpr_balds, tpr_balds, prec_entropies, rec_entropies, prec_balds, rec_balds):
    plt.figure(1)
    plt.figure(2)

    for i, (name,_) in enumerate(models):
      plt.figure(1)
      plt.plot(fpr_entropies[i], tpr_entropies[i], label="{} entropy".format(name))
      plt.plot(fpr_balds[i], tpr_balds[i], label="{} bald".format(name))

      plt.figure(2)
      plt.plot(rec_entropies[i],prec_entropies[i], label="{} entropy".format(name))
      plt.plot(rec_balds[i],prec_balds[i], label="{} bald".format(name))
      
    plt.legend()
    plt.show()

   
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--N_data', type=int, default=100, help="Number of examples \
        of adverserial and non-adverserial examples to use. If 0 will use the \
        entire dataset")

    args = parser.parse_args()

    SYNTH_DATA_SIZE = args.N_data

    x_test, y_test, x_train, y_train = U.get_mnist()


    # load the pre-trained models 
    models_to_eval = get_models() 

    # create a synthetic training set at various epsilons, 
    # and evaluate the ROC curves on it. Combine adversarial and random pertubations

    x_real = x_test[np.random.randint(x_test.shape[0], size=SYNTH_DATA_SIZE)]
    to_adv_inds = np.random.randint(x_test.shape[0], size=SYNTH_DATA_SIZE)
    x_to_adv = x_test[to_adv_inds]
    x_to_adv_labs = y_test[to_adv_inds]
    x_plus_noise = x_test[np.random.randint(x_test.shape[0], size=SYNTH_DATA_SIZE)]

    x_advs_plot = [U.tile_images([x_to_adv[i] for i in range(15)], horizontal=False)]

    # label zero for non adverserial input
    x_real_label = [0 for _ in range(SYNTH_DATA_SIZE)]
    x_plus_noise_label = [0 for _ in range(SYNTH_DATA_SIZE)]
    x_adv_label = [1 for _ in range(SYNTH_DATA_SIZE)]

    fpr_entropies = []
    tpr_entropies = []

    fpr_balds = []
    tpr_balds = []

    prec_entropies = []
    rec_entropies = []

    prec_balds = []
    rec_balds = []

    AUC_entropies = []
    AUC_balds = []

    AP_entropies = []
    AP_balds = []


    for i, (name, m) in enumerate(models_to_eval):

        input_t = K.placeholder(shape=(None, 28, 28, 1))

        wrap = CallableModelWrapper(m, 'probs')

        attack = attacks.BasicIterativeMethod(wrap, sess=K.get_session(), back='tf')

        adv_tensor = attack.generate(input_t,
                                     eps=50,
                                     nb_iter=10, 
                                     eps_iter=5,
                                  ord=1,
                                  clip_min=0,
                                  clip_max=1,
                                  y_targets = make_random_targets(x_to_adv_labs) 
        )
        x_adv = adv_tensor.eval(session=K.get_session(),
                                feed_dict={input_t: x_to_adv})
        #check the examples are really adversarial
        preds = m.predict(x_adv).argmax(axis=1)
        print("Accuracy on adv examples:", np.mean(np.equal(preds, x_to_adv_labs.argmax(axis=1))))

        # calculate the L-norm distances between the adv examples and the originals
        dists = U.batch_L_norm_distances(x_to_adv, x_adv, ord=2)
        noise = np.random.random(size=x_plus_noise.shape)
        noise /= (dists * np.linalg.norm(noise.reshape(x_plus_noise.shape[0], -1), axis=1))[:, None, None, None]
        x_plus_noise += noise
        x_plus_noise = np.clip(x_plus_noise, 0, 1)
        x_synth = np.concatenate([x_real, x_adv, x_plus_noise])
        y_synth = np.array(x_real_label + x_adv_label + x_plus_noise_label)
        #x_synth, y_synth = shuffle_dataset(x_synth, y_synth) no points

        # save the adverserial examples to plot
        x_advs_plot.append(U.tile_images([x_adv[i] for i in range(15)],
                                    horizontal=False))

        # get the entropy and bald on this task
        if hasattr(m, 'get_results'): 
            _, entropy, bald = m.get_results(x_synth)
        else:
            res = m.predict(x_synth)
            entropy = np.sum( - res * np.log(res + 1e-6), axis=1)
            bald = np.zeros(entropy.shape) # undefined

        fpr_entropy, tpr_entropy, _ = roc_curve(y_synth, entropy, pos_label=1)
        fpr_bald, tpr_bald, _ = roc_curve(y_synth, bald,    pos_label=1)

        prec_entr, rec_entr, _ = precision_recall_curve(y_synth, entropy, pos_label=1)
        prec_bald, rec_bald , _ = precision_recall_curve(y_synth, bald, pos_label=1)

        AUC_entropy = roc_auc_score(y_synth, entropy)
        AUC_bald = roc_auc_score(y_synth, bald)

        AP_entropy = average_precision_score(y_synth, entropy)
        AP_bald = average_precision_score(y_synth, bald)

        fpr_entropies.append(fpr_entropy)
        tpr_entropies.append(tpr_entropy)

        prec_entropies.append(prec_entr) 
        rec_entropies.append(rec_entr)

        prec_balds.append(prec_bald) 
        rec_balds.append(rec_bald)

        fpr_balds.append(fpr_bald)
        tpr_balds.append(tpr_bald)

        AUC_entropies.append(AUC_entropy)
        AUC_balds.append(AUC_bald)

        AP_entropies.append(AP_entropy)
        AP_balds.append(AP_bald)




        
