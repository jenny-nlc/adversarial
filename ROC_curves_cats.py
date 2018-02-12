import argparse
import os
import sys
import h5py
import json

import numpy as np
from keras import backend as K
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from keras.utils import to_categorical
import src.utilities as U
from cleverhans import attacks
from cleverhans.model import CallableModelWrapper

from cats_and_dogs import define_model, H5PATH
"""
This script calculates the ROC for various models for the basic iterative method.
TODO: use CW attack? but this has a non-straightforward generalisation...
"""
def load_model(deterministic=False, name = 'save/cats_dogs_vgg_w.h5'):
    lp = not deterministic
    K.set_learning_phase(lp)
    model = define_model()
    model.load_weights(name)
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    return model


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
def get_models(n_mc=10):

    model = load_model(deterministic=True)
    yield ('Deterministic Model', model)

    model = load_model(deterministic=False)
    input_tensor = model.input
    mc_model = U.MCModel(model, input_tensor, n_mc = n_mc )
    yield ('MC Model', mc_model)
    
    # load a model ensemble 
#   ms = []
#   for name in filter(lambda x: 'vgg' in x, os.listdir('save')):
#       print('loading model {}'.format(name))
#       model = load_model(deterministic=False, name=('save/' + name))
#       ms.append(model)
#   mc_model = U.MCEnsembleWrapper(ms, n_mc=5)
#   yield ('Ensemble Model', mc_model)

def batch_gen(array, batch_size=256):
    N = array.shape[0]
    n_batches = N // batch_size + (N % batch_size != 0)
    bs=batch_size
    return (array[i*bs:(i+1)*bs] for i in range(n_batches))

def batch_eval(tensor, input_t, x,batch_size=256,verbose=False):
    bg = batch_gen(x, batch_size=batch_size)
    res = []
    for i,b in enumerate(bg):
        res.append(tensor.eval(session=K.get_session(), feed_dict={input_t: b}))
        if verbose:
            print(verbose, 'iteration: ', i)
    return np.concatenate(res, axis=0)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--N_data', type=int, default=100, help="Number of examples \
        of adverserial and non-adverserial examples to use. If 0 will use the \
        entire dataset")
    parser.add_argument('--N_mc', type=int, default=20, help="number of mc passes")
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size to use')
    parser.add_argument('--use_same_examples', action='store_true')

    args = parser.parse_args()

    SYNTH_DATA_SIZE = args.N_data

    print('Loading data...')
    h5database = h5py.File(H5PATH, 'r')
    x_test = h5database['test']['X'].value
    y_test = h5database['test']['Y'].value
    h5database.close()

    # load the pre-trained models 
    models_to_eval = get_models(n_mc=args.N_mc) 

    # create a synthetic training set at various epsilons, 
    # and evaluate the ROC curves on it. Combine adversarial and random pertubations

    x_real = x_test[np.random.randint(x_test.shape[0], size=SYNTH_DATA_SIZE)]
    to_adv_inds = np.random.randint(x_test.shape[0], size=SYNTH_DATA_SIZE)
    x_to_adv = x_test[to_adv_inds]
    x_to_adv_labs = y_test[to_adv_inds]
    x_plus_noise = x_test[np.random.randint(x_test.shape[0], size=SYNTH_DATA_SIZE)]

    adv_save_num = 15 if x_to_adv.shape[0] >= 15 else x_to_adv.shape[0]

    x_advs_plot = [U.tile_images([x_to_adv[i] for i in range(adv_save_num)], horizontal=False)]

    # label zero for non adverserial input
    x_real_label = [0 for _ in range(SYNTH_DATA_SIZE)]
    x_plus_noise_label = [0 for _ in range(SYNTH_DATA_SIZE)]
    x_adv_label = [1 for _ in range(SYNTH_DATA_SIZE)]

    dists_ls = []
            
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
    #records on succesful values
    fpr_entropies_succ = []
    tpr_entropies_succ = []

    fpr_balds_succ = []
    tpr_balds_succ = []

    prec_entropies_succ = []
    rec_entropies_succ = []

    prec_balds_succ = []
    rec_balds_succ = []

    AUC_entropies_succ = []
    AUC_balds_succ = []

    AP_entropies_succ = []
    AP_balds_succ = []


    accs = []
    modelnames = []
    for i, (name, m) in enumerate(models_to_eval):
        modelnames.append(name)

        input_t = K.placeholder(shape=(None, 224, 224, 3))
        wrap = CallableModelWrapper(m, 'probs') 
        if (not args.use_same_examples) or i == 0:
            attack = attacks.BasicIterativeMethod(wrap,
                    sess=K.get_session(),
                    back='tf')

            attack_method = 'bim'

            adv_tensor = attack.generate(input_t,
                    eps = 15,
                    eps_iter = 3,
                    nb_iter = 10,
                    ord=np.inf,
                    clip_min=-103.939,
                    clip_max=131.32)
            x_adv = batch_eval(adv_tensor, input_t, x_to_adv, batch_size=args.batch_size, verbose="Generating adv examples") 
        
            attack_params = {p: getattr(attack, p) for p in list(attack.feedable_kwargs.keys())
                    if hasattr(attack, p) } 
            attack_params['method'] = attack_method
            attack_params['use_same'] = args.use_same_examples

        #check the examples are really adversarial
        preds = np.concatenate([m.predict(x).argmax(axis=1) for x in batch_gen(x_adv, batch_size=args.batch_size)], axis=0)
        acc = np.mean(np.equal(preds, x_to_adv_labs.argmax(axis=1)))
        print("Accuracy on adv examples:", acc)
        accs.append(acc)
        
        succ_adv_inds = np.logical_not(np.equal(preds, x_to_adv_labs.argmax(axis=1))) #seperate out succesful adv examples

        dists = U.batch_L_norm_distances(x_to_adv, x_adv, ord=2)
        noise = np.random.random(size=x_plus_noise.shape)
        noise /= (dists * np.linalg.norm(noise.reshape(x_plus_noise.shape[0], -1), axis=1))[:, None, None, None]
        x_plus_noise += noise
        x_plus_noise = np.clip(x_plus_noise, 0, 1)
        x_synth = np.concatenate([x_real, x_adv, x_plus_noise])
        y_synth = np.array(x_real_label + x_adv_label + x_plus_noise_label)
        #x_synth, y_synth = shuffle_dataset(x_synth, y_synth) no points
        dists_ls.append(dists) 
        succ_adv_inds = np.concatenate([np.ones(len(x_real_label)), succ_adv_inds, np.ones(len(x_plus_noise_label))]).astype(np.bool)
        # save the adverserial examples to plot
        x_advs_plot.append(U.tile_images([x_adv[i] for i in range(adv_save_num)],
                                    horizontal=False))

        batches = U.batches_generator(x_synth, y_synth, batch_size=args.batch_size)
        # get the entropy and bald on this task
        try: 
            #we can now clean up the adv tensor
            del input_t
            del adv_tensor
        except:
            pass #if these aren't defined, ignore

        entropy = []
        bald = []
        for j,(bx, by) in enumerate(batches):
            print('Evaluating entropy/bald: batch ',j) 
            if hasattr(m, 'get_results'): 
                _, e, b = m.get_results(bx)
            else:
                res = m.predict(bx)
                e = np.sum( - res * np.log(res + 1e-6), axis=1)
                b = np.zeros(e.shape) # undefined
            entropy.append(e)
            bald.append(b)

        entropy = np.concatenate(entropy, axis=0)
        bald = np.concatenate(bald, axis=0)

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
        
        #record stats on succesful adv examples only
        y_synth = y_synth[succ_adv_inds]
        entropy = entropy[succ_adv_inds]
        bald = bald[succ_adv_inds]

        fpr_entropy, tpr_entropy, _ = roc_curve(y_synth, entropy, pos_label=1)
        fpr_bald, tpr_bald, _ = roc_curve(y_synth, bald,    pos_label=1)

        prec_entr, rec_entr, _ = precision_recall_curve(y_synth, entropy, pos_label=1)
        prec_bald, rec_bald , _ = precision_recall_curve(y_synth, bald, pos_label=1)

        AUC_entropy = roc_auc_score(y_synth, entropy)
        AUC_bald = roc_auc_score(y_synth, bald)

        AP_entropy = average_precision_score(y_synth, entropy)
        AP_bald = average_precision_score(y_synth, bald)

        fpr_entropies_succ.append(fpr_entropy)
        tpr_entropies_succ.append(tpr_entropy)

        prec_entropies_succ.append(prec_entr) 
        rec_entropies_succ.append(rec_entr)

        prec_balds_succ.append(prec_bald) 
        rec_balds_succ.append(rec_bald)

        fpr_balds_succ.append(fpr_bald)
        tpr_balds_succ.append(tpr_bald)

        AUC_entropies_succ.append(AUC_entropy)
        AUC_balds_succ.append(AUC_bald)

        AP_entropies_succ.append(AP_entropy)
        AP_balds_succ.append(AP_bald)
 
    fname = U.gen_save_name('save/roc_curves_cats_results_fin_run.h5')
    
    with h5py.File(fname, 'w') as f:
        #record some meta-data in case i forget what i was doing
        f.create_dataset('attack', data=json.dumps(attack_params))
        f.create_dataset('dists', data = np.array(dists_ls))
        f.create_dataset('N_data', data=args.N_data)
        for i,name in enumerate(modelnames):
            g = f.create_group(name)
            g.create_dataset('entropy_fpr', data=fpr_entropies[i])
            g.create_dataset('entropy_tpr', data=tpr_entropies[i])
            g.create_dataset('bald_fpr', data=fpr_balds[i])
            g.create_dataset('bald_tpr', data=tpr_balds[i])
            g.create_dataset('entropy_prec', data=prec_entropies[i])
            g.create_dataset('entropy_rec', data=rec_entropies[i])
            g.create_dataset('bald_prec', data=prec_balds[i])
            g.create_dataset('bald_rec', data=rec_balds[i])
            g.create_dataset('entropy_AUC', data=AUC_entropies[i])
            g.create_dataset('bald_AUC', data=AUC_balds[i])
            g.create_dataset('entropy_AP', data=AP_entropies[i])
            g.create_dataset('bald_AP', data=AP_balds[i])

            g.create_dataset('entropy_fpr_succ', data=fpr_entropies_succ[i])
            g.create_dataset('entropy_tpr_succ', data=tpr_entropies_succ[i])
            g.create_dataset('bald_fpr_succ', data=fpr_balds_succ[i])
            g.create_dataset('bald_tpr_succ', data=tpr_balds_succ[i])
            g.create_dataset('entropy_prec_succ', data=prec_entropies_succ[i])
            g.create_dataset('entropy_rec_succ', data=rec_entropies_succ[i])
            g.create_dataset('bald_prec_succ', data=prec_balds_succ[i])
            g.create_dataset('bald_rec_succ', data=rec_balds_succ[i])
            g.create_dataset('entropy_AUC_succ', data=AUC_entropies_succ[i])
            g.create_dataset('bald_AUC_succ', data=AUC_balds_succ[i])
            g.create_dataset('entropy_AP_succ', data=AP_entropies_succ[i])
            g.create_dataset('bald_AP_succ', data=AP_balds_succ[i])

            g.create_dataset('adv_accuracy', data=accs[i])

        f.create_dataset('example_imgs', data=np.concatenate(x_advs_plot, axis=1))
    
