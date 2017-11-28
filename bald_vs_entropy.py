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

from src.utilities import *
# %%
x_test, y_test, x_train, y_train = get_mnist()


K.set_learning_phase(True)
#load the pre-trained model (trained by another file)
model = load_model('mnist_cnn.h5')

norm = 2
tst = x_test[:10]
tsty = y_test[:10]
n_mc = 50

x = K.placeholder(shape = [None] +  list(x_test.shape[1:]))
mc_preds_tensor = mc_dropout_preds(model, x, n_mc)
mean_entropy_tensor  = m_entropy(mc_preds_tensor)
bald_tensor     = BALD(mc_preds_tensor)
get_output = K.function([x], [mc_preds_tensor,
                              mean_entropy_tensor,
                              bald_tensor])
mc_samples, entropy, bald_acq = get_output([tst])

preds_tensor = K.mean(mc_preds_tensor, axis = 0)

#plot entropy and MI as a eps increases
batches = list(batches_generator(tst, tsty, batch_size = 20))
batches[-1]

entropies = []
balds = []
accs = []
eps = np.linspace(0,10, 50)
preds_tensor = K.mean(mc_preds_tensor, axis = 0)


for ep in eps:
    adv_tensor = fgm(x, preds_tensor, eps = ep, clip_min = 0, clip_max = 1, ord = norm)
    b_entropies = []
    b_balds    = []
    b_accs     = []
    for bx, by in batches:

        adv = adv_tensor.eval(session = K.get_session(), feed_dict = {x: bx})
        mc_samples, e_adv, b_adv = get_output([adv])
        b_entropies.append(e_adv.mean() ) #mean across the batch
        b_balds.append(b_adv.mean())      #ditto
        preds = mc_samples.mean(axis = 0)         #mean across the mc samples per point
        b_accs.append( np.mean(np.equal(preds.argmax(axis=1),
                                        by.argmax(axis=1)) ))
                                        #mean accuracy across the batch
    entropies.append( np.array(b_entropies).mean() ) #mean across all batches
    balds.append(  np.array(b_balds).mean() ) #ditto
    accs.append(  np.array(b_accs).mean())          #ditto


to_plot = [entropies, balds, accs]
names = ['Entropy', 'BALD', 'Accuracy']

def mk_plot(value, name):

    outdir = "output"
    plt.figure()
    plt.plot(eps, value)
    plt.xlabel('FGSM Epsilon')
    plt.ylabel(name)
    figname = "".join(["fgsm_untargeted_",name,".png"])
    plt.savefig(os.path.join(outdir, figname))

for v, n in zip(to_plot, names):
    mk_plot(v, n)
