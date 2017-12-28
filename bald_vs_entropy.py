import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from keras.models import load_model
from keras import backend as K
from cleverhans.attacks_tf import fgm
import os
import sys
import src.utilities as U
import argparse
"""
This script calculates adversarial examples using fgm for a range of step-sizes
epsilon, and plots the entropy and the BALD score at all of them.
"""
parser = argparse.ArgumentParser()
parser.add_argument('--eps_min', type=float, default=0.1, help="Minimum value \
    of epsilon to generate adverserial examples with FGM")
parser.add_argument('--eps_max', type=float, default=1, help="Max value of \
    epsilon to generate adverserial examples with")
parser.add_argument('--N_eps', type=float, default=50, help="Number of values \
    of epsilon to use (linspace eps_min eps_max)")
parser.add_argument('--N_data', type=int, default=100, help="Number of examples \
    from the training set to use")
parser.add_argument('--norm', default='inf', help="which norm to use: \
    currently <- {1,2,inf}")
parser.add_argument('--N_mc', default=50, type=int, help="Number of MC forward \
    passes to use.")

args = parser.parse_args()

x_test, y_test, x_train, y_train = U.get_mnist()


K.set_learning_phase(True)
# load the pre-trained model 
# (trained using this script https://github.com/yaringal/acquisition_example)
model = load_model('mnist_cnn.h5')

eps = np.linspace(args.eps_min, args.eps_max, args.N_eps)
if args.norm == 'inf':
    norm = np.inf
elif args.norm == '1':
    norm = 1
elif args.norm == '2':
    norm = 2
else:
    raise NotImplementedError("Norms other than 1,2, inf not implemented")

# could be an idea to shuffle before we do this later on.
tst = x_test[:args.N_data]
tsty = y_test[:args.N_data]
n_mc = args.N_mc

x = K.placeholder(shape=[None] + list(x_test.shape[1:]))
mc_preds_tensor = U.mc_dropout_preds(model, x, n_mc)
entropy_mean_tensor = U.predictive_entropy(mc_preds_tensor)
bald_tensor = U.BALD(mc_preds_tensor)
get_output = K.function([x], [mc_preds_tensor,
                              entropy_mean_tensor,
                              bald_tensor])


# plot entropy and MI as eps increases

entropies = []
balds = []
accs = []

preds_tensor = K.mean(mc_preds_tensor, axis=0)


for i, ep in enumerate(eps):

    print("iteration", i, "of", len(eps), "epsilon", ep)
    sys.stdout.flush()

    adv_tensor = fgm(x, preds_tensor, eps=ep, clip_min=0, clip_max=1, ord=norm)
    b_entropies = []
    b_balds = []
    b_accs = []

    batches = U.batches_generator(tst, tsty, batch_size=500)
    for j, (bx, by) in enumerate(batches):

        print('    batch', j)
        sys.stdout.flush()  # in case we are writing to a log file not stdout

        adv = adv_tensor.eval(session=K.get_session(), feed_dict={x: bx})
        mc_samples, e_adv, b_adv = get_output([adv])
        b_entropies.append(e_adv.mean())  # mean across the batch
        b_balds.append(b_adv.mean())  # ditto
        preds = mc_samples.mean(axis=0)  # mean across the mc samples per point
        b_accs.append(np.mean(np.equal(preds.argmax(axis=1),
                                       by.argmax(axis=1))))
        # mean accuracy across the batch
    entropies.append(np.array(b_entropies).mean())  # mean across all batches
    balds.append(np.array(b_balds).mean())  # ditto
    accs.append(np.array(b_accs).mean())  # ditto


to_plot = [entropies, balds, accs]
names = ['Entropy', 'BALD', 'Accuracy']


def mk_plot(value, name):

    outdir = "output"
    plt.figure()
    plt.plot(eps, value)
    plt.xlabel('FGSM Epsilon')
    plt.ylabel(name)
    figname = "".join(["fgsm_untargeted_", name, ".png"])
    plt.savefig(os.path.join(outdir, figname))


for v, n in zip(to_plot, names):
    mk_plot(v, n)
