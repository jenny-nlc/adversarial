from matplotlib import pyplot as plt
import pickle
import sys
import os
import numpy as np
import src.plot_utils as pu

plt.rcParams['figure.figsize'] = 8, 8
#use true type fonts only
plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42 

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit('Please provide load path and norm used')
    load_path = sys.argv[1]
    ORD       = sys.argv[2]
    with open(os.path.join(load_path, 'mnist_pertub_data.pickle'), 'rb') as f:
       epsilons, adv_preds, adv_entropies, adv_balds, rnd_preds, rnd_entropies, rnd_balds = pickle.load(f)
    with open(os.path.join(load_path, 'mnist_sample.pickle'), 'rb') as f:
        x_test, y_test = pickle.load(f)
    fig, axes = plt.subplots(3,1)

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

    plt.show()
