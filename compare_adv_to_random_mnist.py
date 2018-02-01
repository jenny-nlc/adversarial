"""
Preform a similar function to the compare_adv_to_random script, but on mnist.
"""
import keras
import keras.backend as K
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import scale
import src.utilities as U
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from src.concrete_dropout import ConcreteDropout
from cleverhans.model import CallableModelWrapper
from cleverhans import utils_tf
from cleverhans.attacks import FastGradientMethod
import pickle
import os

import tensorflow as tf
import argparse

def fgm_grad(x, preds, y=None, ord=np.inf,
        clip_min=None, clip_max=None,
        targeted=False):
    """
    Returns the scaled gradient of the fgm attack. Clipping now has to be done outside, but this
    avoids costly recomputation of the gradient.
    """

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = tf.reduce_max(preds, 1, keep_dims=True)
        y = tf.to_float(tf.equal(preds, preds_max))
        y = tf.stop_gradient(y)
    y = y / tf.reduce_sum(y, 1, keep_dims=True)

    # Compute loss
    loss = utils_tf.model_loss(y, preds, mean=False)
    if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    if ord == np.inf:
        # Take sign of gradient
        normalized_grad = tf.sign(grad)
        # The following line should not change the numerical results.
        # It applies only because `normalized_grad` is the output of
        # a `sign` op, which has zero derivative anyway.
        # It should not be applied for the other norms, where the
        # perturbation has a non-zero derivative.
        normalized_grad = tf.stop_gradient(normalized_grad)
    elif ord == 1:
        red_ind = list(range(1, len(x.get_shape())))
        normalized_grad = grad / tf.reduce_sum(tf.abs(grad),
                                               reduction_indices=red_ind,
                                               keep_dims=True)
    elif ord == 2:
        red_ind = list(range(1, len(x.get_shape())))
        square = tf.reduce_sum(tf.square(grad),
                               reduction_indices=red_ind,
                               keep_dims=True)
        normalized_grad = grad / tf.sqrt(square)
    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                  "currently implemented.")
    
    return normalized_grad

class FastGrad(FastGradientMethod):
    """
    Re-work of the cleverhans method to just return the gradient, to save recomputing it constantly.
    """
    def __init__(self, model, back='tf', sess=None):
        super(FastGrad, self).__init__(model, back, sess)
        self.feedable_kwargs = {'y': np.float32,
                                'y_target': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32}

    def generate(self, x, **kwargs):
        """
        Replace the generate method of the FastGradientMethod class
        """
        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)

        return fgm_grad(x, self.model.get_probs(x), y=labels,
                   ord=self.ord, clip_min=self.clip_min,
                   clip_max=self.clip_max,
                   targeted=(self.y_target is not None))
    
    def parse_params(self,   y=None,
                     ord=np.inf, clip_min=None, clip_max=None,
                     y_target=None, **kwargs):
        # Save attack-specific parameters
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True

def H(p):
    return - np.sum( p * np.log(p + 1e-10), axis=-1)

def variance_score(mc_p):
    return np.mean(mc_p.var(axis=0), axis=-1) 

def eval_perturbations(perturbations, mc_model, batch_size=256): 
    preds = []
    entropies = []
    balds = []
    var_s = []
    N = len(perturbations)
    for i,a in enumerate(perturbations):
        print('Eps {} of {}'.format(i,N))
        mc_preds = np.concatenate([mc_model.get_mc_preds(x) for x in batch_gen(a, batch_size=batch_size)] , axis=1)
        bpred = mc_preds.mean(axis=0)
        bentropy = H(bpred) 
        bbald = bentropy - H(mc_preds).mean(axis=0)
        bvars = variance_score(mc_preds)
        preds.append(bpred)
        entropies.append(bentropy)
        balds.append(bbald)
        var_s.append(bvars)

    preds = np.array(preds)
    entropies = np.array(entropies)
    balds = np.array(balds)
    var_s = np.array(var_s)
    return preds, entropies, balds, var_s

def batch_gen(array, batch_size=256):
    N = array.shape[0]
    n_batches = N // batch_size + (N % batch_size != 0)
    bs=batch_size
    return (array[i*bs:(i+1)*bs] for i in range(n_batches))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eps_min', type=float, default=0, help="Minimum value \
        of epsilon to generate adverserial examples with FGM")
    parser.add_argument('--eps_max', type=float, default=1, help="Max value of \
        epsilon to generate adverserial examples with")
    parser.add_argument('--N_eps', type=float, default=50, help="Number of values \
        of epsilon to use (linspace eps_min eps_max)")
    parser.add_argument('--N_data', type=int, default=100, help="Number of examples \
        from the test set to use")
    parser.add_argument('--norm', default='inf', help="which norm to use: \
        currently <- {1,2,inf}")
    parser.add_argument('--N_mc', default=50, type=int, help="Number of MC forward \
        passes to use.")
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    args = parser.parse_args()

    if args.norm == 'inf':
        ORD = np.inf
    elif args.norm == '1':
        ORD = 1
    elif args.norm == '2':
        ORD = 2
    else:
        raise NotImplementedError("Norms other than 1,2, inf not implemented")

    batch_size=args.batch_size
    epsilons = np.linspace(args.eps_min, args.eps_max, args.N_eps)


    x_train, y_train, x_test, y_test = U.get_mnist()
    x_test = x_test[:args.N_data]
    y_test = y_test[:args.N_data]
    model = U.load_cdropout_model()
    input_tensor = model.input
    mc_model = U.MCModel(model, input_tensor, n_mc = args.N_mc )

    fg = FastGrad(CallableModelWrapper(mc_model, 'probs'), sess=K.get_session())

    adv_grad = fg.generate(input_tensor, ord=ORD)
    
    adv_etas = np.concatenate([adv_grad.eval(session=K.get_session(), feed_dict={input_tensor: x})
                               for x in batch_gen(x_test, batch_size=batch_size)])
    #create adversaral examples

    advs = [x_test + ep * adv_etas for ep in epsilons]
    advs = np.clip(advs, 0,1)
    print('Calculating adversarial pertubations...')
    adv_preds, adv_entropies, adv_balds, adv_vars = eval_perturbations(advs, mc_model, batch_size=batch_size) 
     
    

    #observe the same for a random pertubation of unit norm and step size epsilon 
    eta = np.random.normal(size=x_test.shape)
    #normalise the pertubation
    eta = eta.reshape(x_test.shape[0], -1)
    eta /= np.linalg.norm(eta, axis=1, ord=ORD)[:, None]
    eta = eta.reshape(x_test.shape)
    #apply the pertubations at various step sizes

    perturbs = [x_test + ep * eta for ep in epsilons]
    perturbs = np.clip(perturbs, 0, 1)

    print('Calculating random pertubations...')
    rnd_preds, rnd_entropies, rnd_balds, rnd_vars = eval_perturbations(perturbs, mc_model, batch_size=batch_size)

    save_path = U.create_unique_folder(os.path.join('save', 'mnist_pertubations_{}_norm_run_'.format(ORD)))

    with open(os.path.join(save_path, 'mnist_pertub_data.pickle'),'wb') as f:
        pickle.dump([epsilons,
                     adv_preds,
                     adv_entropies,
                     adv_balds,
                     adv_vars,
                     rnd_preds,
                     rnd_entropies,
                     rnd_balds,
                     rnd_vars], f)
    with open(os.path.join(save_path, 'mnist_sample.pickle'), 'wb') as f:
        pickle.dump((x_test, y_test), f)

    with open(os.path.join(save_path, 'perturbations.pickle'), 'wb') as f:
        pickle.dump((advs, perturbs), f)
