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


def load_model():
    model = Sequential()

    act_fn = 'relu'
    input_shape = (28,28,1)
    num_classes = 10


    model.add(ConcreteDropout(Conv2D(32, kernel_size=(3,3),
        activation=act_fn),
        input_shape=input_shape))
    model.add(ConcreteDropout(Conv2D(64, (3,3), activation=act_fn)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(ConcreteDropout(Dense(128, activation=act_fn)))
    model.add(ConcreteDropout(Dense(num_classes, activation='softmax')))

    model.load_weights('mnist_cdrop_cnn.h5')
    return model

def eval_perturbations(perturbations, mc_model, batch_size=256): 
    preds = []
    entropies = []
    balds = []
    for a in perturbations:
        zipres = [mc_model.get_results(x) for x in batch_gen(a, batch_size=batch_size)] 
        bpred = np.concatenate([x[0] for x in zipres]) 
        bentropy = np.concatenate([x[1] for x in zipres])
        bbald = np.concatenate([x[2] for x in zipres])
        preds.append(bpred)
        entropies.append(bentropy)
        balds.append(bbald)
    preds = np.array(preds)
    entropies = np.array(entropies)
    balds = np.array(balds)
    return preds, entropies, balds

def batch_gen(array, batch_size=256):
    N = array.shape[0]
    n_batches = N // batch_size + (N % batch_size != 0)
    bs=batch_size
    return (array[i*bs:(i+1)*bs] for i in range(n_batches))

if __name__ == '__main__':
    
    batch_size = 20
    ORD = np.inf 
    epsilons = np.linspace(0,1,40)


    x_train, y_train, x_test, y_test = U.get_mnist()
    x_test = x_test[:50]     
    model = load_model()
    input_tensor = model.input
    mc_model = U.MCModel(model, input_tensor, n_mc = 50 )

    fg = FastGrad(CallableModelWrapper(mc_model, 'probs'), sess=K.get_session())

    adv_grad = fg.generate(input_tensor, ord=ORD)
    
    adv_etas = np.concatenate([adv_grad.eval(session=K.get_session(), feed_dict={input_tensor: x})
                               for x in batch_gen(x_test, batch_size=batch_size)])
    #create adversaral examples

    advs = [x_test + ep * adv_etas for ep in epsilons]
    advs = np.clip(advs, 0,1)
    adv_preds, adv_entropies, adv_balds = eval_perturbations(advs, mc_model, batch_size=batch_size) 
     
    

    #observe the same for a random pertubation of unit norm and step size epsilon 
    eta = np.random.normal(size=x_test.shape)
    #normalise the pertubation
    eta = eta.reshape(x_test.shape[0], -1)
    eta /= np.linalg.norm(eta, axis=1, ord=ORD)[:, None]
    eta = eta.reshape(x_test.shape)
    #apply the pertubations at various step sizes

    perturbs = [x_test + ep * eta for ep in epsilons]
    perturbs = np.clip(perturbs, 0, 1)

    rnd_preds, rnd_entropies, rnd_balds = eval_perturbations(perturbs, mc_model, batch_size=batch_size)

    save_path = U.create_unique_folder(os.path.join('save', 'mnist_pertubations'))

    with open(os.path.join(save_path, 'mnist_pertub_data.pickle'),'wb') as f:
        pickle.dump([epsilons,
                     adv_preds,
                     adv_entropies,
                     adv_balds,
                     rnd_preds,
                     rnd_entropies,
                     rnd_balds], f)
    with open(os.path.join(save_path, 'mnist_data.pickle'), 'wb') as f:
        pickle.dump((x_train, y_train, x_test, y_test), f)
        
